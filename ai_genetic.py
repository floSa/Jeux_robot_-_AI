"""Neuroévolution : MLP numpy piloté par capteurs + algorithme génétique.

Génotype = concaténation à plat des poids et biais du réseau. Fitness d'un
individu = Y maximal atteint (moyenné sur plusieurs épisodes). Évolution par
élitisme, sélection par tournoi, croisement uniforme et mutation gaussienne.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    CROSSOVER_RATE,
    CURRICULUM_FRACTION,
    CURRICULUM_WEIGHTS,
    ELITE_COUNT,
    EPISODES_PER_EVAL,
    FITNESS_SURVIVAL_BONUS,
    MODEL_PATH,
    MUTATION_RATE,
    MUTATION_SIGMA,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    POPULATION_SIZE,
    TOURNAMENT_SIZE,
    TRAIN_TEMPERATURE,
    VALIDATION_SEEDS,
)
from engine import Engine
from ai_base import BaseAI
from neural import MLP, Layout, genome_size, random_flat, softmax
from sensors import sense_full as sense

LAYOUT: Layout = (NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
GENOME_SIZE = genome_size(LAYOUT)


class NeuralAI(BaseAI):
    """IA pilotée par le réseau : argmax des logits, ou softmax à température.

    temperature = 0 : politique déterministe (évaluation, bench).
    temperature > 0 : échantillonnage softmax (exploration à l'entraînement,
    évite les boucles comportementales des politiques argmax figées).
    """

    name = "nn"
    family = "ia"

    def __init__(self, net: MLP, temperature: float = 0.0, seed: int = 0) -> None:
        super().__init__()
        self.net = net
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_genome(cls, genome: np.ndarray, **kwargs: float) -> "NeuralAI":
        return cls(MLP(LAYOUT, flat=np.asarray(genome)), **kwargs)

    @classmethod
    def from_file(cls, path: str = MODEL_PATH) -> "NeuralAI":
        return cls.from_genome(load_genome(path))

    def get_move(self, game_state: Engine) -> Action:
        logits = self.net.forward(sense(game_state))
        if self.temperature > 0.0:
            probs = softmax(logits / self.temperature)
            return ACTIONS[int(self.rng.choice(len(ACTIONS), p=probs))]
        return ACTIONS[int(np.argmax(logits))]


def _eval_genome(
    args: tuple[np.ndarray, tuple[int, ...], float, dict[int, float] | None, bool, float],
) -> float:
    """Évaluation d'un génome — fonction de module, compatible multiprocessing."""
    genome, seeds, temperature, line_weights, shaped, world_noise = args
    total = 0.0
    for s in seeds:
        score, ticks = GeneticTrainer.run_episode(
            genome, s, temperature, line_weights, world_noise
        )
        total += score + (FITNESS_SURVIVAL_BONUS * ticks if shaped else 0.0)
    return total / len(seeds)


# ------------------------------------------------------------- persistance

def save_genome(genome: np.ndarray, path: str = MODEL_PATH) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, genome=genome)


def load_genome(path: str = MODEL_PATH) -> np.ndarray:
    with np.load(path) as data:
        return data["genome"]


# ----------------------------------------------------------------- évolution

class GeneticTrainer:
    """Algorithme génétique sur les génomes du réseau.

    Chaque génération est évaluée sur des graines tirées au sort (anti-surapprentissage) ;
    le champion est revalidé sur des graines fixes pour suivre un meilleur global comparable.
    """

    def __init__(
        self,
        population_size: int = POPULATION_SIZE,
        episodes: int = EPISODES_PER_EVAL,
        seed: int = 0,
        world_noise: float = 0.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.episodes = episodes
        self.world_noise = world_noise  # bruit du monde pendant l'entraînement
        self.population: list[np.ndarray] = [
            random_flat(LAYOUT, self.rng) for _ in range(population_size)
        ]
        self.best_genome: np.ndarray = self.population[0].copy()
        self.best_fitness: float = float("-inf")

    # -- fitness ------------------------------------------------------------

    @staticmethod
    def run_episode(
        genome: np.ndarray,
        seed: int,
        temperature: float = 0.0,
        line_weights: dict[int, float] | None = None,
        world_noise: float = 0.0,
    ) -> tuple[int, int]:
        """Joue un épisode complet ; retourne (Y maximal, ticks survécus)."""
        ai = NeuralAI.from_genome(genome, temperature=temperature, seed=seed)
        engine = Engine(seed=seed, line_weights=line_weights, noise=world_noise)
        while not engine.game_over:
            engine.step(ai.get_move(engine))
        return engine.score, engine.tick

    def fitness(
        self,
        genome: np.ndarray,
        seeds: tuple[int, ...],
        temperature: float = 0.0,
        line_weights: dict[int, float] | None = None,
        shaped: bool = True,
    ) -> float:
        """Fitness = Y max + bonus de survie (shaping), moyennée sur les graines.

        Le bonus (0,005/tick) reste très inférieur à une ligne franchie : il
        densifie le signal en début d'évolution sans récompenser le camping.
        """
        return _eval_genome(
            (genome, seeds, temperature, line_weights, shaped, self.world_noise)
        )

    # -- opérateurs génétiques ----------------------------------------------

    def _tournament(self, fitnesses: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(self.population), TOURNAMENT_SIZE)
        winner = idx[int(np.argmax(fitnesses[idx]))]
        return self.population[winner]

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > CROSSOVER_RATE:
            return a.copy()
        mask = self.rng.random(GENOME_SIZE) < 0.5
        return np.where(mask, a, b)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mask = self.rng.random(GENOME_SIZE) < MUTATION_RATE
        noise = self.rng.normal(0.0, MUTATION_SIGMA, int(mask.sum()))
        genome[mask] += noise
        return genome

    # -- boucle d'évolution ---------------------------------------------------

    def evolve(
        self,
        generations: int,
        log: Callable[[str], None] = print,
        curriculum: bool = False,
        workers: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Fait évoluer la population ; retourne (meilleur génome, score validé).

        curriculum=True : les premières générations (CURRICULUM_FRACTION)
        s'entraînent sur des mondes sans rivière (routes seules), puis mixtes.
        L'entraînement échantillonne en softmax (TRAIN_TEMPERATURE) ; la
        validation est toujours en argmax, mondes complets, score brut.
        workers > 1 : évaluation de la population en parallèle (résultats
        strictement identiques au séquentiel, tout l'aléa est tiré ici).
        """
        pool = multiprocessing.Pool(workers) if workers > 1 else None
        try:
            return self._evolve(generations, log, curriculum, pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    def _evolve(
        self,
        generations: int,
        log: Callable[[str], None],
        curriculum: bool,
        pool: "multiprocessing.pool.Pool | None",
    ) -> tuple[np.ndarray, float]:
        cutoff = int(generations * CURRICULUM_FRACTION) if curriculum else 0
        for gen in range(generations):
            weights = CURRICULUM_WEIGHTS if gen < cutoff else None
            seeds = tuple(int(s) for s in self.rng.integers(0, 100_000, self.episodes))
            jobs = [
                (g, seeds, TRAIN_TEMPERATURE, weights, True, self.world_noise)
                for g in self.population
            ]
            if pool is not None:
                fits = np.array(pool.map(_eval_genome, jobs, chunksize=4))
            else:
                fits = np.array([_eval_genome(j) for j in jobs])
            order = np.argsort(fits)[::-1]
            champion = self.population[order[0]]

            val_fit = self.fitness(
                champion, VALIDATION_SEEDS, temperature=0.0, shaped=False
            )
            if val_fit > self.best_fitness:
                self.best_fitness = val_fit
                self.best_genome = champion.copy()

            phase = "route" if gen < cutoff else "mixte"
            log(
                f"gen {gen:3d} [{phase}] | fitness max {fits[order[0]]:6.1f} | "
                f"moyenne {fits.mean():6.1f} | validation {val_fit:6.1f} | "
                f"meilleur global {self.best_fitness:6.1f}"
            )

            elites = [self.population[i].copy() for i in order[:ELITE_COUNT]]
            offspring = [
                self._mutate(self._crossover(self._tournament(fits), self._tournament(fits)))
                for _ in range(len(self.population) - ELITE_COUNT)
            ]
            self.population = elites + offspring
        return self.best_genome, self.best_fitness
