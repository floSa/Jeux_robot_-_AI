"""Neuroévolution : MLP numpy piloté par capteurs + algorithme génétique.

Génotype = concaténation à plat des poids et biais du réseau. Fitness d'un
individu = Y maximal atteint (moyenné sur plusieurs épisodes). Évolution par
élitisme, sélection par tournoi, croisement uniforme et mutation gaussienne.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    CROSSOVER_RATE,
    ELITE_COUNT,
    EPISODES_PER_EVAL,
    GRID_WIDTH,
    MODEL_PATH,
    MUTATION_RATE,
    MUTATION_SIGMA,
    NN_HIDDEN_SIZE,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    NN_SENSOR_ROWS,
    POPULATION_SIZE,
    RIVER,
    ROAD,
    SAFE,
    TOURNAMENT_SIZE,
    VALIDATION_SEEDS,
)
from engine import Engine
from ai_base import BaseAI

# --- Découpage du génome ---
_W1 = NN_INPUT_SIZE * NN_HIDDEN_SIZE
_B1 = NN_HIDDEN_SIZE
_W2 = NN_HIDDEN_SIZE * NN_OUTPUT_SIZE
_B2 = NN_OUTPUT_SIZE
GENOME_SIZE = _W1 + _B1 + _W2 + _B2


# ------------------------------------------------------------------- capteurs

def _scan(x: int, occ: frozenset[int], step: int) -> float:
    """Distance normalisée au premier obstacle dans une direction (1.0 = aucun).

    Balayage circulaire (modulo), cohérent avec la topologie du monde.
    """
    for d in range(1, GRID_WIDTH):
        if (x + step * d) % GRID_WIDTH in occ:
            return d / GRID_WIDTH
    return 1.0


def sense(engine: Engine) -> np.ndarray:
    """Vecteur d'entrée du réseau (taille NN_INPUT_SIZE).

    Par ligne relative (-1, 0, +1, +2, +3) : distance gauche/droite au premier
    obstacle (voiture, tronc ou arbre selon le type), one-hot du type, vitesse
    signée. Puis X normalisé dans [-1, 1] et flag «sur un tronc».
    """
    x, y, tick = engine.player_x, engine.player_y, engine.tick
    features: list[float] = []
    for dy in NN_SENSOR_ROWS:
        ly = y + dy
        if ly < 0:
            features.extend((0.0, 0.0, 1.0, 0.0, 0.0, 0.0))  # bord bas = mur sûr
            continue
        line = engine.line_at(ly)
        occ = (
            engine.tree_columns(ly)
            if line.kind == SAFE
            else engine.occupied_columns(ly, tick)
        )
        features.append(_scan(x, occ, -1))
        features.append(_scan(x, occ, +1))
        features.append(1.0 if line.kind == SAFE else 0.0)
        features.append(1.0 if line.kind == ROAD else 0.0)
        features.append(1.0 if line.kind == RIVER else 0.0)
        features.append(line.direction / line.period)
    features.append(x / (GRID_WIDTH - 1) * 2.0 - 1.0)
    on_log = (
        engine.line_at(y).kind == RIVER and x in engine.occupied_columns(y, tick)
    )
    features.append(1.0 if on_log else 0.0)
    return np.asarray(features, dtype=np.float64)


# --------------------------------------------------------------------- réseau

class NeuralNet:
    """Perceptron multicouche : entrée -> cachée (16, tanh) -> 5 logits."""

    def __init__(self, genome: np.ndarray) -> None:
        g = np.asarray(genome, dtype=np.float64)
        if g.size != GENOME_SIZE:
            raise ValueError(f"génome de taille {g.size}, attendu {GENOME_SIZE}")
        i = 0
        self.w1 = g[i : i + _W1].reshape(NN_INPUT_SIZE, NN_HIDDEN_SIZE)
        i += _W1
        self.b1 = g[i : i + _B1]
        i += _B1
        self.w2 = g[i : i + _W2].reshape(NN_HIDDEN_SIZE, NN_OUTPUT_SIZE)
        i += _W2
        self.b2 = g[i : i + _B2]
        self.genome = g

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(inputs @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """Initialisation par couche en 1/sqrt(fan_in) (régime non saturé du tanh)."""
    w1 = rng.normal(0.0, 1.0 / np.sqrt(NN_INPUT_SIZE), _W1)
    b1 = np.zeros(_B1)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(NN_HIDDEN_SIZE), _W2)
    b2 = np.zeros(_B2)
    return np.concatenate((w1, b1, w2, b2))


class NeuralAI(BaseAI):
    """IA pilotée par le réseau : action = argmax des logits sur les capteurs."""

    name = "nn"

    def __init__(self, net: NeuralNet) -> None:
        super().__init__()
        self.net = net

    @classmethod
    def from_genome(cls, genome: np.ndarray) -> "NeuralAI":
        return cls(NeuralNet(genome))

    @classmethod
    def from_file(cls, path: str = MODEL_PATH) -> "NeuralAI":
        return cls.from_genome(load_genome(path))

    def get_move(self, game_state: Engine) -> Action:
        logits = self.net.forward(sense(game_state))
        return ACTIONS[int(np.argmax(logits))]


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
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.episodes = episodes
        self.population: list[np.ndarray] = [
            random_genome(self.rng) for _ in range(population_size)
        ]
        self.best_genome: np.ndarray = self.population[0].copy()
        self.best_fitness: float = float("-inf")

    # -- fitness ------------------------------------------------------------

    @staticmethod
    def run_episode(genome: np.ndarray, seed: int) -> int:
        """Joue un épisode complet ; retourne le Y maximal atteint."""
        ai = NeuralAI.from_genome(genome)
        engine = Engine(seed=seed)
        while not engine.game_over:
            engine.step(ai.get_move(engine))
        return engine.score

    def fitness(self, genome: np.ndarray, seeds: tuple[int, ...]) -> float:
        return float(np.mean([self.run_episode(genome, s) for s in seeds]))

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
    ) -> tuple[np.ndarray, float]:
        """Fait évoluer la population ; retourne (meilleur génome, fitness validée)."""
        for gen in range(generations):
            seeds = tuple(int(s) for s in self.rng.integers(0, 100_000, self.episodes))
            fits = np.array([self.fitness(g, seeds) for g in self.population])
            order = np.argsort(fits)[::-1]
            champion = self.population[order[0]]

            val_fit = self.fitness(champion, VALIDATION_SEEDS)
            if val_fit > self.best_fitness:
                self.best_fitness = val_fit
                self.best_genome = champion.copy()

            log(
                f"gen {gen:3d} | fitness max {fits[order[0]]:6.1f} | "
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
