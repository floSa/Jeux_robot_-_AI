"""DQN : Q-learning profond en numpy pur (IA, apprentissage par renforcement).

Différence fondamentale avec la neuroévolution : au lieu d'un signal terminal
(Y max en fin de partie), le réseau apprend une fonction de VALEUR Q(s, a) sur
une récompense dense, par équation de Bellman :
Q(s, a) <- r + gamma^n * max_a' Q_cible(s', a').

Cinq briques issues de la littérature sont implémentées et PILOTÉES PAR LA
CONFIG, chacune visant une cause identifiée dans ANALYSE_IA.md :
- récompense POTENTIELLE (RL_SHAPING, shaping.py) : signal immédiat pour
  entrer sur un tronc ou esquiver une voiture imminente (cause 2) ;
- MÉMOIRE (FRAME_STACK > 1) : pile des K dernières observations (cause 3) ;
- DOUBLE DQN (DQN_DOUBLE) : l'action suivante est choisie par le réseau en
  ligne mais évaluée par le réseau cible, contre la surestimation (cause 1) ;
- retours MULTI-PAS (DQN_NSTEP > 1) : le crédit remonte n fois plus vite (cause 2) ;
- CURRICULUM rivières (DQN_CURRICULUM_PROB > 0) : sur-échantillonner la
  sous-tâche la plus dure à l'entraînement (cause 2).

Verdict d'ablation (voir ANALYSE_IA.md) : à notre budget (~10^5 pas), aucune
de ces briques ne bat la recette simple — les défauts de config reflètent la
mesure, pas la théorie. Le levier qui marche : entraînement multi-graines
(`train_multi`) avec sélection sur validation, car l'issue varie du simple
au double selon la graine.

Ingrédients classiques : replay en tampon circulaire numpy (décorrélation,
échantillonnage O(batch)), réseau cible synchronisé (stabilité),
epsilon-greedy décroissant (exploration).
"""

from __future__ import annotations

import multiprocessing
import random
from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_CURRICULUM_PROB,
    DQN_CURRICULUM_WEIGHTS,
    DQN_DOUBLE,
    DQN_EPS_DECAY,
    DQN_EPS_END,
    DQN_EPS_START,
    DQN_EPISODES,
    DQN_GAMMA,
    DQN_HIDDEN_SIZE,
    DQN_LR,
    DQN_MODEL_PATH,
    DQN_NSTEP,
    DQN_TARGET_SYNC,
    DQN_TRAIN_EVERY,
    FRAME_STACK,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    RL_SHAPING,
    VALIDATION_SEEDS,
)
from engine import Engine
from ai_base import BaseAI
from neural import MLP, Layout
from sensors import FrameStack, SENSOR_FULL_SIZE, sense_full as sense
from shaping import base_reward, potential, shaped_reward

DQN_LAYOUT: Layout = (NN_INPUT_SIZE * FRAME_STACK, DQN_HIDDEN_SIZE, NN_OUTPUT_SIZE)


class DQNAI(BaseAI):
    """Politique gloutonne sur la fonction Q apprise.

    La profondeur de mémoire K est déduite du réseau chargé (entrée / taille
    des capteurs) : les anciens modèles sans mémoire (K = 1) restent jouables.
    """

    name = "dqn"
    family = "ia"

    def __init__(self, net: MLP) -> None:
        super().__init__()
        self.net = net
        self.stack = FrameStack(max(1, net.layout[0] // SENSOR_FULL_SIZE))

    @classmethod
    def from_file(cls, path: str = DQN_MODEL_PATH) -> "DQNAI":
        return cls(MLP.load(path))

    def get_move(self, game_state: Engine) -> Action:
        q = self.net.forward(self.stack.push(sense(game_state)))
        return ACTIONS[int(np.argmax(q))]

    def reset(self) -> None:
        super().reset()
        self.stack.reset()


class Replay:
    """Tampon circulaire numpy : insertion et échantillonnage sans copie de liste."""

    def __init__(self, capacity: int, input_size: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.s = np.zeros((capacity, input_size), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, input_size), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._next = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._next

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        i = self._next
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, float(done)
        self._next = (i + 1) % self.capacity
        self._full = self._full or self._next == 0

    def sample(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, len(self), n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


class DQNTrainer:
    """Boucle d'apprentissage : collecte epsilon-greedy + descente TD."""

    def __init__(self, seed: int = 0, world_noise: float = 0.0) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.world_noise = world_noise  # bruit du monde pendant l'entraînement
        self.input_size = DQN_LAYOUT[0]
        self.online = MLP(DQN_LAYOUT, seed=seed)
        self.target = self.online.copy()
        self.buffer = Replay(DQN_BUFFER_SIZE, self.input_size, self.np_rng)
        self.best_net = self.online.copy()
        self.best_score = float("-inf")
        self.step_count = 0

    # ------------------------------------------------------------ interaction

    def _epsilon_greedy(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(NN_OUTPUT_SIZE)
        return int(np.argmax(self.online.forward(state)))

    def _new_episode_env(self) -> Engine:
        """Monde standard, ou « rivières denses » pour une part des épisodes.

        Curriculum ciblé sur la sous-tâche la plus dure : l'agent voit
        beaucoup plus de traversées de rivière que le hasard n'en offrirait.
        La validation, elle, reste toujours sur mondes standards.
        """
        weights = (
            DQN_CURRICULUM_WEIGHTS
            if self.rng.random() < DQN_CURRICULUM_PROB
            else None
        )
        return Engine(
            seed=self.rng.randrange(1_000_000),
            line_weights=weights,
            noise=self.world_noise,
        )

    def _flush_nstep(
        self,
        pending: list[tuple[np.ndarray, int, float]],
        next_state: np.ndarray,
        done: bool,
        full_only: bool,
    ) -> None:
        """Émet vers le replay les transitions n-pas prêtes (fenêtre pleine ou fin)."""
        while pending and (len(pending) == DQN_NSTEP or not full_only):
            ret = 0.0
            for i, (_s, _a, r) in enumerate(pending):
                ret += (DQN_GAMMA**i) * r
            s0, a0, _ = pending.pop(0)
            self.buffer.push(s0, a0, ret, next_state, done)
            if full_only:
                break

    # ------------------------------------------------------------- optimisation

    def _learn_step(self, gamma_n: float) -> None:
        s, a, r, s2, done = self.buffer.sample(DQN_BATCH_SIZE)
        rows = np.arange(len(a))
        if DQN_DOUBLE:
            # Double DQN : l'en-ligne choisit, la cible évalue (anti-surestimation)
            a_next = np.argmax(self.online.forward(s2), axis=1)
            q_next = self.target.forward(s2)[rows, a_next]
        else:
            q_next = self.target.forward(s2).max(axis=1)
        td_target = r + gamma_n * (1.0 - done) * q_next
        q = self.online.forward(s, cache=True)
        # perte MSE sur la seule action jouée : gradient nul ailleurs
        grad = np.zeros_like(q)
        grad[rows, a] = 2.0 * (q[rows, a] - td_target) / len(a)
        self.online.adam_step(self.online.backward(grad), lr=DQN_LR)

    # ------------------------------------------------------------ entraînement

    def evaluate(self, seeds: tuple[int, ...] = VALIDATION_SEEDS) -> float:
        """Score moyen (Y max) de la politique gloutonne sur des graines fixes."""
        scores = []
        for seed in seeds:
            agent = DQNAI(self.online)  # pile de mémoire neuve par graine
            engine = Engine(seed=seed, noise=self.world_noise)
            while not engine.game_over:
                engine.step(agent.get_move(engine))
            scores.append(engine.score)
        return float(np.mean(scores))

    def train(
        self,
        episodes: int = DQN_EPISODES,
        log: Callable[[str], None] = print,
        eval_every: int = 100,
    ) -> tuple[MLP, float]:
        """Retourne (meilleur réseau validé, score validé)."""
        epsilon = DQN_EPS_START
        gamma_n = DQN_GAMMA**DQN_NSTEP
        recent: list[int] = []
        for ep in range(episodes):
            engine = self._new_episode_env()
            stack = FrameStack(FRAME_STACK)
            state = stack.push(sense(engine)).astype(np.float32)
            phi = potential(engine)
            pending: list[tuple[np.ndarray, int, float]] = []
            while not engine.game_over:
                action = self._epsilon_greedy(state, epsilon)
                prev_score = engine.score
                engine.step(ACTIONS[action])
                if RL_SHAPING:
                    reward, phi = shaped_reward(engine, prev_score, phi, DQN_GAMMA)
                else:
                    reward = base_reward(engine, prev_score)
                done = engine.game_over
                next_state = (
                    stack.push(sense(engine)).astype(np.float32)
                    if not done
                    else np.zeros(self.input_size, dtype=np.float32)
                )
                pending.append((state, action, reward))
                self._flush_nstep(pending, next_state, done, full_only=not done)
                state = next_state
                self.step_count += 1
                if (
                    len(self.buffer) >= DQN_BATCH_SIZE
                    and self.step_count % DQN_TRAIN_EVERY == 0
                ):
                    self._learn_step(gamma_n)
                if self.step_count % DQN_TARGET_SYNC == 0:
                    self.target = self.online.copy()
            recent.append(engine.score)
            if len(recent) > 25:
                recent.pop(0)
            epsilon = max(DQN_EPS_END, epsilon * DQN_EPS_DECAY)

            if (ep + 1) % eval_every == 0 or ep == episodes - 1:
                val = self.evaluate()
                if val > self.best_score:
                    self.best_score = val
                    self.best_net = self.online.copy()
                log(
                    f"ep {ep + 1:5d} | eps {epsilon:.2f} | score moyen (25 ep) "
                    f"{np.mean(recent):5.1f} | validation {val:5.1f} | "
                    f"meilleur {self.best_score:5.1f} | buffer {len(self.buffer)}"
                )
        return self.best_net, self.best_score


# ---------------------------------------------------- entraînement multi-graines

def _train_job(args: tuple[int, float, int]) -> tuple[int, float, np.ndarray]:
    """Un entraînement complet — fonction de module, compatible multiprocessing."""
    seed, world_noise, episodes = args
    trainer = DQNTrainer(seed=seed, world_noise=world_noise)
    net, val = trainer.train(episodes=episodes, log=lambda _s: None)
    return seed, val, net.get_flat()


def train_multi(
    workers: int,
    episodes: int = DQN_EPISODES,
    world_noise: float = 0.0,
    base_seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[MLP, float]:
    """`workers` entraînements indépendants en parallèle ; garde le mieux validé.

    L'issue d'un entraînement par renforcement est très sensible à la graine
    (initialisation du réseau + trajectoire d'exploration) : la même recette
    donne du simple au triple d'une graine à l'autre. Entraîner plusieurs
    graines et sélectionner sur la validation lisse cette variance — c'est le
    levier le plus rentable mesuré sur ce projet (voir ANALYSE_IA.md).
    """
    jobs = [(base_seed + i, world_noise, episodes) for i in range(workers)]
    best: tuple[float, int, np.ndarray] | None = None
    with multiprocessing.Pool(workers) as pool:
        for seed, val, flat in pool.imap_unordered(_train_job, jobs):
            log(f"  graine {seed} : validation {val:.1f}")
            if best is None or val > best[0]:
                best = (val, seed, flat)
    assert best is not None
    log(f"  retenu : graine {best[1]} (validation {best[0]:.1f})")
    return MLP(DQN_LAYOUT, flat=best[2]), best[0]
