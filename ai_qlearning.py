"""DQN : Q-learning profond en numpy pur (IA, apprentissage par renforcement).

Différence fondamentale avec la neuroévolution : au lieu d'un signal terminal
(Y max en fin de partie), le réseau apprend une fonction de VALEUR Q(s, a) sur
une récompense dense (chaque ligne franchie, chaque tick, chaque mort), par
équation de Bellman : Q(s, a) <- r + gamma * max_a' Q_cible(s', a').

Ingrédients classiques : replay buffer (décorrélation des transitions),
réseau cible synchronisé périodiquement (stabilité), epsilon-greedy décroissant
(exploration). Mêmes capteurs (42) que la neuroévolution : comparaison à
armes égales entre gradient et évolution.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_EPS_DECAY,
    DQN_EPS_END,
    DQN_EPS_START,
    DQN_EPISODES,
    DQN_GAMMA,
    DQN_HIDDEN_SIZE,
    DQN_LR,
    DQN_MODEL_PATH,
    DQN_TARGET_SYNC,
    DQN_TRAIN_EVERY,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    REWARD_DEATH,
    REWARD_PROGRESS,
    REWARD_STEP,
    REWARD_WIN,
    VALIDATION_SEEDS,
)
from engine import Engine
from ai_base import BaseAI
from neural import MLP, Layout
from sensors import sense_full as sense

DQN_LAYOUT: Layout = (NN_INPUT_SIZE, DQN_HIDDEN_SIZE, NN_OUTPUT_SIZE)

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class DQNAI(BaseAI):
    """Politique gloutonne sur la fonction Q apprise."""

    name = "dqn"
    family = "ia"

    def __init__(self, net: MLP) -> None:
        super().__init__()
        self.net = net

    @classmethod
    def from_file(cls, path: str = DQN_MODEL_PATH) -> "DQNAI":
        return cls(MLP.load(path))

    def get_move(self, game_state: Engine) -> Action:
        q = self.net.forward(sense(game_state))
        return ACTIONS[int(np.argmax(q))]


class DQNTrainer:
    """Boucle d'apprentissage : collecte epsilon-greedy + descente TD."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.online = MLP(DQN_LAYOUT, seed=seed)
        self.target = self.online.copy()
        self.buffer: deque[Transition] = deque(maxlen=DQN_BUFFER_SIZE)
        self.best_net = self.online.copy()
        self.best_score = float("-inf")
        self.step_count = 0

    # ------------------------------------------------------------ interaction

    def _epsilon_greedy(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(NN_OUTPUT_SIZE)
        return int(np.argmax(self.online.forward(state)))

    @staticmethod
    def _reward(engine: Engine, prev_score: int) -> float:
        r = REWARD_STEP + REWARD_PROGRESS * (engine.score - prev_score)
        if engine.won:
            r += REWARD_WIN
        elif not engine.alive:
            r += REWARD_DEATH
        return r

    # ------------------------------------------------------------- optimisation

    def _learn_step(self) -> None:
        batch = self.rng.sample(list(self.buffer), DQN_BATCH_SIZE)
        s = np.stack([t[0] for t in batch])
        a = np.array([t[1] for t in batch])
        r = np.array([t[2] for t in batch])
        s2 = np.stack([t[3] for t in batch])
        done = np.array([t[4] for t in batch], dtype=np.float64)

        q_next = self.target.forward(s2).max(axis=1)
        td_target = r + DQN_GAMMA * (1.0 - done) * q_next
        q = self.online.forward(s, cache=True)
        idx = np.arange(len(batch))
        # perte MSE sur la seule action jouée : gradient nul ailleurs
        grad = np.zeros_like(q)
        grad[idx, a] = 2.0 * (q[idx, a] - td_target) / len(batch)
        self.online.adam_step(self.online.backward(grad), lr=DQN_LR)

    # ------------------------------------------------------------ entraînement

    def evaluate(self, seeds: tuple[int, ...] = VALIDATION_SEEDS) -> float:
        """Score moyen (Y max) de la politique gloutonne sur des graines fixes."""
        agent = DQNAI(self.online)
        scores = []
        for seed in seeds:
            engine = Engine(seed=seed)
            while not engine.game_over:
                engine.step(agent.get_move(engine))
            scores.append(engine.score)
        return float(np.mean(scores))

    def train(
        self,
        episodes: int = DQN_EPISODES,
        log: Callable[[str], None] = print,
        eval_every: int = 50,
    ) -> tuple[MLP, float]:
        """Retourne (meilleur réseau validé, score validé)."""
        epsilon = DQN_EPS_START
        recent: deque[int] = deque(maxlen=25)
        for ep in range(episodes):
            engine = Engine(seed=self.rng.randrange(1_000_000))
            state = sense(engine)
            while not engine.game_over:
                action = self._epsilon_greedy(state, epsilon)
                prev_score = engine.score
                engine.step(ACTIONS[action])
                reward = self._reward(engine, prev_score)
                next_state = sense(engine) if not engine.game_over else np.zeros(NN_INPUT_SIZE)
                self.buffer.append(
                    (state, action, reward, next_state, engine.game_over)
                )
                state = next_state
                self.step_count += 1
                if (
                    len(self.buffer) >= DQN_BATCH_SIZE
                    and self.step_count % DQN_TRAIN_EVERY == 0
                ):
                    self._learn_step()
                if self.step_count % DQN_TARGET_SYNC == 0:
                    self.target = self.online.copy()
            recent.append(engine.score)
            epsilon = max(DQN_EPS_END, epsilon * DQN_EPS_DECAY)

            if (ep + 1) % eval_every == 0 or ep == episodes - 1:
                val = self.evaluate()
                if val > self.best_score:
                    self.best_score = val
                    self.best_net = self.online.copy()
                log(
                    f"ep {ep + 1:4d} | eps {epsilon:.2f} | score moyen (25 ep) "
                    f"{np.mean(recent):5.1f} | validation {val:5.1f} | "
                    f"meilleur {self.best_score:5.1f} | buffer {len(self.buffer)}"
                )
        return self.best_net, self.best_score
