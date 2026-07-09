"""PPO : policy gradient actor-critic en numpy pur (IA, apprentissage par renforcement).

Troisième voie d'apprentissage après l'évolution (nn) et la valeur (dqn) :
optimiser DIRECTEMENT la politique par gradient, avec le surrogate clippé de
PPO qui borne chaque pas d'optimisation (ratio politique nouvelle/ancienne
contraint à [1-eps, 1+eps]) — l'algorithme de référence du RL moderne.

Acteur (politique softmax) et critique (valeur d'état) séparés, avantages
estimés par GAE(lambda), bonus d'entropie contre l'effondrement prématuré.
Version 2 : mêmes remèdes que le DQN (voir ANALYSE_IA.md) — récompense
POTENTIELLE (shaping.py) et MÉMOIRE (pile de FRAME_STACK observations) —
pour rester comparable à armes égales avec lui.
"""

from __future__ import annotations

import multiprocessing
import random
from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    FRAME_STACK,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    PPO_BATCH,
    PPO_CLIP,
    PPO_ENTROPY,
    PPO_EPOCHS,
    PPO_GAMMA,
    PPO_HIDDEN_SIZE,
    PPO_ITERATIONS,
    PPO_LAMBDA,
    PPO_LR,
    PPO_MODEL_PATH,
    PPO_ROLLOUT,
    RL_SHAPING,
    VALIDATION_SEEDS,
)
from engine import Engine
from ai_base import BaseAI
from neural import MLP, Layout, softmax
from sensors import FrameStack, SENSOR_FULL_SIZE, sense_full as sense
from shaping import POTENTIALS, base_reward, shaped_reward

POTENTIAL = POTENTIALS.get(RL_SHAPING)  # None si "off" : récompense brute
PPO_INPUT_SIZE: int = NN_INPUT_SIZE * FRAME_STACK
POLICY_LAYOUT: Layout = (PPO_INPUT_SIZE, PPO_HIDDEN_SIZE, NN_OUTPUT_SIZE)
VALUE_LAYOUT: Layout = (PPO_INPUT_SIZE, PPO_HIDDEN_SIZE, 1)


class PPOAI(BaseAI):
    """Politique gloutonne (argmax) sur l'acteur appris.

    La profondeur de mémoire K est déduite du réseau chargé : les anciens
    modèles sans mémoire (K = 1) restent jouables.
    """

    name = "ppo"
    family = "ia"

    def __init__(self, net: MLP) -> None:
        super().__init__()
        self.net = net
        self.stack = FrameStack(max(1, net.layout[0] // SENSOR_FULL_SIZE))

    @classmethod
    def from_file(cls, path: str = PPO_MODEL_PATH) -> "PPOAI":
        return cls(MLP.load(path))

    def get_move(self, game_state: Engine) -> Action:
        logits = self.net.forward(self.stack.push(sense(game_state)))
        return ACTIONS[int(np.argmax(logits))]

    def reset(self) -> None:
        super().reset()
        self.stack.reset()


class PPOTrainer:
    """Boucle rollout -> GAE -> optimisation clippée."""

    def __init__(self, seed: int = 0, world_noise: float = 0.0) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.world_noise = world_noise  # bruit du monde pendant l'entraînement
        self.policy = MLP(POLICY_LAYOUT, seed=seed)
        self.value = MLP(VALUE_LAYOUT, seed=seed + 1)
        self.best_policy = self.policy.copy()
        self.best_score = float("-inf")
        # environnement persistant entre rollouts (+ mémoire et potentiel associés)
        self._env: Engine | None = None
        self._state: np.ndarray | None = None
        self._stack = FrameStack(FRAME_STACK)
        self._phi = 0.0

    # -------------------------------------------------------------- collecte

    def _rollout(self, n_steps: int) -> dict[str, np.ndarray]:
        states = np.empty((n_steps, PPO_INPUT_SIZE))
        actions = np.empty(n_steps, dtype=np.int64)
        rewards = np.empty(n_steps)
        dones = np.empty(n_steps)
        logps = np.empty(n_steps)
        for i in range(n_steps):
            if self._env is None or self._env.game_over:
                self._env = Engine(seed=self.rng.randrange(1_000_000), noise=self.world_noise)
                self._stack.reset()
                self._state = self._stack.push(sense(self._env))
                self._phi = POTENTIAL(self._env) if POTENTIAL else 0.0
            assert self._state is not None
            probs = softmax(self.policy.forward(self._state))
            a = int(self.np_rng.choice(NN_OUTPUT_SIZE, p=probs))
            prev_score = self._env.score
            self._env.step(ACTIONS[a])
            states[i] = self._state
            actions[i] = a
            if POTENTIAL:
                rewards[i], self._phi = shaped_reward(
                    self._env, prev_score, self._phi, PPO_GAMMA, POTENTIAL
                )
            else:
                rewards[i] = base_reward(self._env, prev_score)
            dones[i] = float(self._env.game_over)
            logps[i] = float(np.log(probs[a] + 1e-12))
            self._state = (
                self._stack.push(sense(self._env))
                if not self._env.game_over
                else np.zeros(PPO_INPUT_SIZE)
            )
        return {
            "s": states, "a": actions, "r": rewards, "done": dones, "logp": logps
        }

    def _gae(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Avantages GAE(lambda) et retours cibles du critique."""
        values = self.value.forward(batch["s"]).ravel()
        assert self._state is not None
        bootstrap = (
            0.0
            if batch["done"][-1]
            else float(self.value.forward(self._state).ravel()[0])
        )
        n = len(values)
        adv = np.zeros(n)
        last = 0.0
        for t in range(n - 1, -1, -1):
            next_v = bootstrap if t == n - 1 else values[t + 1]
            non_terminal = 1.0 - batch["done"][t]
            delta = batch["r"][t] + PPO_GAMMA * next_v * non_terminal - values[t]
            last = delta + PPO_GAMMA * PPO_LAMBDA * non_terminal * last
            adv[t] = last
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # ---------------------------------------------------------- optimisation

    def _update(self, batch: dict[str, np.ndarray], adv: np.ndarray, returns: np.ndarray) -> None:
        n = len(adv)
        idx = np.arange(n)
        for _ in range(PPO_EPOCHS):
            self.np_rng.shuffle(idx)
            for start in range(0, n, PPO_BATCH):
                sel = idx[start : start + PPO_BATCH]
                s, a = batch["s"][sel], batch["a"][sel]
                a_adv, old_logp = adv[sel], batch["logp"][sel]
                b = len(sel)
                rows = np.arange(b)

                # --- acteur : surrogate clippé + entropie ---
                logits = self.policy.forward(s, cache=True)
                probs = softmax(logits)
                logp = np.log(probs[rows, a] + 1e-12)
                ratio = np.exp(logp - old_logp)
                # gradient actif seulement là où le min ne clippe pas
                active = np.where(
                    a_adv >= 0, ratio < 1.0 + PPO_CLIP, ratio > 1.0 - PPO_CLIP
                )
                coef = np.where(active, -a_adv * ratio, 0.0)
                onehot = np.zeros_like(probs)
                onehot[rows, a] = 1.0
                grad = coef[:, None] * (onehot - probs)
                # entropie H = -sum p log p ; d(-c*H)/dlogits = c * p * (log p + H)
                logp_all = np.log(probs + 1e-12)
                entropy = -(probs * logp_all).sum(axis=1, keepdims=True)
                grad += PPO_ENTROPY * probs * (logp_all + entropy)
                self.policy.adam_step(self.policy.backward(grad / b), lr=PPO_LR)

                # --- critique : MSE sur les retours ---
                v = self.value.forward(s, cache=True)
                grad_v = (v - returns[sel][:, None]) / b
                self.value.adam_step(self.value.backward(grad_v), lr=PPO_LR)

    # ------------------------------------------------------------ évaluation

    def evaluate(self, seeds: tuple[int, ...] = VALIDATION_SEEDS) -> float:
        scores = []
        for seed in seeds:
            agent = PPOAI(self.policy)  # pile de mémoire neuve par graine
            engine = Engine(seed=seed, noise=self.world_noise)
            while not engine.game_over:
                engine.step(agent.get_move(engine))
            scores.append(engine.score)
        return float(np.mean(scores))

    def train(
        self,
        iterations: int = PPO_ITERATIONS,
        log: Callable[[str], None] = print,
        eval_every: int = 10,
    ) -> tuple[MLP, float]:
        """Retourne (meilleure politique validée, score validé)."""
        for it in range(iterations):
            batch = self._rollout(PPO_ROLLOUT)
            adv, returns = self._gae(batch)
            self._update(batch, adv, returns)
            if (it + 1) % eval_every == 0 or it == iterations - 1:
                val = self.evaluate()
                if val > self.best_score:
                    self.best_score = val
                    self.best_policy = self.policy.copy()
                log(
                    f"iter {it + 1:4d} | récompense/pas {batch['r'].mean():+.3f} | "
                    f"validation {val:5.1f} | meilleur {self.best_score:5.1f}"
                )
        return self.best_policy, self.best_score


# ---------------------------------------------------- entraînement multi-graines

def _train_job(args: tuple[int, float, int]) -> tuple[int, float, np.ndarray]:
    """Un entraînement complet — fonction de module, compatible multiprocessing."""
    seed, world_noise, iterations = args
    trainer = PPOTrainer(seed=seed, world_noise=world_noise)
    net, val = trainer.train(iterations=iterations, log=lambda _s: None)
    return seed, val, net.get_flat()


def train_multi(
    workers: int,
    iterations: int = PPO_ITERATIONS,
    world_noise: float = 0.0,
    base_seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[MLP, float]:
    """`workers` entraînements indépendants en parallèle ; garde le mieux validé.

    Même logique de contrôle de variance que pour le DQN (ANALYSE_IA.md) : la
    graine d'entraînement pèse plus que la plupart des hyperparamètres.
    """
    jobs = [(base_seed + i, world_noise, iterations) for i in range(workers)]
    best: tuple[float, int, np.ndarray] | None = None
    with multiprocessing.Pool(workers) as pool:
        for seed, val, flat in pool.imap_unordered(_train_job, jobs):
            log(f"  graine {seed} : validation {val:.1f}")
            if best is None or val > best[0]:
                best = (val, seed, flat)
    assert best is not None
    log(f"  retenu : graine {best[1]} (validation {best[0]:.1f})")
    return MLP(POLICY_LAYOUT, flat=best[2]), best[0]
