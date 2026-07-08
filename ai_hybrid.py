"""Imitation du planificateur (IA) et recherche guidée par politique (hybride).

Idée (patron AlphaZero en miniature) : la recherche exhaustive `search` est un
expert parfait mais coûteux en nœuds ; on la DISTILLE dans un petit réseau.

1. Dataset : on rejoue des parties de `search` et on enregistre, à chaque tick,
   (capteurs complets de l'état, coup choisi par l'expert).
2. Politique : MLP entraîné par entropie croisée à prédire le coup de l'expert.
3. `clone` (IA) : joue directement la politique — un réflexe distillé, sans
   aucune recherche au moment de jouer.
4. `guided` (hybride) : la recherche A* garde sa borne admissible, mais le
   départage des ex æquo près de la racine suit la politique au lieu de la
   distance au centre. Objectif mesuré : réduire les nœuds explorés. Le
   résultat, positif ou négatif, est documenté dans ROBOTS.md.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from config import (
    ACTIONS,
    Action,
    CENTER_X,
    GUIDED_PRIOR_DEPTH,
    IMITATION_BATCH,
    IMITATION_EPOCHS,
    IMITATION_LR,
    IMITATION_SAMPLES,
    NN_INPUT_SIZE,
    NN_OUTPUT_SIZE,
    POLICY_HIDDEN_SIZE,
    POLICY_MODEL_PATH,
    SEARCH_HORIZON,
)
from engine import Engine
from ai_base import BaseAI
from ai_search import SearchAI
from neural import MLP, Layout, softmax
from sensors import sense_full as sense

POLICY_LAYOUT: Layout = (NN_INPUT_SIZE, POLICY_HIDDEN_SIZE, NN_OUTPUT_SIZE)


# ------------------------------------------------------------------- dataset

def collect_dataset(
    samples: int = IMITATION_SAMPLES,
    base_seed: int = 20_000,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejoue l'expert `search` et collecte (capteurs, coup) à chaque tick."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    expert = SearchAI()
    seed = base_seed
    while len(xs) < samples:
        engine = Engine(seed=seed)
        while not engine.game_over and len(xs) < samples:
            move = expert.get_move(engine)
            xs.append(sense(engine))
            ys.append(ACTIONS.index(move))
            engine.step(move)
        seed += 1
        if seed % 50 == 0:
            log(f"  collecte : {len(xs)}/{samples} exemples ({seed - base_seed} parties)")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    counts = np.bincount(y, minlength=NN_OUTPUT_SIZE)
    log(
        "  répartition des coups expert : "
        + ", ".join(f"{n}" for n in counts)
        + f" (sur {len(y)})"
    )
    return x, y


# ---------------------------------------------------------------- imitation

def balanced_accuracy(net: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Rappel moyen par classe (hasard = 0,2) : insensible au déséquilibre."""
    pred = net.forward(x).argmax(axis=1)
    recalls = [
        float((pred[y == c] == c).mean())
        for c in range(NN_OUTPUT_SIZE)
        if np.any(y == c)
    ]
    return float(np.mean(recalls))


def train_policy(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = IMITATION_EPOCHS,
    lr: float = IMITATION_LR,
    batch: int = IMITATION_BATCH,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[MLP, float]:
    """Entropie croisée pondérée sur (capteurs -> coup expert).

    Le dataset est très déséquilibré (l'expert AVANCE ~3 coups sur 4) or les
    coups rares — attendre devant une voiture, pas de côté vers un tronc —
    sont les coups vitaux : la perte est pondérée en inverse-fréquence des
    classes. Retourne (réseau, précision de validation).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]
    net = MLP(POLICY_LAYOUT, seed=seed)

    counts = np.bincount(y[train_idx], minlength=NN_OUTPUT_SIZE).astype(np.float64)
    # racine de l'inverse-fréquence : la pondération pleine sur-corrige (politique
    # qui n'avance plus), l'absence de pondération s'effondre sur la classe majoritaire
    class_w = np.sqrt(len(train_idx) / (NN_OUTPUT_SIZE * np.maximum(counts, 1.0)))
    class_w /= class_w.mean()

    def accuracy(sel: np.ndarray) -> float:
        pred = net.forward(x[sel]).argmax(axis=1)
        return float((pred == y[sel]).mean())

    for epoch in range(epochs):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), batch):
            sel = train_idx[start : start + batch]
            logits = net.forward(x[sel], cache=True)
            probs = softmax(logits)
            onehot = np.zeros_like(probs)
            rows = np.arange(len(sel))
            onehot[rows, y[sel]] = 1.0
            w = class_w[y[sel]]
            losses.append(float((-np.log(probs[rows, y[sel]] + 1e-12) * w).mean()))
            net.adam_step(net.backward((probs - onehot) * w[:, None] / len(sel)), lr=lr)
        log(
            f"  epoch {epoch + 1:2d}/{epochs} | perte pondérée {np.mean(losses):.4f} | "
            f"précision val {accuracy(val_idx) * 100:.1f} % | "
            f"équilibrée {balanced_accuracy(net, x[val_idx], y[val_idx]) * 100:.1f} %"
        )
    return net, accuracy(val_idx)


# ------------------------------------------------------------------- agents

class CloneAI(BaseAI):
    """IA par imitation : le réflexe distillé du planificateur, sans recherche."""

    name = "clone"
    family = "ia"

    def __init__(self, net: MLP) -> None:
        super().__init__()
        self.net = net

    @classmethod
    def from_file(cls, path: str = POLICY_MODEL_PATH) -> "CloneAI":
        return cls(MLP.load(path))

    def get_move(self, game_state: Engine) -> Action:
        logits = self.net.forward(sense(game_state))
        return ACTIONS[int(np.argmax(logits))]


class GuidedSearchAI(SearchAI):
    """Hybride : A* admissible + départage par la politique près de la racine.

    Aux profondeurs <= GUIDED_PRIOR_DEPTH, les ex æquo de la borne f sont
    ordonnés par (1 - p_politique(action)) au lieu de la distance au centre ;
    au-delà, départage classique (le coût d'un forward par nœud profond
    dépasserait le gain, la recherche entière coûtant ~0,3 ms).
    """

    name = "guided"
    family = "hybride"

    def __init__(self, net: MLP, horizon: int = SEARCH_HORIZON) -> None:
        super().__init__(horizon)
        self.net = net
        self._prior_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._engine: Engine | None = None

    @classmethod
    def from_file(cls, path: str = POLICY_MODEL_PATH) -> "GuidedSearchAI":
        return cls(MLP.load(path))

    def _tie(self, px: int, py: int, depth: int, action_idx: int, child_x: int) -> float:
        if depth > GUIDED_PRIOR_DEPTH:
            return float(abs(child_x - CENTER_X))
        assert self._engine is not None
        key = (px, py, depth)
        probs = self._prior_cache.get(key)
        if probs is None:
            state = sense(self._engine, px, py, self._engine.tick + depth)
            probs = softmax(self.net.forward(state))
            self._prior_cache[key] = probs
        return float(1.0 - probs[action_idx])

    def get_move(self, game_state: Engine) -> Action:
        self.stats.clear()
        self._engine = game_state
        self._prior_cache.clear()
        move, _final_y, _completed = self._search(
            game_state, self.horizon, None, tie_fn=self._tie
        )
        self.stats["priors"] = len(self._prior_cache)
        return move
