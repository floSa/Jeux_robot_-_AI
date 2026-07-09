"""Capteurs partagés par les agents apprenants (neuroévolution, DQN, PPO, imitation).

Deux jeux de capteurs, calculables pour un état (x, y, tick) arbitraire :

- `sense_base` (32) : distances gauche/droite au premier obstacle, type de
  ligne en one-hot, vitesse signée, pour 5 lignes relatives ; + X normalisé
  et flag « sur un tronc ».
- `sense_full` (42) : base + capteurs de PHASE — pour chaque ligne, délai
  normalisé avant que la case du joueur soit occupée (tto) puis libre (ttf).
  C'est l'information temporelle que la recherche exploite via la formule
  modulo et qu'un réseau purement spatial ne peut pas deviner.

Implémentation vectorisée : le moteur précalcule par ligne, sur son cycle
exact, les tables occupation / distances / phases (`Engine.line_tables`) ;
chaque capteur devient un accès tabulaire O(1). En monde stochastique, le
décalage de turbulence courant s'applique par translation de colonne.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from config import (
    GRID_SENSOR_ROWS,
    GRID_SENSOR_SIZE,
    GRID_TIME_PLANES,
    GRID_WIDTH,
    NN_SENSOR_ROWS,
    PHASE_LOOKAHEAD,
    RIVER,
    ROAD,
    SAFE,
    STAGNATION_LIMIT,
)
from engine import Engine

SENSOR_BASE_SIZE: int = len(NN_SENSOR_ROWS) * 6 + 2
SENSOR_PHASE_SIZE: int = len(NN_SENSOR_ROWS) * 2
SENSOR_FULL_SIZE: int = SENSOR_BASE_SIZE + SENSOR_PHASE_SIZE

_PAD_BASE = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # sous la grille = mur sûr
_PAD_PHASE = (1.0, 0.0)


def _resolve(engine: Engine, x: int | None, y: int | None, tick: int | None) -> tuple[int, int, int]:
    return (
        engine.player_x if x is None else x,
        engine.player_y if y is None else y,
        engine.tick if tick is None else tick,
    )


def _row_features(engine: Engine, ly: int, x: int, tick: int) -> tuple[float, ...]:
    """(dist gauche, dist droite, one-hot type x3, vitesse signée) de la ligne ly."""
    line = engine.line_at(ly)
    if line.kind == SAFE:
        dl, dr = engine.tree_dists(ly)
        left, right = dl[x], dr[x]
    else:
        occ, dl, dr, _tto, _ttf = engine.line_tables(ly)
        cycle = occ.shape[0]
        xb = (x - engine.noise_offset(ly)) % GRID_WIDTH
        tb = tick % cycle
        left, right = dl[tb, xb], dr[tb, xb]
    return (
        left / GRID_WIDTH if left < 255 else 1.0,
        right / GRID_WIDTH if right < 255 else 1.0,
        1.0 if line.kind == SAFE else 0.0,
        1.0 if line.kind == ROAD else 0.0,
        1.0 if line.kind == RIVER else 0.0,
        line.direction / line.period,
    )


def sense_base(
    engine: Engine, x: int | None = None, y: int | None = None, tick: int | None = None
) -> np.ndarray:
    """Capteurs spatiaux (SENSOR_BASE_SIZE) pour l'état donné (défaut : joueur)."""
    x, y, tick = _resolve(engine, x, y, tick)
    features: list[float] = []
    for dy in NN_SENSOR_ROWS:
        ly = y + dy
        if ly < 0:
            features.extend(_PAD_BASE)
            continue
        features.extend(_row_features(engine, ly, x, tick))
    features.append(x / (GRID_WIDTH - 1) * 2.0 - 1.0)
    line = engine.line_at(y)
    on_log = False
    if line.kind == RIVER:
        occ, *_ = engine.line_tables(y)
        on_log = bool(
            occ[tick % occ.shape[0], (x - engine.noise_offset(y)) % GRID_WIDTH]
        )
    features.append(1.0 if on_log else 0.0)
    return np.asarray(features, dtype=np.float64)


def _phase(engine: Engine, x: int, ly: int, tick: int) -> tuple[float, float]:
    """(tto, ttf) : délais normalisés avant occupation puis libération de (x, ly)."""
    line = engine.line_at(ly)
    if not line.blocks:  # SAFE : aucune dynamique
        return 1.0, 0.0
    occ, _dl, _dr, tto, ttf = engine.line_tables(ly)
    cycle = occ.shape[0]
    xb = (x - engine.noise_offset(ly)) % GRID_WIDTH
    tb = tick % cycle
    v_tto, v_ttf = tto[tb, xb], ttf[tb, xb]
    return (
        v_tto / PHASE_LOOKAHEAD if v_tto < 255 else 1.0,
        v_ttf / PHASE_LOOKAHEAD if v_ttf < 255 else 1.0,
    )


def sense_full(
    engine: Engine, x: int | None = None, y: int | None = None, tick: int | None = None
) -> np.ndarray:
    """Capteurs complets (SENSOR_FULL_SIZE) : base + phase des 5 lignes."""
    x, y, tick = _resolve(engine, x, y, tick)
    base = sense_base(engine, x, y, tick)
    phase: list[float] = []
    for dy in NN_SENSOR_ROWS:
        ly = y + dy
        if ly < 0:
            phase.extend(_PAD_PHASE)
            continue
        phase.extend(_phase(engine, x, ly, tick))
    return np.concatenate((base, np.asarray(phase, dtype=np.float64)))


def _row_standable(engine: Engine, ly: int, x: int, tick: int) -> np.ndarray:
    """Praticabilité de la ligne ly, vue égocentrique, aux instants t+k.

    Retourne un tableau (len(GRID_TIME_PLANES), GRID_WIDTH) : la colonne j
    correspond à la case x + j - GRID_WIDTH//2 (joueur au centre). Une case
    vaut 1.0 si le joueur pourrait s'y tenir vivant à t+k : herbe sans arbre,
    route sans véhicule, rivière AVEC tronc ou nénuphar (logique inversée).
    Hors grille = 0.0 (le bord est un mur mortel).
    """
    half = GRID_WIDTH // 2
    cols = np.arange(GRID_WIDTH) + x - half          # colonnes absolues visées
    valid = (0 <= cols) & (cols < GRID_WIDTH)
    out = np.zeros((len(GRID_TIME_PLANES), GRID_WIDTH))
    line = engine.line_at(ly)
    if line.kind == SAFE:
        trees = engine.tree_columns(ly)
        free = np.array([bool(v) and (c not in trees) for c, v in zip(cols, valid)])
        out[:] = free.astype(np.float64)
        return out
    occ, *_ = engine.line_tables(ly)
    cycle = occ.shape[0]
    xb = (cols - engine.noise_offset(ly)) % GRID_WIDTH
    for i, k in enumerate(GRID_TIME_PLANES):
        occupied = occ[(tick + k) % cycle, xb]
        standable = occupied if line.kind == RIVER else ~occupied
        out[i] = np.where(valid, standable.astype(np.float64), 0.0)
    return out


def sense_grid(
    engine: Engine, x: int | None = None, y: int | None = None, tick: int | None = None
) -> np.ndarray:
    """Capteurs « vision grille » (GRID_SENSOR_SIZE) : la projection spatiale
    et temporelle complète que les capteurs agrégés (sense_full) résument.

    Par ligne perçue : la carte de praticabilité 19 colonnes x len(GRID_TIME_PLANES)
    instants (monde périodique : la projection est exacte, même classe
    d'information que les capteurs de phase tto/ttf), + type one-hot + vitesse
    signée. Global : X normalisé + compteur de stagnation normalisé (sans lui,
    la mort par stagnation est invisible : l'état serait non markovien, et une
    politique argmax bloquée resterait bloquée pour l'éternité).
    """
    x, y, tick = _resolve(engine, x, y, tick)
    parts: list[np.ndarray] = []
    for dy in GRID_SENSOR_ROWS:
        ly = y + dy
        if ly < 0:
            block = np.zeros(GRID_WIDTH * len(GRID_TIME_PLANES) + 4)
            block[-4] = 1.0  # sous la grille : « herbe » infranchissable
            parts.append(block)
            continue
        line = engine.line_at(ly)
        parts.append(
            np.concatenate((
                _row_standable(engine, ly, x, tick).ravel(),
                (
                    1.0 if line.kind == SAFE else 0.0,
                    1.0 if line.kind == ROAD else 0.0,
                    1.0 if line.kind == RIVER else 0.0,
                    line.direction / line.period,
                ),
            ))
        )
    parts.append(np.asarray((
        x / (GRID_WIDTH - 1) * 2.0 - 1.0,
        min(engine.ticks_since_progress / STAGNATION_LIMIT, 1.0),
    )))
    return np.concatenate(parts)


# ---------------------------------------------------------------- dispatch

# capteurs disponibles : nom -> (fonction, taille d'un état)
SENSOR_SETS: dict[str, tuple] = {
    "grid": (sense_grid, GRID_SENSOR_SIZE),
    "full": (sense_full, SENSOR_FULL_SIZE),
}


def sensor_for_input(n_inputs: int):
    """Retrouve (fonction capteur, profondeur de pile K) depuis la taille
    d'entrée d'un réseau chargé — les tailles des jeux de capteurs ne sont
    pas multiples l'une de l'autre, le dispatch est sans ambiguïté."""
    for fn, size in SENSOR_SETS.values():
        if n_inputs % size == 0:
            return fn, n_inputs // size
    raise ValueError(f"aucun jeu de capteurs ne correspond à une entrée de {n_inputs}")


class FrameStack:
    """Mémoire courte : concatène les K dernières observations (ancien -> récent).

    Réponse à la cause n°3 de l'analyse (politique réactive sans mémoire) :
    empiler les K derniers vecteurs de capteurs donne au réseau les dérivées
    temporelles (une plateforme qui approche, une voiture qui accélère…) sans
    passer à une architecture récurrente. Au premier pas d'un épisode, la
    première observation est répétée K fois (démarrage neutre). K = 1 est
    l'identité : les modèles sans mémoire restent chargeables tels quels.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"pile de {k} observations (minimum 1)")
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Ajoute l'observation du tick et retourne l'état empilé."""
        if self.k == 1:
            return obs
        if not self._frames:
            self._frames.extend([obs] * self.k)
        else:
            self._frames.append(obs)
        return np.concatenate(self._frames)
