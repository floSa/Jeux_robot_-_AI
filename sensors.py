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

import numpy as np

from config import (
    GRID_WIDTH,
    NN_SENSOR_ROWS,
    PHASE_LOOKAHEAD,
    RIVER,
    ROAD,
    SAFE,
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
    if line.spacing == 0:  # SAFE : aucune dynamique
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
