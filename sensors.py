"""Capteurs partagés par les agents apprenants (neuroévolution, DQN, imitation).

Deux jeux de capteurs, calculables pour un état (x, y, tick) arbitraire :

- `sense_base` (32) : distances gauche/droite au premier obstacle, type de
  ligne en one-hot, vitesse signée, pour 5 lignes relatives ; + X normalisé
  et flag « sur un tronc ».
- `sense_full` (42) : base + capteurs de PHASE — pour chaque ligne, délai
  normalisé avant que la case du joueur soit occupée (tto) puis libre (ttf).
  C'est l'information temporelle que la recherche exploite via la formule
  modulo et qu'un réseau purement spatial ne peut pas deviner.
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


def _scan(x: int, occ: frozenset[int], step: int) -> float:
    """Distance normalisée au premier obstacle dans une direction (1.0 = aucun).

    Balayage circulaire (modulo), cohérent avec la topologie du monde.
    """
    for d in range(1, GRID_WIDTH):
        if (x + step * d) % GRID_WIDTH in occ:
            return d / GRID_WIDTH
    return 1.0


def _resolve(engine: Engine, x: int | None, y: int | None, tick: int | None) -> tuple[int, int, int]:
    return (
        engine.player_x if x is None else x,
        engine.player_y if y is None else y,
        engine.tick if tick is None else tick,
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
    on_log = engine.line_at(y).kind == RIVER and x in engine.occupied_columns(y, tick)
    features.append(1.0 if on_log else 0.0)
    return np.asarray(features, dtype=np.float64)


def _phase(engine: Engine, x: int, ly: int, tick: int) -> tuple[float, float]:
    """(tto, ttf) : délais normalisés avant occupation puis libération de (x, ly).

    tto = premier dt >= 0 où la case est couverte par un obstacle mobile ;
    ttf = premier dt >= 0 où elle est libre. 1.0 si jamais dans le lookahead.
    """
    line = engine.line_at(ly)
    if line.spacing == 0:  # SAFE : aucune dynamique
        return 1.0, 0.0
    tto = ttf = None
    for dt in range(PHASE_LOOKAHEAD + 1):
        occupied = x in engine.occupied_columns(ly, tick + dt)
        if tto is None and occupied:
            tto = dt / PHASE_LOOKAHEAD
        if ttf is None and not occupied:
            ttf = dt / PHASE_LOOKAHEAD
        if tto is not None and ttf is not None:
            break
    return (1.0 if tto is None else tto), (1.0 if ttf is None else ttf)


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
            phase.extend((1.0, 0.0))
            continue
        tto, ttf = _phase(engine, x, ly, tick)
        phase.extend((tto, ttf))
    return np.concatenate((base, np.asarray(phase, dtype=np.float64)))
