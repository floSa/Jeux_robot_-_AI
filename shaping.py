"""Récompense dense partagée par les IA à renforcement (DQN, PPO).

Réponse à la cause n°2 de l'analyse (ANALYSE_IA.md) : le signal « +1 par
ligne » est trop creux pour apprendre les sous-tâches dures. On ajoute un
shaping POTENTIEL (Ng et al., 1999) : F = gamma * Phi(s') - Phi(s), qui
densifie le signal sans changer la politique optimale (invariance des
politiques). Phi(état terminal) = 0 : la pénalité de mort reste bornée.

Deux potentiels disponibles (config RL_SHAPING), tous deux BORNÉS et LOCAUX
(pas de terme en Y absolu : la récompense de progression existe déjà, et un
potentiel proportionnel à Y ferait exploser la variance à la mort) :

- « plateforme » (`potential`) : +PHI_PLATFORM vivant sur une rivière,
  -PHI_DANGER*(1 - tto/horizon) sous une voiture imminente. Mesuré NEUTRE
  avec les capteurs agrégés (Épisode 2 de l'analyse).
- « alignement » (`potential_align`) : -PHI_ALIGN par case d'écart latéral
  avec le passage praticable le plus proche de la ligne SUIVANTE (à t+1).
  Se rapprocher du trou rapporte immédiatement — c'est l'anti-stagnation.
  Mesuré GAGNANT avec les capteurs vision-grille (Épisode 3) : le réseau
  voit alors le passage vers lequel le potentiel le guide.
"""

from __future__ import annotations

import numpy as np

from config import (
    GRID_WIDTH,
    PHI_ALIGN,
    PHI_ALIGN_CAP,
    PHI_DANGER,
    PHI_DANGER_HORIZON,
    PHI_PLATFORM,
    REWARD_DEATH,
    REWARD_PROGRESS,
    REWARD_STEP,
    REWARD_WIN,
    RIVER,
    ROAD,
    SAFE,
)
from engine import Engine


def base_reward(engine: Engine, prev_score: int) -> float:
    """Récompense brute : progression + coût du temps + issues terminales."""
    r = REWARD_STEP + REWARD_PROGRESS * (engine.score - prev_score)
    if engine.won:
        r += REWARD_WIN
    elif not engine.alive:
        r += REWARD_DEATH
    return r


def potential(engine: Engine) -> float:
    """Potentiel Phi de l'état courant (0 sur état terminal)."""
    if engine.game_over:
        return 0.0
    x, y = engine.player_x, engine.player_y
    line = engine.line_at(y)
    if line.kind == RIVER:
        return PHI_PLATFORM  # vivant sur une rivière = sur une plateforme
    if line.kind == ROAD and line.blocks:
        occ, _dl, _dr, tto, _ttf = engine.line_tables(y)
        xb = (x - engine.noise_offset(y)) % GRID_WIDTH
        delay = int(tto[engine.tick % occ.shape[0], xb])
        if delay < PHI_DANGER_HORIZON:
            return -PHI_DANGER * (1.0 - delay / PHI_DANGER_HORIZON)
    return 0.0


def potential_align(engine: Engine) -> float:
    """Potentiel d'alignement : écart latéral au passage praticable le plus
    proche de la ligne SUIVANTE, évalué à t+1 (0 sur état terminal).

    C'est l'anti-stagnation : se décaler vers le trou de la ligne d'après
    rapporte immédiatement, sans attendre le +1 de la ligne franchie.
    """
    if engine.game_over:
        return 0.0
    x, y, tick = engine.player_x, engine.player_y, engine.tick
    line = engine.line_at(y + 1)
    if line.kind == SAFE:
        trees = engine.tree_columns(y + 1)
        good = np.array([c not in trees for c in range(GRID_WIDTH)])
    else:
        occ, *_ = engine.line_tables(y + 1)
        row = occ[
            (tick + 1) % occ.shape[0],
            (np.arange(GRID_WIDTH) - engine.noise_offset(y + 1)) % GRID_WIDTH,
        ]
        good = row if line.kind == RIVER else ~row
    cand = np.nonzero(good)[0]
    if len(cand) == 0:
        return 0.0
    return -PHI_ALIGN * min(int(np.min(np.abs(cand - x))), PHI_ALIGN_CAP)


# potentiels sélectionnables par la config RL_SHAPING ("off" = récompense brute)
POTENTIALS: dict[str, object] = {
    "plateforme": potential,
    "alignement": potential_align,
}


def shaped_reward(
    engine: Engine,
    prev_score: int,
    prev_phi: float,
    gamma: float,
    potential_fn=potential,
) -> tuple[float, float]:
    """Récompense façonnée du pas qui vient d'être joué.

    Retourne (récompense, nouveau potentiel) — le potentiel est à re-passer
    au pas suivant pour éviter de le recalculer deux fois.
    """
    phi = potential_fn(engine)
    return base_reward(engine, prev_score) + gamma * phi - prev_phi, phi
