"""Récompense dense partagée par les IA à renforcement (DQN, PPO).

Réponse à la cause n°2 de l'analyse (ANALYSE_IA.md) : le signal « +1 par
ligne » est trop creux pour apprendre les sous-tâches dures (entrer sur un
tronc, esquiver une voiture imminente). On ajoute un shaping POTENTIEL
(Ng et al., 1999) : F = gamma * Phi(s') - Phi(s), qui densifie le signal
sans changer la politique optimale (invariance des politiques).

Phi est volontairement BORNÉ et LOCAL (pas de terme en Y absolu : la
récompense de progression existe déjà, et un potentiel proportionnel à Y
ferait exploser la variance à la mort) :

- +PHI_PLATFORM si le joueur est vivant sur une rivière (donc sur un tronc
  ou un nénuphar) : le coup d'ENTRÉE en rivière — celui que l'exploration
  epsilon-greedy n'ose presque jamais — reçoit un bonus immédiat ;
- -PHI_DANGER * (1 - tto/horizon) sur route quand une voiture arrive dans
  moins de PHI_DANGER_HORIZON ticks : s'écarter d'une trajectoire mortelle
  rapporte tout de suite, sans attendre la mort pour l'apprendre ;
- Phi(état terminal) = 0 : la pénalité de mort reste bornée.
"""

from __future__ import annotations

from config import (
    GRID_WIDTH,
    PHI_DANGER,
    PHI_DANGER_HORIZON,
    PHI_PLATFORM,
    REWARD_DEATH,
    REWARD_PROGRESS,
    REWARD_STEP,
    REWARD_WIN,
    RIVER,
    ROAD,
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


def shaped_reward(
    engine: Engine, prev_score: int, prev_phi: float, gamma: float
) -> tuple[float, float]:
    """Récompense façonnée du pas qui vient d'être joué.

    Retourne (récompense, nouveau potentiel) — le potentiel est à re-passer
    au pas suivant pour éviter de le recalculer deux fois.
    """
    phi = potential(engine)
    return base_reward(engine, prev_score) + gamma * phi - prev_phi, phi
