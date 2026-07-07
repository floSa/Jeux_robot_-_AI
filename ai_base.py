"""Interface commune des IA et IA heuristique à horizon T+1."""

from __future__ import annotations

from abc import ABC, abstractmethod

from config import (
    Action,
    ATTENDRE,
    AVANCER,
    CENTER_X,
    DROITE,
    GAUCHE,
    RECULER,
)
from engine import Engine


class BaseAI(ABC):
    """Contrat minimal d'un robot : un état de jeu -> une action par tick.

    `stats` expose des compteurs internes (nœuds explorés, branches élaguées…)
    consommés par le HUD ; aucun impact sur la logique de décision.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.stats: dict[str, float] = {}

    @abstractmethod
    def get_move(self, game_state: Engine) -> Action:
        """Choisit une action pour le tick courant. Ne doit pas muter game_state."""

    def reset(self) -> None:
        """Réinitialise l'état interne entre deux épisodes."""
        self.stats.clear()


class HeuristicAI(BaseAI):
    """IA gloutonne : évalue les 5 actions à T+1 seulement.

    Priorité absolue : AVANCER > ATTENDRE > GAUCHE/DROITE > RECULER.
    Égalité (latéraux) : action minimisant la distance au centre en X.
    Une action bloquée (arbre, bord) est ignorée : elle gaspillerait le tick.
    """

    name = "heuristic"

    _PRIORITY: tuple[tuple[Action, ...], ...] = (
        (AVANCER,),
        (ATTENDRE,),
        (GAUCHE, DROITE),
        (RECULER,),
    )
    _INF = float("inf")

    def get_move(self, game_state: Engine) -> Action:
        x, y, tick = game_state.player_x, game_state.player_y, game_state.tick
        base_x = game_state.apply_drift(x, y, tick)  # position réelle avant action
        evaluated = 0
        for group in self._PRIORITY:
            best: Action | None = None
            best_dist = self._INF
            for action in group:
                dx, dy = action
                tx, ty = base_x + dx, y + dy
                blocked = not (0 <= tx < game_state.width and ty >= 0) or (
                    tx in game_state.tree_columns(ty)  # force la génération de la ligne
                )
                if blocked and action != ATTENDRE:
                    continue
                nx, ny, alive = game_state.next_state(x, y, tick, action)
                evaluated += 1
                if not alive:
                    continue
                dist = abs(nx - CENTER_X)
                if dist < best_dist:
                    best, best_dist = action, dist
            if best is not None:
                self.stats["evaluees"] = evaluated
                return best
        self.stats["evaluees"] = evaluated
        return ATTENDRE  # aucune action ne survit : mort inévitable
