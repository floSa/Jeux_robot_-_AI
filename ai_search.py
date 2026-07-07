"""IA de recherche prédictive best-first à horizon T+15, avec mémoization.

Le monde étant déterministe et indépendant du joueur, un état futur est
entièrement décrit par (x, y, profondeur) : la mémoization coupe toute
branche retombant sur un état déjà visité, sans clonage du moteur.

File de priorité (heapq) ordonnée par f = y + (horizon - profondeur), borne
supérieure admissible du Y atteignable : le premier nœud extrait à l'horizon
maximise Y (schéma A*). Départage par distance au centre en X.
"""

from __future__ import annotations

import heapq
from itertools import count

from config import (
    ACTIONS,
    Action,
    ATTENDRE,
    CENTER_X,
    SEARCH_HORIZON,
)
from engine import Engine
from ai_base import BaseAI

# Nœud dans le tas : (-f, dist_centre, n_ordre, x, y, profondeur, 1er coup)
_Node = tuple[int, int, int, int, int, int, Action]


class SearchAI(BaseAI):
    """Recherche exhaustive élaguée sur 15 ticks d'anticipation."""

    name = "search"

    def __init__(self, horizon: int = SEARCH_HORIZON) -> None:
        super().__init__()
        self.horizon = horizon

    def get_move(self, game_state: Engine) -> Action:
        x0, y0 = game_state.player_x, game_state.player_y
        t0 = game_state.tick
        horizon = self.horizon
        next_state = game_state.next_state  # liaison locale (hot path)

        visited: set[tuple[int, int, int]] = {(x0, y0, 0)}
        order = count()  # départage FIFO, évite la comparaison des actions
        frontier: list[_Node] = []
        pruned = deaths = expanded = 0

        # Meilleur nœud de repli si aucun chemin n'atteint l'horizon :
        # maximise (profondeur, y, -dist) => survivre le plus longtemps possible.
        best_fallback: tuple[int, int, int] | None = None
        best_fallback_move: Action = ATTENDRE

        # Expansion de la racine : chaque branche mémorise son premier coup.
        for action in ACTIONS:
            nx, ny, alive = next_state(x0, y0, t0, action)
            if not alive:
                deaths += 1
                continue
            state = (nx, ny, 1)
            if state in visited:
                pruned += 1
                continue
            visited.add(state)
            f = ny + horizon - 1
            heapq.heappush(
                frontier, (-f, abs(nx - CENTER_X), next(order), nx, ny, 1, action)
            )

        while frontier:
            _, dist, _, x, y, depth, first = heapq.heappop(frontier)
            expanded += 1
            key = (depth, y, -dist)
            if best_fallback is None or key > best_fallback:
                best_fallback, best_fallback_move = key, first
            if depth >= horizon:
                # f admissible : premier nœud extrait à l'horizon = Y optimal.
                self._flush_stats(expanded, pruned, deaths, len(visited))
                return first
            tick = t0 + depth
            for action in ACTIONS:
                nx, ny, alive = next_state(x, y, tick, action)
                if not alive:
                    deaths += 1
                    continue
                state = (nx, ny, depth + 1)
                if state in visited:
                    pruned += 1
                    continue
                visited.add(state)
                f = ny + horizon - depth - 1
                heapq.heappush(
                    frontier,
                    (-f, abs(nx - CENTER_X), next(order), nx, ny, depth + 1, first),
                )

        # Aucun chemin ne survit jusqu'à T+15 : jouer la branche la plus durable.
        self._flush_stats(expanded, pruned, deaths, len(visited))
        return best_fallback_move

    def _flush_stats(self, expanded: int, pruned: int, deaths: int, visited: int) -> None:
        self.stats["explores"] = expanded
        self.stats["elagues"] = pruned
        self.stats["morts"] = deaths
        self.stats["visites"] = visited
