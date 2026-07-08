"""IA de recherche prédictive best-first avec mémoization, horizon fixe ou adaptatif.

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
from time import perf_counter

from config import (
    ACTIONS,
    Action,
    AI_TIME_BUDGET_MS,
    ATTENDRE,
    CENTER_X,
    SEARCH_HORIZON,
    SEARCH_HORIZON_STEP,
    SEARCH_MAX_HORIZON,
)
from engine import Engine
from ai_base import BaseAI

# Nœud dans le tas : (-f, dist_centre, n_ordre, x, y, profondeur, 1er coup)
_Node = tuple[int, int, int, int, int, int, Action]


class SearchAI(BaseAI):
    """Recherche exhaustive élaguée à horizon fixe (T+15 par défaut)."""

    name = "search"

    def __init__(self, horizon: int = SEARCH_HORIZON) -> None:
        super().__init__()
        self.horizon = horizon

    def get_move(self, game_state: Engine) -> Action:
        self.stats.clear()
        move, _final_y, _completed = self._search(game_state, self.horizon, None)
        return move

    # ------------------------------------------------------------------ cœur

    def _search(
        self, game_state: Engine, horizon: int, deadline: float | None
    ) -> tuple[Action, int, bool]:
        """Une passe de recherche.

        Retourne (premier coup du meilleur chemin, Y final de ce chemin,
        recherche terminée sans interruption par la deadline).
        Les compteurs de self.stats sont incrémentés (cumulables entre passes).
        """
        x0, y0 = game_state.player_x, game_state.player_y
        t0 = game_state.tick
        next_state = game_state.next_state  # liaison locale (hot path)

        visited: set[tuple[int, int, int]] = {(x0, y0, 0)}
        order = count()  # départage FIFO, évite la comparaison des actions
        frontier: list[_Node] = []
        pruned = deaths = expanded = 0

        # Meilleur nœud de repli si aucun chemin n'atteint l'horizon :
        # maximise (profondeur, y, -dist) => survivre le plus longtemps possible.
        best_fallback: tuple[int, int, int] | None = None
        best_fallback_move: Action = ATTENDRE
        completed = True

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

        result: tuple[Action, int] | None = None
        while frontier:
            if deadline is not None and perf_counter() >= deadline:
                completed = False
                break
            _, dist, _, x, y, depth, first = heapq.heappop(frontier)
            expanded += 1
            key = (depth, y, -dist)
            if best_fallback is None or key > best_fallback:
                best_fallback, best_fallback_move = key, first
            if depth >= horizon:
                # f admissible : premier nœud extrait à l'horizon = Y optimal.
                result = (first, y)
                break
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

        if result is None:
            # Aucun chemin ne survit jusqu'à l'horizon (ou deadline) :
            # jouer la branche la plus durable.
            final_y = best_fallback[1] if best_fallback is not None else y0
            result = (best_fallback_move, final_y)

        self.stats["explores"] = self.stats.get("explores", 0) + expanded
        self.stats["elagues"] = self.stats.get("elagues", 0) + pruned
        self.stats["morts"] = self.stats.get("morts", 0) + deaths
        self.stats["visites"] = self.stats.get("visites", 0) + len(visited)
        return result[0], result[1], completed


class AdaptiveSearchAI(SearchAI):
    """Recherche à horizon adaptatif.

    Si le meilleur chemin à T+horizon ne progresse pas (Y final <= Y actuel,
    cas des rivières dont l'alignement des troncs dépasse l'horizon), relance
    la recherche avec un horizon élargi, tant que le budget temps le permet.
    Évite la mort par stagnation sans jamais dépasser 20 ms par coup.
    """

    name = "search+"

    def __init__(
        self,
        horizon: int = SEARCH_HORIZON,
        max_horizon: int = SEARCH_MAX_HORIZON,
        step: int = SEARCH_HORIZON_STEP,
    ) -> None:
        super().__init__(horizon)
        self.max_horizon = max_horizon
        self.step = step

    def get_move(self, game_state: Engine) -> Action:
        self.stats.clear()
        y0 = game_state.player_y
        deadline = perf_counter() + AI_TIME_BUDGET_MS * 0.7 / 1000.0

        horizon = self.horizon
        move, final_y, completed = self._search(game_state, horizon, deadline)
        while (
            completed
            and final_y <= y0
            and horizon < self.max_horizon
            and perf_counter() < deadline
        ):
            horizon += self.step
            deeper = self._search(game_state, horizon, deadline)
            # une passe interrompue n'est fiable que si elle a trouvé mieux
            if deeper[2] or deeper[1] > final_y:
                move, final_y, completed = deeper
        self.stats["horizon"] = horizon
        return move
