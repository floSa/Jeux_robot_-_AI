"""MCTS : Monte-Carlo Tree Search mono-joueur (ROBOT, aucun apprentissage).

Classement assumé : un MCTS seul n'est pas une IA — c'est un algorithme de
planification par échantillonnage, ses paramètres ne sont pas appris.

Pertinence ici : le monde étant déterministe et parfaitement prévisible, la
recherche exhaustive élaguée (search) est l'outil optimal ; le MCTS sert de
point de comparaison scientifique — que vaut un planificateur STOCHASTIQUE à
budget de calcul égal (15 ms/tick) ? Il deviendrait pertinent en propre si le
jeu devenait aléatoire (transitions bruitées), où search ne s'applique plus.

Schéma UCT classique : sélection (UCB1) -> expansion -> rollout (politique
aléatoire biaisée survie) -> rétropropagation de la récompense (progression
en Y). Les états étant (x, y, tick) déterministes, l'arbre est exact.
"""

from __future__ import annotations

import math
import random
from time import perf_counter

from config import (
    ACTIONS,
    Action,
    ATTENDRE,
    AVANCER,
    MCTS_BUDGET_MS,
    MCTS_FORWARD_BIAS,
    MCTS_ROLLOUT_DEPTH,
    MCTS_UCT_C,
)
from engine import Engine
from ai_base import BaseAI


class _Node:
    """Nœud d'arbre : état (x, y, profondeur) + statistiques UCT."""

    __slots__ = ("x", "y", "depth", "visits", "value", "children", "untried")

    def __init__(self, x: int, y: int, depth: int) -> None:
        self.x = x
        self.y = y
        self.depth = depth
        self.visits = 0
        self.value = 0.0  # somme des récompenses rétropropagées
        self.children: dict[Action, "_Node"] = {}
        self.untried: list[Action] = list(ACTIONS)

    def uct_select(self, c: float) -> tuple[Action, "_Node"]:
        log_n = math.log(self.visits)
        best_score = -math.inf
        best: tuple[Action, "_Node"] | None = None
        for action, child in self.children.items():
            score = child.value / child.visits + c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score, best = score, (action, child)
        assert best is not None
        return best


class MCTSAI(BaseAI):
    """UCT sous budget temps strict ; décision = enfant le plus visité."""

    name = "mcts"
    family = "robot"

    def __init__(
        self,
        budget_ms: float = MCTS_BUDGET_MS,
        uct_c: float = MCTS_UCT_C,
        rollout_depth: int = MCTS_ROLLOUT_DEPTH,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.budget_ms = budget_ms
        self.uct_c = uct_c
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    # -------------------------------------------------------------- rollout

    def _rollout(self, game_state: Engine, x: int, y: int, depth: int) -> tuple[int, bool]:
        """Partie aléatoire biaisée survie depuis (x, y) à la profondeur donnée.

        Retourne (meilleur Y atteint, mort rencontrée). À chaque pas : AVANCER
        est tenté d'abord avec proba MCTS_FORWARD_BIAS, sinon ordre aléatoire ;
        la première action qui survit est jouée.
        """
        next_state = game_state.next_state
        t0 = game_state.tick
        rng = self.rng
        best_y = y
        for step in range(self.rollout_depth):
            tick = t0 + depth + step
            if rng.random() < MCTS_FORWARD_BIAS:
                order = [AVANCER] + rng.sample(ACTIONS, len(ACTIONS))
            else:
                order = rng.sample(ACTIONS, len(ACTIONS))
            moved = False
            for action in order:
                nx, ny, alive = next_state(x, y, tick, action)
                if alive:
                    x, y = nx, ny
                    best_y = max(best_y, ny)
                    moved = True
                    break
            if not moved:  # aucune action ne survit : cul-de-sac mortel
                return best_y, True
        return best_y, False

    # ------------------------------------------------------------- décision

    def get_move(self, game_state: Engine) -> Action:
        x0, y0, t0 = game_state.player_x, game_state.player_y, game_state.tick
        next_state = game_state.next_state
        budget_end = perf_counter() + self.budget_ms / 1000.0
        root = _Node(x0, y0, 0)
        root.visits = 1
        iterations = 0
        last_iter = 0.0  # garde adaptative : 2x la dernière itération + 1 ms

        while True:
            now = perf_counter()
            if now + 2.0 * last_iter + 0.001 >= budget_end:
                break
            iterations += 1
            node = root
            path = [root]
            # 1. sélection
            while not node.untried and node.children:
                _, node = node.uct_select(self.uct_c)
                path.append(node)
            # 2. expansion
            if node.untried:
                action = node.untried.pop(self.rng.randrange(len(node.untried)))
                nx, ny, alive = next_state(node.x, node.y, t0 + node.depth, action)
                if not alive:
                    reward = 0.1 * (node.y - y0) - 1.0  # branche mortelle
                    for n in path:
                        n.visits += 1
                        n.value += reward
                    continue
                child = _Node(nx, ny, node.depth + 1)
                node.children[action] = child
                path.append(child)
                node = child
            # 3. rollout
            best_y, died = self._rollout(game_state, node.x, node.y, node.depth)
            # récompense = progression en Y (0,1/ligne), pénalité si mort en route
            reward = 0.1 * (max(best_y, node.y) - y0) - (0.5 if died else 0.0)
            # 4. rétropropagation
            for n in path:
                n.visits += 1
                n.value += reward
            last_iter = perf_counter() - now

        if not root.children:
            return ATTENDRE
        best_action = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        self.stats["iterations"] = iterations
        self.stats["noeuds"] = sum(1 for _ in self._walk(root))
        return best_action

    @staticmethod
    def _walk(node: _Node):
        yield node
        for child in node.children.values():
            yield from MCTSAI._walk(child)
