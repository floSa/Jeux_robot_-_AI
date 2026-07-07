"""Moteur de simulation Crossy Road — pur, déterministe, sans dépendance graphique.

Phase 1 : trottoirs (arbres statiques) et routes (voitures mobiles).

Toute la dynamique est prédictible en O(1) à n'importe quel tick futur t :
    X_t = (X_initial + direction * (t // period)) % GRID_WIDTH
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from config import (
    Action,
    CAR_LENGTH,
    CENTER_X,
    GRID_WIDTH,
    LINE_WEIGHTS,
    MAX_TICKS,
    MAX_TREES_PER_LINE,
    PERIODS,
    ROAD,
    SAFE,
    SEARCH_HORIZON,
    SPACINGS,
    STAGNATION_LIMIT,
    START_SAFE_ROWS,
    TARGET_SCORE,
    TREE_DENSITY,
)

EMPTY: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Line:
    """Ligne immuable du monde : type + paramètres cinématiques de ses obstacles."""

    y: int
    kind: int
    direction: int = 0    # -1 vers la gauche, +1 vers la droite, 0 statique
    period: int = 1       # 1 pas de déplacement toutes les `period` ticks
    spacing: int = 0      # espacement entre débuts d'obstacles (6 ou 12)
    length: int = 1       # longueur d'un obstacle en cases
    phase: int = 0        # X initial du premier obstacle

    def shift(self, tick: int) -> int:
        """Décalage cumulé de la ligne au tick t (avant modulo)."""
        return self.direction * (tick // self.period)

    def occupied(self, tick: int) -> frozenset[int]:
        """Colonnes occupées par les obstacles mobiles au tick t (formule modulo)."""
        if self.spacing == 0:
            return EMPTY
        shift = self.shift(tick)
        cols: set[int] = set()
        for i in range(GRID_WIDTH // self.spacing):
            start = (self.phase + i * self.spacing + shift) % GRID_WIDTH
            for k in range(self.length):
                cols.add((start + k) % GRID_WIDTH)
        return frozenset(cols)


class Engine:
    """État complet d'une partie et logique de transition tick par tick.

    Le monde (lignes, obstacles) est indépendant des actions du joueur : les IA
    peuvent donc projeter l'avenir via `next_state` sans muter l'état réel.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.width = GRID_WIDTH
        self.tick = 0
        self.player_x = CENTER_X
        self.player_y = 0
        self.score = 0
        self.alive = True
        self.won = False
        self.ticks_since_progress = 0
        self.lines: list[Line] = []
        self.trees: set[tuple[int, int]] = set()
        self._trees_by_row: dict[int, frozenset[int]] = {}
        self._occ_cache: dict[tuple[int, int], frozenset[int]] = {}
        self._ensure_lines(SEARCH_HORIZON + START_SAFE_ROWS)

    # ------------------------------------------------------------------ monde

    def _ensure_lines(self, y: int) -> None:
        """Génère paresseusement les lignes jusqu'à l'index y inclus."""
        while len(self.lines) <= y:
            self.lines.append(self._generate_line(len(self.lines)))

    def _generate_line(self, y: int) -> Line:
        if y < START_SAFE_ROWS:
            return Line(y=y, kind=SAFE)
        kinds = (SAFE, ROAD)
        weights = tuple(LINE_WEIGHTS[k] for k in kinds)
        kind = self.rng.choices(kinds, weights=weights)[0]
        if kind == SAFE:
            cols = frozenset(
                x for x in range(self.width) if self.rng.random() < TREE_DENSITY
            )
            cols = frozenset(sorted(cols)[:MAX_TREES_PER_LINE])
            self._trees_by_row[y] = cols
            self.trees.update((x, y) for x in cols)
            return Line(y=y, kind=SAFE)
        return Line(
            y=y,
            kind=ROAD,
            direction=self.rng.choice((-1, 1)),
            period=self.rng.choice(PERIODS),
            spacing=self.rng.choice(SPACINGS),
            length=CAR_LENGTH,
            phase=self.rng.randrange(self.width),
        )

    def line_at(self, y: int) -> Line:
        self._ensure_lines(y)
        return self.lines[y]

    def tree_columns(self, y: int) -> frozenset[int]:
        self._ensure_lines(y)
        return self._trees_by_row.get(y, EMPTY)

    # ------------------------------------------------------------- prédiction

    def occupied_columns(self, y: int, tick: int) -> frozenset[int]:
        """Occupation mobile de la ligne y au tick t, mémoïsée en O(1).

        L'occupation est périodique de période `period * width` : la clé de
        cache (y, tick % cycle) borne la mémoire tout en restant exacte.
        """
        line = self.line_at(y)
        if line.spacing == 0:
            return EMPTY
        key = (y, tick % (line.period * self.width))
        cached = self._occ_cache.get(key)
        if cached is None:
            cached = line.occupied(tick)
            self._occ_cache[key] = cached
        return cached

    def get_obstacles_at_tick(
        self, tick: int, y_min: int | None = None, y_max: int | None = None
    ) -> set[tuple[int, int]]:
        """Coordonnées (x, y) mortelles au tick t sur la fenêtre [y_min, y_max]."""
        if y_min is None:
            y_min = self.player_y - 1
        if y_max is None:
            y_max = self.player_y + SEARCH_HORIZON
        danger: set[tuple[int, int]] = set()
        for y in range(max(0, y_min), y_max + 1):
            if self.line_at(y).kind == ROAD:
                danger.update((x, y) for x in self.occupied_columns(y, tick))
        return danger

    # ------------------------------------------------------------- transition

    def next_state(self, x: int, y: int, tick: int, action: Action) -> tuple[int, int, bool]:
        """Transition pure (x, y, tick) -> (x', y', vivant) après `action`.

        Un déplacement hors grille ou vers un arbre est bloqué (position inchangée).
        """
        dx, dy = action
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and ny >= 0:
            self._ensure_lines(ny)
            if (nx, ny) not in self.trees:
                x, y = nx, ny
        if self.line_at(y).kind == ROAD and x in self.occupied_columns(y, tick + 1):
            return x, y, False
        return x, y, True

    def step(self, action: Action) -> bool:
        """Avance le monde de t à t+1 en appliquant l'action. Retourne `alive`."""
        if not self.alive or self.won:
            return self.alive
        x, y, alive = self.next_state(self.player_x, self.player_y, self.tick, action)
        self.tick += 1
        self.player_x, self.player_y, self.alive = x, y, alive
        if y > self.score:
            self.score = y
            self.ticks_since_progress = 0
        else:
            self.ticks_since_progress += 1
        if self.score >= TARGET_SCORE:
            self.won = True
        elif self.ticks_since_progress > STAGNATION_LIMIT or self.tick >= MAX_TICKS:
            self.alive = False
        return self.alive

    # ------------------------------------------------------------------ divers

    @property
    def game_over(self) -> bool:
        return self.won or not self.alive

    def clone(self) -> "Engine":
        """Copie indépendante pour projections. Le cache d'occupation est partagé :
        ses valeurs sont déterministes et identiques pour tout clone."""
        other = object.__new__(Engine)
        other.seed = self.seed
        other.rng = random.Random()
        other.rng.setstate(self.rng.getstate())
        other.width = self.width
        other.tick = self.tick
        other.player_x = self.player_x
        other.player_y = self.player_y
        other.score = self.score
        other.alive = self.alive
        other.won = self.won
        other.ticks_since_progress = self.ticks_since_progress
        other.lines = list(self.lines)
        other.trees = set(self.trees)
        other._trees_by_row = dict(self._trees_by_row)
        other._occ_cache = self._occ_cache
        return other
