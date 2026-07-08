"""Moteur de simulation Crossy Road — pur, déterministe, sans dépendance graphique.

Phase 1 : trottoirs (arbres statiques) et routes (voitures mobiles).
Phase 2 : rivières et troncs — le joueur doit être superposé à un tronc,
          et le tronc porteur l'entraîne horizontalement au tick suivant.

Toute la dynamique est prédictible en O(1) à n'importe quel tick futur t :
    X_t = (X_initial + direction * (t // period)) % GRID_WIDTH
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from config import (
    Action,
    CENTER_X,
    GRID_WIDTH,
    LINE_WEIGHTS,
    LOG_LENGTHS,
    MAX_ADJACENT_TREES,
    MAX_CONSECUTIVE_RIVERS,
    MAX_CONSECUTIVE_SAFE,
    MAX_TICKS,
    MAX_TREES_PER_LINE,
    PERIODS,
    RIVER,
    RIVER_GAP,
    ROAD,
    ROAD_GAP,
    SAFE,
    SEARCH_HORIZON,
    STAGNATION_LIMIT,
    START_SAFE_ROWS,
    TARGET_SCORE,
    TREE_DENSITY,
    VEHICLE_LENGTHS,
)

EMPTY: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Line:
    """Ligne immuable du monde : type + motif d'obstacles qui défile en boucle.

    `blocks` est un motif fixe de (position de départ, longueur) au tick 0 :
    voitures/camions (ROAD, mortels) ou troncs (RIVER, plateformes de survie),
    de longueurs variées. Tout le motif glisse d'un bloc (`direction/period`) ;
    comme il vit sur un anneau de `GRID_WIDTH` cases, il se reproduit à
    l'identique — c'est cette périodicité qui rend le futur anticipable.
    """

    y: int
    kind: int
    direction: int = 0    # -1 vers la gauche, +1 vers la droite, 0 statique
    period: int = 1       # 1 pas de déplacement toutes les `period` ticks
    blocks: tuple[tuple[int, int], ...] = ()  # (position, longueur) au tick 0

    def shift(self, tick: int) -> int:
        """Décalage cumulé de la ligne au tick t (avant modulo)."""
        return self.direction * (tick // self.period)

    def occupied(self, tick: int, extra_shift: int = 0) -> frozenset[int]:
        """Colonnes occupées par les obstacles au tick t (motif décalé, modulo).

        extra_shift : décalage additionnel (turbulence du monde stochastique).
        """
        if not self.blocks:
            return EMPTY
        shift = self.shift(tick) + extra_shift
        cols: set[int] = set()
        for start, length in self.blocks:
            for k in range(length):
                cols.add((start + shift + k) % GRID_WIDTH)
        return frozenset(cols)


class Engine:
    """État complet d'une partie et logique de transition tick par tick.

    Le monde (lignes, obstacles) est indépendant des actions du joueur : les IA
    peuvent donc projeter l'avenir via `next_state` sans muter l'état réel.
    """

    def __init__(
        self,
        seed: int = 0,
        line_weights: dict[int, float] | None = None,
        noise: float = 0.0,
    ) -> None:
        self.seed = seed
        self.line_weights = dict(line_weights or LINE_WEIGHTS)  # profil de génération
        self.noise = noise  # proba/tick/ligne d'un décalage aléatoire persistant
        self.noise_rng = random.Random((seed + 1) * 7919)  # flux séparé de la génération
        self._noise_offsets: dict[int, int] = {}
        self.rng = random.Random(seed)
        self.width = GRID_WIDTH
        self.tick = 0
        self.player_x = CENTER_X
        self.player_y = 0
        self.score = 0
        self.alive = True
        self.won = False
        self.death_cause: str | None = None
        self.ticks_since_progress = 0
        self.lines: list[Line] = []
        self.trees: set[tuple[int, int]] = set()
        self._trees_by_row: dict[int, frozenset[int]] = {}
        self._occ_cache: dict[tuple[int, int, int], frozenset[int]] = {}
        # tables précalculées par ligne (capteurs vectorisés) : (occ, dl, dr, tto, ttf)
        self._tables: dict[int, tuple[np.ndarray, ...]] = {}
        self._tree_dists: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._consecutive_rivers = 0
        self._consecutive_safe = 0
        self._ensure_lines(SEARCH_HORIZON + START_SAFE_ROWS)

    # ------------------------------------------------------------------ monde

    def _ensure_lines(self, y: int) -> None:
        """Génère paresseusement les lignes jusqu'à l'index y inclus."""
        while len(self.lines) <= y:
            self.lines.append(self._generate_line(len(self.lines)))

    def _generate_line(self, y: int) -> Line:
        if y < START_SAFE_ROWS:
            self._consecutive_rivers = 0
            self._consecutive_safe = 0
            return Line(y=y, kind=SAFE)
        kinds = list(self.line_weights)
        weights = [self.line_weights[k] for k in kinds]
        if self._consecutive_rivers >= MAX_CONSECUTIVE_RIVERS:
            weights[kinds.index(RIVER)] = 0.0
        if self._consecutive_safe >= MAX_CONSECUTIVE_SAFE:
            weights[kinds.index(SAFE)] = 0.0  # une seule ligne d'herbe d'affilée
        kind = self.rng.choices(kinds, weights=weights)[0]
        self._consecutive_rivers = self._consecutive_rivers + 1 if kind == RIVER else 0
        self._consecutive_safe = self._consecutive_safe + 1 if kind == SAFE else 0

        if kind == SAFE:
            cols = self._grass_trees()
            self._trees_by_row[y] = cols
            self.trees.update((x, y) for x in cols)
            return Line(y=y, kind=SAFE)

        if kind == RIVER:
            lengths = LOG_LENGTHS
            gap = RIVER_GAP
            prev = self.lines[y - 1] if y > 0 else None
            # Rivières empilées : directions opposées (les troncs finissent par
            # s'aligner, le saut d'une rivière à l'autre existe toujours).
            direction = -prev.direction if (prev and prev.kind == RIVER) else self.rng.choice((-1, 1))
        else:
            lengths = VEHICLE_LENGTHS
            gap = ROAD_GAP
            direction = self.rng.choice((-1, 1))
        return Line(
            y=y,
            kind=kind,
            direction=direction,
            period=self.rng.choice(PERIODS),
            blocks=self._pattern(lengths, gap),
        )

    def _pattern(self, lengths: tuple[int, ...], gap: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        """Motif d'obstacles (position, longueur) sur l'anneau, longueurs mélangées.

        Laisse une marge aux deux bords pour garantir un trou à la couture ;
        les trous (`gap`) garantissent qu'une ligne n'est jamais pleine.
        """
        blocks: list[tuple[int, int]] = []
        pos = self.rng.randint(1, 3)
        while pos < self.width - 1:
            length = self.rng.choice(lengths)
            if pos + length > self.width - 1:
                break
            blocks.append((pos, length))
            pos += length + self.rng.randint(gap[0], gap[1])
        return tuple(blocks)

    def _grass_trees(self) -> frozenset[int]:
        """Colonnes d'arbres : petits bosquets de 1 ou 2 arbres, jamais 3 collés."""
        cols: set[int] = set()
        x = 0
        while x < self.width and len(cols) < MAX_TREES_PER_LINE:
            if self.rng.random() < TREE_DENSITY:
                run = self.rng.randint(1, MAX_ADJACENT_TREES)
                for k in range(run):
                    if x + k < self.width and len(cols) < MAX_TREES_PER_LINE:
                        cols.add(x + k)
                x += run + 1  # au moins une case libre après un bosquet
            else:
                x += 1
        return frozenset(cols)

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
        En monde stochastique, le décalage de turbulence courant s'ajoute :
        les prédictions futures utilisent le décalage CONNU au moment du
        calcul — la réalité peut diverger, c'est le but.
        """
        line = self.line_at(y)
        if not line.blocks:
            return EMPTY
        offset = self._noise_offsets.get(y, 0)
        key = (y, tick % (line.period * self.width), offset)
        cached = self._occ_cache.get(key)
        if cached is None:
            cached = line.occupied(tick, extra_shift=offset)
            self._occ_cache[key] = cached
        return cached

    def _apply_noise(self) -> None:
        """Turbulence : chaque ligne mobile active peut glisser de ±1 case."""
        for y in range(max(0, self.player_y - 5), self.player_y + 26):
            line = self.line_at(y)
            if line.blocks and self.noise_rng.random() < self.noise:
                self._noise_offsets[y] = (
                    self._noise_offsets.get(y, 0) + self.noise_rng.choice((-1, 1))
                ) % self.width

    def noise_offset(self, y: int) -> int:
        """Décalage de turbulence courant de la ligne y (0 en monde déterministe)."""
        return self._noise_offsets.get(y, 0)

    def obstacle_blocks(self, y: int, tick: int) -> list[tuple[int, int]]:
        """Blocs (colonne de départ, longueur) de la ligne y au tick t, pour le rendu.

        Un bloc qui franchit le bord droit est scindé en deux morceaux [0, w).
        """
        line = self.line_at(y)
        if not line.blocks:
            return []
        shift = line.shift(tick) + self._noise_offsets.get(y, 0)
        pieces: list[tuple[int, int]] = []
        for start, length in line.blocks:
            s = (start + shift) % self.width
            if s + length <= self.width:
                pieces.append((s, length))
            else:  # scindé à la couture
                pieces.append((s, self.width - s))
                pieces.append((0, length - (self.width - s)))
        return pieces

    def line_tables(self, y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tables précalculées de la ligne mobile y, sur son cycle exact.

        (occ, dist_gauche, dist_droite, tto, ttf) — formes (cycle, largeur).
        occ : bool ; distances en cases (255 = aucune) ; tto/ttf : délai en
        ticks avant occupation / libération de la colonne (255 = jamais).
        Invariantes par décalage : lire la colonne (x - offset) % largeur.
        """
        tables = self._tables.get(y)
        if tables is None:
            tables = self._build_tables(self.line_at(y))
            self._tables[y] = tables
        return tables

    def _build_tables(self, line: Line) -> tuple[np.ndarray, ...]:
        w = self.width
        cycle = line.period * w
        occ = np.zeros((cycle, w), dtype=bool)
        for t in range(cycle):
            for c in line.occupied(t):
                occ[t, c] = True
        xs = np.arange(w)
        dl = np.full((cycle, w), 255, dtype=np.uint8)
        dr = np.full((cycle, w), 255, dtype=np.uint8)
        for t in range(cycle):
            idx = np.flatnonzero(occ[t])
            if idx.size:
                d_right = (idx[None, :] - xs[:, None]) % w
                d_right = np.where(d_right == 0, w, d_right).min(axis=1)
                dr[t] = np.where(d_right < w, d_right, 255).astype(np.uint8)
                d_left = (xs[:, None] - idx[None, :]) % w
                d_left = np.where(d_left == 0, w, d_left).min(axis=1)
                dl[t] = np.where(d_left < w, d_left, 255).astype(np.uint8)
        tto = np.full((cycle, w), 255, dtype=np.uint8)
        ttf = np.full((cycle, w), 255, dtype=np.uint8)
        ts = np.arange(cycle)
        for x in range(w):
            col = occ[:, x]
            for target, table in ((True, tto), (False, ttf)):
                times = np.flatnonzero(col == target)
                if times.size:
                    pos = np.searchsorted(times, ts) % times.size
                    table[:, x] = ((times[pos] - ts) % cycle).astype(np.uint8)
        return occ, dl, dr, tto, ttf

    def tree_dists(self, y: int) -> tuple[np.ndarray, np.ndarray]:
        """Distances gauche/droite aux arbres de la ligne SAFE y (255 = aucun)."""
        dists = self._tree_dists.get(y)
        if dists is None:
            w = self.width
            xs = np.arange(w)
            idx = np.fromiter(self.tree_columns(y), dtype=np.int64)
            if idx.size:
                d_right = (idx[None, :] - xs[:, None]) % w
                d_right = np.where(d_right == 0, w, d_right).min(axis=1)
                dr = np.where(d_right < w, d_right, 255).astype(np.uint8)
                d_left = (xs[:, None] - idx[None, :]) % w
                d_left = np.where(d_left == 0, w, d_left).min(axis=1)
                dl = np.where(d_left < w, d_left, 255).astype(np.uint8)
            else:
                dl = np.full(w, 255, dtype=np.uint8)
                dr = np.full(w, 255, dtype=np.uint8)
            dists = (dl, dr)
            self._tree_dists[y] = dists
        return dists

    def get_obstacles_at_tick(
        self, tick: int, y_min: int | None = None, y_max: int | None = None
    ) -> set[tuple[int, int]]:
        """Coordonnées (x, y) mortelles au tick t sur la fenêtre [y_min, y_max].

        ROAD : cases des voitures. RIVER : eau libre (complément des troncs).
        """
        if y_min is None:
            y_min = self.player_y - 1
        if y_max is None:
            y_max = self.player_y + SEARCH_HORIZON
        danger: set[tuple[int, int]] = set()
        for y in range(max(0, y_min), y_max + 1):
            kind = self.line_at(y).kind
            if kind == ROAD:
                danger.update((x, y) for x in self.occupied_columns(y, tick))
            elif kind == RIVER:
                occ = self.occupied_columns(y, tick)
                danger.update((x, y) for x in range(self.width) if x not in occ)
        return danger

    # ------------------------------------------------------------- transition

    def apply_drift(self, x: int, y: int, tick: int) -> int:
        """X après portage par le tronc entre t et t+1, SANS wrap.

        Le tronc glisse le joueur d'une case ; s'il le pousse au-delà du bord,
        la valeur retournée sort de [0, largeur) — le joueur ne réapparaît
        jamais de l'autre côté (pas de téléportation façon Snake).
        """
        line = self.line_at(y)
        if line.kind == RIVER and x in self.occupied_columns(y, tick):
            return x + line.shift(tick + 1) - line.shift(tick)
        return x

    def next_state(self, x: int, y: int, tick: int, action: Action) -> tuple[int, int, bool]:
        """Transition pure (x, y, tick) -> (x', y', vivant) après `action`.

        Ordre : 1) portage par le tronc (sans wrap) — un joueur poussé au-delà
        du bord chute AUSSITÔT, aucun rattrapage possible ; 2) action volontaire
        (hors grille ou arbre = bloquée) ; 3) verdict de survie à t+1.
        """
        x = self.apply_drift(x, y, tick)
        if not 0 <= x < self.width:
            return max(0, min(self.width - 1, x)), y, False  # sorti de l'écran = mort
        dx, dy = action
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and ny >= 0:
            self._ensure_lines(ny)
            if (nx, ny) not in self.trees:
                x, y = nx, ny
        kind = self.line_at(y).kind
        if kind == ROAD and x in self.occupied_columns(y, tick + 1):
            return x, y, False
        if kind == RIVER and x not in self.occupied_columns(y, tick + 1):
            return x, y, False
        return x, y, True

    def step(self, action: Action) -> bool:
        """Avance le monde de t à t+1 en appliquant l'action. Retourne `alive`."""
        if not self.alive or self.won:
            return self.alive
        if self.noise > 0.0:
            self._apply_noise()  # la turbulence précède la transition t -> t+1
        x, y, alive = self.next_state(self.player_x, self.player_y, self.tick, action)
        self.tick += 1
        self.player_x, self.player_y, self.alive = x, y, alive
        if not alive:
            self.death_cause = "noyade" if self.line_at(y).kind == RIVER else "collision"
        if y > self.score:
            self.score = y
            self.ticks_since_progress = 0
        else:
            self.ticks_since_progress += 1
        if self.score >= TARGET_SCORE:
            self.won = True
        elif self.alive and (
            self.ticks_since_progress > STAGNATION_LIMIT or self.tick >= MAX_TICKS
        ):
            self.alive = False
            self.death_cause = (
                "stagnation" if self.tick < MAX_TICKS else "limite de ticks"
            )
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
        other.line_weights = dict(self.line_weights)
        other.noise = self.noise
        other.noise_rng = random.Random()
        other.noise_rng.setstate(self.noise_rng.getstate())
        other._noise_offsets = dict(self._noise_offsets)
        other.rng = random.Random()
        other.rng.setstate(self.rng.getstate())
        other.width = self.width
        other.tick = self.tick
        other.player_x = self.player_x
        other.player_y = self.player_y
        other.score = self.score
        other.alive = self.alive
        other.won = self.won
        other.death_cause = self.death_cause
        other.ticks_since_progress = self.ticks_since_progress
        other.lines = list(self.lines)
        other.trees = set(self.trees)
        other._trees_by_row = dict(self._trees_by_row)
        other._occ_cache = self._occ_cache
        other._tables = self._tables  # tables immuables : partage sûr
        other._tree_dists = self._tree_dists
        other._consecutive_rivers = self._consecutive_rivers
        return other
