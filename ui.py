"""Monitoring Pygame minimaliste : rendu 2D à gauche, statistiques à droite.

Aucune logique de jeu ici : le renderer consomme l'état du moteur en lecture
seule. La simulation reste exécutable sans ce module (entraînement headless).
"""

from __future__ import annotations

import pygame

from config import (
    AI_TIME_BUDGET_MS,
    CELL_SIZE,
    FPS,
    GRID_WIDTH,
    PANEL_WIDTH,
    RIVER,
    ROAD,
    SAFE,
    TARGET_SCORE,
    VIEW_ROWS,
)
from engine import Engine
from ai_base import BaseAI

# Palette (RGB)
COL_BG = (18, 18, 24)
COL_PANEL = (28, 28, 36)
COL_TEXT = (230, 230, 230)
COL_TEXT_DIM = (150, 150, 160)
COL_SAFE = (72, 120, 60)
COL_ROAD = (70, 70, 78)
COL_RIVER = (42, 78, 140)
COL_TREE = (30, 62, 26)
COL_CAR = (210, 60, 50)
COL_LOG = (140, 96, 50)
COL_PLAYER = (250, 210, 60)
COL_DEAD = (235, 60, 60)
COL_WON = (80, 220, 120)
COL_GRID = (0, 0, 0)
COL_BUDGET_OK = (80, 200, 120)
COL_BUDGET_KO = (235, 90, 60)

_LINE_COLORS = {SAFE: COL_SAFE, ROAD: COL_ROAD, RIVER: COL_RIVER}


class Renderer:
    """Fenêtre de monitoring : grille + panneau HUD."""

    def __init__(self, title: str = "Crossy IA") -> None:
        pygame.init()
        self.game_width = GRID_WIDTH * CELL_SIZE
        self.height = VIEW_ROWS * CELL_SIZE
        self.screen = pygame.display.set_mode((self.game_width + PANEL_WIDTH, self.height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
        self.font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
        self.font_huge = pygame.font.SysFont("consolas,dejavusansmono,monospace", 40, bold=True)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------- événements

    def handle_events(self) -> bool:
        """False si l'utilisateur demande la fermeture (croix ou Échap)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    # ------------------------------------------------------------------ rendu

    def draw(self, engine: Engine, ai: BaseAI, last_ms: float) -> None:
        self.screen.fill(COL_BG)
        cam_bottom = self._camera(engine)
        self._draw_world(engine, cam_bottom)
        self._draw_panel(engine, ai, last_ms)
        pygame.display.flip()
        self.clock.tick(FPS)

    def draw_end(self, engine: Engine, ai: BaseAI, last_ms: float) -> None:
        """Rendu de l'état final + bandeau de verdict superposé (sans flip du clock)."""
        self.screen.fill(COL_BG)
        cam_bottom = self._camera(engine)
        self._draw_world(engine, cam_bottom)
        self._draw_panel(engine, ai, last_ms)

        if engine.won:
            titre, col = "GAGNÉ !", COL_WON
        else:
            cause = f" ({engine.death_cause})" if engine.death_cause else ""
            titre, col = f"MORT{cause}", COL_DEAD
        band_h = 120
        band = pygame.Surface((self.game_width, band_h), pygame.SRCALPHA)
        band.fill((0, 0, 0, 200))
        self.screen.blit(band, (0, (self.height - band_h) // 2))
        self._center(titre, self.font_huge if engine.won else self.font_big, col, dy=-18)
        self._center(f"score {engine.score} / {TARGET_SCORE}", self.font_big, COL_TEXT, dy=22)
        self._center("appuie sur une touche ou ferme la fenêtre", self.font, COL_TEXT_DIM, dy=48)
        pygame.display.flip()

    def _center(self, text: str, font: pygame.font.Font, color: tuple, dy: int = 0) -> None:
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(self.game_width // 2, self.height // 2 + dy))
        self.screen.blit(surf, rect)

    def wait_until_dismissed(self) -> None:
        """Bloque sur l'écran de fin jusqu'à une touche ou la fermeture de la fenêtre."""
        while True:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    return
            self.clock.tick(30)

    @staticmethod
    def _camera(engine: Engine) -> int:
        """Bas de la fenêtre de vision, ancré sur le meilleur Y atteint.

        engine.score ne décroît jamais : la caméra avance avec la progression
        et ne redescend jamais quand le robot recule (le jeu pousse à avancer).
        """
        return max(0, engine.score - VIEW_ROWS // 3)

    def _cell_rect(self, x: int, y: int, cam_bottom: int, x0: int = 0) -> pygame.Rect:
        row_from_bottom = y - cam_bottom
        py = self.height - (row_from_bottom + 1) * CELL_SIZE
        return pygame.Rect(x0 + x * CELL_SIZE, py, CELL_SIZE, CELL_SIZE)

    def _draw_world(self, engine: Engine, cam_bottom: int, x0: int = 0) -> None:
        """Plateau dessiné à l'offset horizontal x0 (permet plusieurs plateaux)."""
        tick = engine.tick
        for y in range(cam_bottom, cam_bottom + VIEW_ROWS):
            line = engine.line_at(y)
            band = pygame.Rect(
                x0,
                self.height - (y - cam_bottom + 1) * CELL_SIZE,
                self.game_width,
                CELL_SIZE,
            )
            pygame.draw.rect(self.screen, _LINE_COLORS[line.kind], band)
            if line.kind == SAFE:
                for x in engine.tree_columns(y):
                    pygame.draw.rect(
                        self.screen, COL_TREE,
                        self._cell_rect(x, y, cam_bottom, x0).inflate(-6, -6),
                    )
            else:
                color = COL_CAR if line.kind == ROAD else COL_LOG
                for x in engine.occupied_columns(y, tick):
                    pygame.draw.rect(
                        self.screen, color,
                        self._cell_rect(x, y, cam_bottom, x0).inflate(-4, -4),
                    )
        # quadrillage discret
        for gx in range(GRID_WIDTH + 1):
            pygame.draw.line(
                self.screen, COL_GRID,
                (x0 + gx * CELL_SIZE, 0), (x0 + gx * CELL_SIZE, self.height), 1,
            )
        # joueur
        if engine.won:
            color = COL_WON
        elif engine.alive:
            color = COL_PLAYER
        else:
            color = COL_DEAD
        rect = self._cell_rect(
            engine.player_x, engine.player_y, cam_bottom, x0
        ).inflate(-8, -8)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

    def _draw_panel(self, engine: Engine, ai: BaseAI, last_ms: float) -> None:
        panel = pygame.Rect(self.game_width, 0, PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, COL_PANEL, panel)
        x0 = self.game_width + 16
        y = 14

        y = self._text(f"IA : {ai.name}", x0, y, self.font_big, COL_TEXT) + 8
        if engine.won:
            statut, col = "GAGNE", COL_WON
        elif engine.alive:
            statut, col = "VIVANT", COL_TEXT
        else:
            statut, col = "MORT", COL_DEAD
        y = self._text(f"Statut : {statut}", x0, y, self.font, col) + 10

        y = self._text(f"Score Y : {engine.score} / {TARGET_SCORE}", x0, y, self.font, COL_TEXT)
        y = self._text(f"Tick    : {engine.tick}", x0, y, self.font, COL_TEXT)
        y = self._text(f"Seed    : {engine.seed}", x0, y, self.font, COL_TEXT_DIM) + 10

        col_ms = COL_BUDGET_OK if last_ms <= AI_TIME_BUDGET_MS else COL_BUDGET_KO
        y = self._text(
            f"Dernier coup : {last_ms:6.2f} ms (budget {AI_TIME_BUDGET_MS:.0f} ms)",
            x0, y, self.font, col_ms,
        ) + 10

        if ai.stats:
            y = self._text("Efficacite :", x0, y, self.font, COL_TEXT_DIM)
            for key, value in ai.stats.items():
                y = self._text(f"  {key:<10}: {value:,.0f}", x0, y, self.font, COL_TEXT)

    def _text(self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple) -> int:
        self.screen.blit(font.render(text, True, color), (x, y))
        return y + font.get_height() + 4

    # ------------------------------------------------------------------ divers

    def close(self) -> None:
        pygame.quit()


class DuelRenderer(Renderer):
    """Deux plateaux côte à côte (même graine) + panneau comparatif."""

    _GAP = 6  # séparation visuelle entre les deux plateaux (px)

    def __init__(self, title: str = "Crossy IA — duel") -> None:
        pygame.init()
        self.game_width = GRID_WIDTH * CELL_SIZE
        self.height = VIEW_ROWS * CELL_SIZE
        total = 2 * self.game_width + self._GAP + PANEL_WIDTH
        self.screen = pygame.display.set_mode((total, self.height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15)
        self.font_big = pygame.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
        self.font_huge = pygame.font.SysFont("consolas,dejavusansmono,monospace", 40, bold=True)
        self.clock = pygame.time.Clock()

    def _x1(self) -> int:
        return self.game_width + self._GAP

    def draw_duel(
        self,
        e1: Engine, ai1: BaseAI, ms1: float,
        e2: Engine, ai2: BaseAI, ms2: float,
    ) -> None:
        self.screen.fill(COL_BG)
        self._draw_world(e1, self._camera(e1), 0)
        self._draw_world(e2, self._camera(e2), self._x1())
        self._draw_duel_panel(e1, ai1, ms1, e2, ai2, ms2)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _statut(self, engine: Engine) -> tuple[str, tuple[int, int, int]]:
        if engine.won:
            return "GAGNÉ", COL_WON
        if engine.alive:
            return "vivant", COL_TEXT
        return f"mort ({engine.death_cause})", COL_DEAD

    def _draw_duel_panel(
        self,
        e1: Engine, ai1: BaseAI, ms1: float,
        e2: Engine, ai2: BaseAI, ms2: float,
    ) -> None:
        x0 = 2 * self.game_width + self._GAP
        pygame.draw.rect(
            self.screen, COL_PANEL, pygame.Rect(x0, 0, PANEL_WIDTH, self.height)
        )
        x0 += 16
        y = 14
        y = self._text("DUEL", x0, y, self.font_big, COL_TEXT) + 6
        for engine, ai, ms, tag in ((e1, ai1, ms1, "◄"), (e2, ai2, ms2, "►")):
            y = self._text(
                f"{tag} {ai.name} ({ai.family})", x0, y, self.font_big, COL_PLAYER
            )
            statut, col = self._statut(engine)
            y = self._text(f"  {statut}", x0, y, self.font, col)
            y = self._text(
                f"  score {engine.score}/{TARGET_SCORE} | tick {engine.tick}",
                x0, y, self.font, COL_TEXT,
            )
            y = self._text(f"  décision {ms:6.2f} ms", x0, y, self.font, COL_TEXT_DIM) + 10
        y = self._text(f"seed {e1.seed}", x0, y, self.font, COL_TEXT_DIM)
        if e1.noise > 0:
            self._text(f"bruit {e1.noise}", x0, y, self.font, COL_TEXT_DIM)

    def draw_duel_end(
        self,
        e1: Engine, ai1: BaseAI,
        e2: Engine, ai2: BaseAI,
    ) -> None:
        """Bandeau de verdict par-dessus l'état final des deux plateaux."""
        k1, k2 = (e1.score, e1.won), (e2.score, e2.won)
        if k1 > k2:
            titre, col = f"{ai1.name} L'EMPORTE", COL_WON
        elif k2 > k1:
            titre, col = f"{ai2.name} L'EMPORTE", COL_WON
        else:
            titre, col = "ÉGALITÉ", COL_TEXT
        largeur = 2 * self.game_width + self._GAP
        band = pygame.Surface((largeur, 110), pygame.SRCALPHA)
        band.fill((0, 0, 0, 200))
        self.screen.blit(band, (0, (self.height - 110) // 2))
        for text, font, c, dy in (
            (titre, self.font_huge, col, -14),
            (f"{ai1.name} {e1.score} — {e2.score} {ai2.name}", self.font_big, COL_TEXT, 26),
        ):
            surf = font.render(text, True, c)
            rect = surf.get_rect(center=(largeur // 2, self.height // 2 + dy))
            self.screen.blit(surf, rect)
        pygame.display.flip()
