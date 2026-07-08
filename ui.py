"""Monitoring Pygame soigné : rendu 2D à gauche, panneau de stats à droite.

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

# --- Palette harmonisée (RGB) ---
COL_BG = (22, 24, 32)
COL_PANEL = (30, 33, 43)
COL_CARD = (40, 44, 57)
COL_TEXT = (233, 237, 246)
COL_TEXT_DIM = (144, 152, 168)
COL_SEP = (54, 59, 74)

# terrains
COL_GRASS = (88, 160, 94)
COL_GRASS_ALT = (80, 150, 86)
COL_ROAD = (58, 61, 71)
COL_ROAD_ALT = (54, 57, 66)
COL_LANE = (198, 178, 96)
COL_RIVER = (56, 120, 178)
COL_RIVER_ALT = (50, 112, 168)
COL_RIPPLE = (78, 142, 198)

# objets
COL_TREE = (44, 104, 58)
COL_TREE_DK = (32, 82, 46)
COL_TRUNK = (96, 68, 46)
COL_LOG = (152, 106, 60)
COL_LOG_DK = (120, 82, 46)
CAR_COLORS = (
    (226, 88, 74), (232, 152, 66), (98, 176, 220), (152, 202, 112), (212, 122, 192),
)
COL_SHADOW = (0, 0, 0, 70)

# joueur / états
COL_PLAYER = (250, 214, 74)
COL_PLAYER_DK = (214, 176, 44)
COL_DEAD = (232, 86, 80)
COL_WON = (96, 214, 128)
COL_EYE = (40, 40, 48)

# accents par famille
FAMILY_COLORS = {"robot": (92, 172, 240), "ia": (240, 162, 82), "hybride": (192, 132, 242)}
COL_OK = (96, 206, 132)
COL_KO = (236, 96, 72)

_LINE_BASE = {SAFE: (COL_GRASS, COL_GRASS_ALT), ROAD: (COL_ROAD, COL_ROAD_ALT),
              RIVER: (COL_RIVER, COL_RIVER_ALT)}


class Renderer:
    """Fenêtre de monitoring : grille illustrée + panneau HUD."""

    def __init__(self, title: str = "Crossy IA") -> None:
        pygame.init()
        self.game_width = GRID_WIDTH * CELL_SIZE
        self.height = VIEW_ROWS * CELL_SIZE
        self.screen = pygame.display.set_mode((self.game_width + PANEL_WIDTH, self.height))
        pygame.display.set_caption(title)
        self._init_fonts()
        self.clock = pygame.time.Clock()

    def _init_fonts(self) -> None:
        f = "consolas,dejavusansmono,menlo,monospace"
        self.font = pygame.font.SysFont(f, 15)
        self.font_sm = pygame.font.SysFont(f, 12)
        self.font_big = pygame.font.SysFont(f, 19, bold=True)
        self.font_huge = pygame.font.SysFont(f, 42, bold=True)

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
        self._draw_world(engine, self._camera(engine))
        self._draw_panel(engine, ai, last_ms)
        pygame.display.flip()
        self.clock.tick(FPS)

    def draw_end(self, engine: Engine, ai: BaseAI, last_ms: float) -> None:
        """Rendu de l'état final + bandeau de verdict superposé."""
        self.screen.fill(COL_BG)
        self._draw_world(engine, self._camera(engine))
        self._draw_panel(engine, ai, last_ms)
        if engine.won:
            titre, col = "GAGNÉ !", COL_WON
        else:
            cause = f" — {engine.death_cause}" if engine.death_cause else ""
            titre, col = f"MORT{cause}", COL_DEAD
        self._banner(self.game_width // 2, titre, col,
                     f"score {engine.score} / {TARGET_SCORE}")
        pygame.display.flip()

    def _banner(self, cx: int, titre: str, col: tuple, sub: str, width: int | None = None) -> None:
        width = width or self.game_width
        band_h = 128
        band = pygame.Surface((width, band_h), pygame.SRCALPHA)
        band.fill((0, 0, 0, 205))
        pygame.draw.rect(band, (*col, 90), band.get_rect(), width=3)
        self.screen.blit(band, (cx - width // 2, (self.height - band_h) // 2))
        self._center(titre, self.font_huge, col, cx, -20)
        self._center(sub, self.font_big, COL_TEXT, cx, 24)
        self._center("appuie sur une touche ou ferme la fenêtre", self.font_sm, COL_TEXT_DIM, cx, 50)

    def _center(self, text: str, font: pygame.font.Font, color: tuple, cx: int, dy: int = 0) -> None:
        surf = font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=(cx, self.height // 2 + dy)))

    def wait_until_dismissed(self) -> None:
        """Bloque sur l'écran de fin jusqu'à une touche ou la fermeture."""
        while True:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    return
            self.clock.tick(30)

    # --------------------------------------------------------------- caméra

    @staticmethod
    def _camera(engine: Engine) -> int:
        """Bas de la fenêtre de vision, ancré sur le meilleur Y atteint.

        engine.score ne décroît jamais : la caméra avance avec la progression
        et ne redescend jamais quand le robot recule (le jeu pousse à avancer).
        """
        return max(0, engine.score - VIEW_ROWS // 3)

    def _px(self, x: int, y: int, cam: int, x0: int) -> tuple[int, int]:
        return x0 + x * CELL_SIZE, self.height - (y - cam + 1) * CELL_SIZE

    # ---------------------------------------------------------------- monde

    def _draw_world(self, engine: Engine, cam: int, x0: int = 0) -> None:
        """Plateau dessiné à l'offset horizontal x0 (permet plusieurs plateaux)."""
        tick = engine.tick
        c = CELL_SIZE
        for y in range(cam, cam + VIEW_ROWS):
            line = engine.line_at(y)
            base, alt = _LINE_BASE[line.kind]
            top = self.height - (y - cam + 1) * c
            pygame.draw.rect(self.screen, base, (x0, top, self.game_width, c))
            if line.kind == SAFE:
                self._draw_grass(x0, top, alt)
                for gx in engine.tree_columns(y):
                    self._draw_tree(*self._px(gx, y, cam, x0))
            elif line.kind == ROAD:
                self._draw_lane_marks(x0, top)
                for gx in engine.occupied_columns(y, tick):
                    self._draw_car(*self._px(gx, y, cam, x0), line.direction, y)
            else:  # RIVER
                self._draw_ripples(x0, top)
                for gx in engine.occupied_columns(y, tick):
                    self._draw_log(*self._px(gx, y, cam, x0))
        self._draw_player(engine, cam, x0)

    def _draw_grass(self, x0: int, top: int, alt: tuple) -> None:
        c = CELL_SIZE
        for gx in range(0, GRID_WIDTH, 2):
            pygame.draw.rect(self.screen, alt, (x0 + gx * c, top, c, c))

    def _draw_lane_marks(self, x0: int, top: int) -> None:
        c = CELL_SIZE
        y = top + c // 2
        for px in range(x0 + 4, x0 + self.game_width, 18):
            pygame.draw.line(self.screen, COL_LANE, (px, y), (px + 9, y), 2)

    def _draw_ripples(self, x0: int, top: int) -> None:
        c = CELL_SIZE
        for gx in range(0, GRID_WIDTH, 2):
            bx = x0 + gx * c + 6
            pygame.draw.line(self.screen, COL_RIPPLE, (bx, top + 8), (bx + c - 10, top + 8), 1)
            pygame.draw.line(self.screen, COL_RIPPLE, (bx + 3, top + c - 8),
                             (bx + c - 6, top + c - 8), 1)

    def _shadow(self, px: int, py: int, w: int, h: int, r: int = 6) -> None:
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, COL_SHADOW, s.get_rect(), border_radius=r)
        self.screen.blit(s, (px, py))

    def _draw_tree(self, px: int, py: int) -> None:
        c = CELL_SIZE
        cx, base = px + c // 2, py + c - 6
        pygame.draw.rect(self.screen, COL_TRUNK, (cx - 2, base - 6, 4, 8))
        pygame.draw.circle(self.screen, COL_TREE_DK, (cx, py + c // 2 + 1), c // 2 - 4)
        pygame.draw.circle(self.screen, COL_TREE, (cx - 2, py + c // 2 - 2), c // 2 - 6)

    def _draw_car(self, px: int, py: int, direction: int, y: int) -> None:
        c = CELL_SIZE
        body = pygame.Rect(px + 3, py + 5, c - 6, c - 10)
        self._shadow(body.x + 2, body.y + 3, body.w, body.h, r=6)
        color = CAR_COLORS[y % len(CAR_COLORS)]
        pygame.draw.rect(self.screen, color, body, border_radius=6)
        # toit plus clair
        roof = body.inflate(-8, -10)
        pygame.draw.rect(self.screen, tuple(min(255, v + 34) for v in color), roof, border_radius=4)
        # roues
        wy = body.bottom - 2
        pygame.draw.circle(self.screen, (26, 26, 30), (body.left + 6, wy), 3)
        pygame.draw.circle(self.screen, (26, 26, 30), (body.right - 6, wy), 3)
        # phare dans le sens de marche
        hx = body.right - 3 if direction > 0 else body.left + 3
        pygame.draw.circle(self.screen, (255, 244, 200), (hx, body.centery), 2)

    def _draw_log(self, px: int, py: int) -> None:
        c = CELL_SIZE
        bar = pygame.Rect(px + 1, py + 6, c - 2, c - 12)
        self._shadow(bar.x + 1, bar.y + 3, bar.w, bar.h, r=7)
        pygame.draw.rect(self.screen, COL_LOG, bar, border_radius=7)
        for gy in (bar.centery - 4, bar.centery, bar.centery + 4):
            pygame.draw.line(self.screen, COL_LOG_DK, (bar.left + 4, gy), (bar.right - 4, gy), 1)

    def _draw_player(self, engine: Engine, cam: int, x0: int) -> None:
        c = CELL_SIZE
        px, py = self._px(engine.player_x, engine.player_y, cam, x0)
        if engine.won:
            col, dk = COL_WON, tuple(max(0, v - 40) for v in COL_WON)
        elif engine.alive:
            col, dk = COL_PLAYER, COL_PLAYER_DK
        else:
            col, dk = COL_DEAD, tuple(max(0, v - 40) for v in COL_DEAD)
        # ombre douce
        sh = pygame.Surface((c, c), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 90), (6, c - 12, c - 12, 8))
        self.screen.blit(sh, (px, py))
        body = pygame.Rect(px + 5, py + 4, c - 10, c - 10)
        pygame.draw.rect(self.screen, dk, body.move(0, 2), border_radius=7)
        pygame.draw.rect(self.screen, col, body, border_radius=7)
        # yeux (regarde vers le haut = sens du jeu)
        ey = body.top + 6
        pygame.draw.circle(self.screen, COL_EYE, (body.centerx - 4, ey), 2)
        pygame.draw.circle(self.screen, COL_EYE, (body.centerx + 4, ey), 2)

    # ---------------------------------------------------------------- panneau

    def _draw_panel(self, engine: Engine, ai: BaseAI, last_ms: float) -> None:
        x = self.game_width
        pygame.draw.rect(self.screen, COL_PANEL, (x, 0, PANEL_WIDTH, self.height))
        pygame.draw.line(self.screen, COL_SEP, (x, 0), (x, self.height), 2)
        pad = x + 18
        w = PANEL_WIDTH - 36
        y = 18

        # en-tête : nom de l'agent + pastille de famille
        self.screen.blit(self.font_big.render(ai.name, True, COL_TEXT), (pad, y))
        fam_col = FAMILY_COLORS.get(ai.family, COL_TEXT_DIM)
        self._pill(pad + self.font_big.size(ai.name)[0] + 10, y + 1, ai.family, fam_col)
        y += 34

        # statut
        if engine.won:
            st, sc = "GAGNÉ", COL_WON
        elif engine.alive:
            st, sc = "en cours", COL_TEXT
        else:
            st, sc = f"MORT · {engine.death_cause or '?'}", COL_DEAD
        self._pill(pad, y, st, sc, wide=True)
        y += 34

        # progression Y / 200
        y = self._stat_bar(pad, y, w, "Progression", f"{engine.score} / {TARGET_SCORE}",
                           engine.score / TARGET_SCORE, COL_WON)
        # temps de décision vs budget
        col_ms = COL_OK if last_ms <= AI_TIME_BUDGET_MS else COL_KO
        y = self._stat_bar(pad, y, w, "Décision", f"{last_ms:.2f} ms / {AI_TIME_BUDGET_MS:.0f}",
                           min(1.0, last_ms / AI_TIME_BUDGET_MS), col_ms)
        y += 6

        # infos brutes
        self._kv(pad, y, "tick", str(engine.tick))
        self._kv(pad + w // 2, y, "seed", str(engine.seed))
        y += 32
        if engine.noise > 0:
            self._kv(pad, y, "bruit", str(engine.noise)); y += 32
        y += 6

        # compteurs d'efficacité de l'agent
        if ai.stats:
            pygame.draw.line(self.screen, COL_SEP, (pad, y), (pad + w, y)); y += 12
            self.screen.blit(self.font_sm.render("EFFICACITÉ", True, COL_TEXT_DIM), (pad, y)); y += 22
            for k, v in ai.stats.items():
                self._kv(pad, y, k, f"{v:,.0f}"); y += 30

    def _pill(self, x: int, y: int, text: str, col: tuple, wide: bool = False) -> None:
        pad = 10 if wide else 7
        w = self.font_sm.size(text)[0] + 2 * pad
        rect = pygame.Rect(x, y, w, 20)
        bg = pygame.Surface((w, 20), pygame.SRCALPHA)
        pygame.draw.rect(bg, (*col, 46), bg.get_rect(), border_radius=10)
        self.screen.blit(bg, (x, y))
        pygame.draw.rect(self.screen, col, rect, width=1, border_radius=10)
        self.screen.blit(self.font_sm.render(text, True, col), (x + pad, y + 4))

    def _stat_bar(self, x: int, y: int, w: int, label: str, value: str,
                  frac: float, col: tuple) -> int:
        self.screen.blit(self.font_sm.render(label, True, COL_TEXT_DIM), (x, y))
        vs = self.font_sm.render(value, True, COL_TEXT)
        self.screen.blit(vs, (x + w - vs.get_width(), y))
        y += 18
        pygame.draw.rect(self.screen, COL_CARD, (x, y, w, 9), border_radius=5)
        fw = max(0, min(w, int(w * frac)))
        if fw > 0:
            pygame.draw.rect(self.screen, col, (x, y, fw, 9), border_radius=5)
        return y + 22

    def _kv(self, x: int, y: int, key: str, value: str) -> None:
        self.screen.blit(self.font_sm.render(key, True, COL_TEXT_DIM), (x, y))
        self.screen.blit(self.font.render(value, True, COL_TEXT), (x, y + 12))

    # ------------------------------------------------------------------ divers

    def close(self) -> None:
        pygame.quit()


class DuelRenderer(Renderer):
    """Deux plateaux côte à côte (même graine) + panneau comparatif."""

    _GAP = 8

    def __init__(self, title: str = "Crossy IA — duel") -> None:
        pygame.init()
        self.game_width = GRID_WIDTH * CELL_SIZE
        self.height = VIEW_ROWS * CELL_SIZE
        total = 2 * self.game_width + self._GAP + PANEL_WIDTH
        self.screen = pygame.display.set_mode((total, self.height))
        pygame.display.set_caption(title)
        self._init_fonts()
        self.clock = pygame.time.Clock()

    def _x1(self) -> int:
        return self.game_width + self._GAP

    def draw_duel(self, e1: Engine, ai1: BaseAI, ms1: float,
                  e2: Engine, ai2: BaseAI, ms2: float) -> None:
        self.screen.fill(COL_BG)
        self._draw_world(e1, self._camera(e1), 0)
        pygame.draw.rect(self.screen, COL_BG, (self.game_width, 0, self._GAP, self.height))
        self._draw_world(e2, self._camera(e2), self._x1())
        self._draw_duel_panel(e1, ai1, ms1, e2, ai2, ms2)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_duel_panel(self, e1, ai1, ms1, e2, ai2, ms2) -> None:
        x = 2 * self.game_width + self._GAP
        pygame.draw.rect(self.screen, COL_PANEL, (x, 0, PANEL_WIDTH, self.height))
        pygame.draw.line(self.screen, COL_SEP, (x, 0), (x, self.height), 2)
        pad = x + 18
        w = PANEL_WIDTH - 36
        y = 18
        self.screen.blit(self.font_big.render("DUEL", True, COL_TEXT), (pad, y)); y += 34
        for engine, ai, ms in ((e1, ai1, ms1), (e2, ai2, ms2)):
            fam = FAMILY_COLORS.get(ai.family, COL_TEXT_DIM)
            self.screen.blit(self.font_big.render(ai.name, True, COL_TEXT), (pad, y))
            self._pill(pad + self.font_big.size(ai.name)[0] + 10, y + 1, ai.family, fam)
            y += 30
            if engine.won:
                st, sc = "GAGNÉ", COL_WON
            elif engine.alive:
                st, sc = "en cours", COL_TEXT
            else:
                st, sc = f"MORT · {engine.death_cause or '?'}", COL_DEAD
            self._pill(pad, y, st, sc, wide=True); y += 30
            y = self._stat_bar(pad, y, w, "Progression",
                               f"{engine.score}/{TARGET_SCORE}",
                               engine.score / TARGET_SCORE, fam)
            self._kv(pad, y, "tick", str(engine.tick))
            self._kv(pad + w // 2, y, "décision", f"{ms:.2f} ms")
            y += 30
            pygame.draw.line(self.screen, COL_SEP, (pad, y), (pad + w, y)); y += 14
        self._kv(pad, y, "seed", str(e1.seed))
        if e1.noise > 0:
            self._kv(pad + w // 2, y, "bruit", str(e1.noise))

    def draw_duel_end(self, e1: Engine, ai1: BaseAI, e2: Engine, ai2: BaseAI) -> None:
        """Bandeau de verdict par-dessus l'état final des deux plateaux."""
        k1, k2 = (e1.score, e1.won), (e2.score, e2.won)
        if k1 > k2:
            titre, col = f"{ai1.name} L'EMPORTE", FAMILY_COLORS.get(ai1.family, COL_WON)
        elif k2 > k1:
            titre, col = f"{ai2.name} L'EMPORTE", FAMILY_COLORS.get(ai2.family, COL_WON)
        else:
            titre, col = "ÉGALITÉ", COL_TEXT
        cx = self.game_width + self._GAP // 2
        self._banner(cx, titre, col, f"{ai1.name} {e1.score} — {e2.score} {ai2.name}",
                     width=2 * self.game_width)
        pygame.display.flip()
