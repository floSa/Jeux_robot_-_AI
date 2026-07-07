"""Constantes globales de la simulation Crossy Road et des IA."""

from __future__ import annotations

from typing import Final

Action = tuple[int, int]

# --- Grille ---
GRID_WIDTH: Final[int] = 19            # colonnes 0..18, hauteur infinie (génération paresseuse)
CENTER_X: Final[int] = GRID_WIDTH // 2
TARGET_SCORE: Final[int] = 200         # nombre d'avancées Y pour gagner

# --- Types de lignes ---
SAFE: Final[int] = 0                   # trottoir/herbe : arbres statiques bloquants
ROAD: Final[int] = 1                   # route : voitures mortelles
RIVER: Final[int] = 2                  # rivière : mortelle sauf superposition à un tronc
LINE_KINDS: Final[tuple[int, ...]] = (SAFE, ROAD, RIVER)

# --- Actions (dx, dy) ---
AVANCER: Final[Action] = (0, 1)
RECULER: Final[Action] = (0, -1)
GAUCHE: Final[Action] = (-1, 0)
DROITE: Final[Action] = (1, 0)
ATTENDRE: Final[Action] = (0, 0)
ACTIONS: Final[tuple[Action, ...]] = (AVANCER, ATTENDRE, GAUCHE, DROITE, RECULER)
ACTION_NAMES: Final[dict[Action, str]] = {
    AVANCER: "AVANCER",
    RECULER: "RECULER",
    GAUCHE: "GAUCHE",
    DROITE: "DROITE",
    ATTENDRE: "ATTENDRE",
}

# --- Obstacles dynamiques ---
SPACINGS: Final[tuple[int, ...]] = (6, 12)   # espacement strict entre débuts d'obstacles d'une ligne
RIVER_SPACING_WEIGHTS: Final[tuple[int, ...]] = (3, 1)  # les rivières favorisent l'espacement 6
CAR_LENGTH: Final[int] = 1                   # longueur d'une voiture (cases)
LOG_LENGTH: Final[int] = 3                   # longueur d'un tronc (cases)
PERIODS: Final[tuple[int, ...]] = (1, 2, 3)  # 1 pas toutes les `period` ticks ; vitesse = dir/period

# --- Génération du monde ---
START_SAFE_ROWS: Final[int] = 3              # bande de départ sans danger
LINE_WEIGHTS: Final[dict[int, float]] = {SAFE: 0.30, ROAD: 0.45, RIVER: 0.25}
MAX_CONSECUTIVE_RIVERS: Final[int] = 3
TREE_DENSITY: Final[float] = 0.18
MAX_TREES_PER_LINE: Final[int] = 6           # < GRID_WIDTH : jamais de mur d'arbres infranchissable

# --- Garde-fous de partie ---
MAX_TICKS: Final[int] = 20_000               # borne dure anti-partie infinie
STAGNATION_LIMIT: Final[int] = 120           # ticks sans nouveau Y max avant mort

# --- IA ---
SEARCH_HORIZON: Final[int] = 15              # profondeur de recherche T+15
AI_TIME_BUDGET_MS: Final[float] = 20.0       # budget de décision par tick

# --- Neuroévolution ---
NN_SENSOR_ROWS: Final[tuple[int, ...]] = (-1, 0, 1, 2, 3)      # lignes perçues (relatives à Y)
NN_FEATURES_PER_ROW: Final[int] = 6          # dist. gauche/droite + one-hot type (3) + vitesse signée
NN_INPUT_SIZE: Final[int] = len(NN_SENSOR_ROWS) * NN_FEATURES_PER_ROW + 2  # + X normalisé + flag tronc
NN_HIDDEN_SIZE: Final[int] = 16
NN_OUTPUT_SIZE: Final[int] = len(ACTIONS)
POPULATION_SIZE: Final[int] = 100
ELITE_COUNT: Final[int] = 5
TOURNAMENT_SIZE: Final[int] = 5
CROSSOVER_RATE: Final[float] = 0.9
MUTATION_RATE: Final[float] = 0.10
MUTATION_SIGMA: Final[float] = 0.30
EPISODES_PER_EVAL: Final[int] = 3
VALIDATION_SEEDS: Final[tuple[int, ...]] = (9001, 9002, 9003, 9004, 9005)
DEFAULT_GENERATIONS: Final[int] = 60
MODEL_PATH: Final[str] = "models/best.npz"

# --- UI ---
CELL_SIZE: Final[int] = 28
VIEW_ROWS: Final[int] = 21                   # lignes visibles à l'écran
PANEL_WIDTH: Final[int] = 320                # panneau de statistiques (px)
FPS: Final[int] = 15
