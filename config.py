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

# --- Robots ---
SEARCH_HORIZON: Final[int] = 15              # profondeur de recherche par défaut (réglable --horizon)
SEARCH_HORIZON_MIN: Final[int] = 1
SEARCH_HORIZON_MAX: Final[int] = 30          # garde-fou du budget 20 ms
AI_TIME_BUDGET_MS: Final[float] = 20.0       # budget de décision par tick

# --- Capteurs des agents apprenants ---
NN_SENSOR_ROWS: Final[tuple[int, ...]] = (-1, 0, 1, 2, 3)      # lignes perçues (relatives à Y)
NN_FEATURES_PER_ROW: Final[int] = 6          # dist. gauche/droite + one-hot type (3) + vitesse signée
PHASE_LOOKAHEAD: Final[int] = 57             # horizon des capteurs de phase (cycle max = 3 x 19)
# base (6/ligne + X + flag tronc) + phase (tto/ttf par ligne)
NN_INPUT_SIZE: Final[int] = len(NN_SENSOR_ROWS) * (NN_FEATURES_PER_ROW + 2) + 2

# --- Neuroévolution ---
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
FITNESS_SURVIVAL_BONUS: Final[float] = 0.005  # shaping : bonus/tick survécu (<< 1 ligne)
TRAIN_TEMPERATURE: Final[float] = 0.5         # softmax d'exploration à l'entraînement
CURRICULUM_FRACTION: Final[float] = 0.4       # part des générations sur mondes sans rivière
CURRICULUM_WEIGHTS: Final[dict[int, float]] = {SAFE: 0.35, ROAD: 0.65, RIVER: 0.0}

# --- DQN ---
DQN_HIDDEN_SIZE: Final[int] = 32
DQN_LR: Final[float] = 1e-3
DQN_GAMMA: Final[float] = 0.97
DQN_EPS_START: Final[float] = 1.0
DQN_EPS_END: Final[float] = 0.05
DQN_EPS_DECAY: Final[float] = 0.995           # décroissance par épisode
DQN_BUFFER_SIZE: Final[int] = 50_000
DQN_BATCH_SIZE: Final[int] = 64
DQN_TRAIN_EVERY: Final[int] = 2               # 1 mise à jour tous les 2 pas
DQN_TARGET_SYNC: Final[int] = 500             # synchronisation du réseau cible (pas)
DQN_EPISODES: Final[int] = 800
DQN_MODEL_PATH: Final[str] = "models/dqn.npz"
REWARD_PROGRESS: Final[float] = 1.0           # nouvelle ligne max franchie
REWARD_STEP: Final[float] = -0.01             # coût du temps
REWARD_DEATH: Final[float] = -1.0
REWARD_WIN: Final[float] = 5.0

# --- PPO ---
PPO_HIDDEN_SIZE: Final[int] = 32
PPO_LR: Final[float] = 3e-4
PPO_GAMMA: Final[float] = 0.97
PPO_LAMBDA: Final[float] = 0.95            # GAE
PPO_CLIP: Final[float] = 0.2               # epsilon du surrogate clippé
PPO_ROLLOUT: Final[int] = 2048             # pas collectés par itération
PPO_EPOCHS: Final[int] = 4                 # passes d'optimisation par rollout
PPO_BATCH: Final[int] = 256
PPO_ENTROPY: Final[float] = 0.01           # bonus d'exploration
PPO_ITERATIONS: Final[int] = 200
PPO_MODEL_PATH: Final[str] = "models/ppo.npz"

# --- MCTS (robot) ---
MCTS_BUDGET_MS: Final[float] = 15.0
MCTS_UCT_C: Final[float] = 1.2
MCTS_ROLLOUT_DEPTH: Final[int] = 25
MCTS_FORWARD_BIAS: Final[float] = 0.5         # proba d'essayer AVANCER d'abord en rollout

# --- Imitation (clone du planificateur) + recherche guidée ---
POLICY_MODEL_PATH: Final[str] = "models/policy.npz"
POLICY_HIDDEN_SIZE: Final[int] = 32
IMITATION_SAMPLES: Final[int] = 60_000
IMITATION_EPOCHS: Final[int] = 12
IMITATION_LR: Final[float] = 1e-3
IMITATION_BATCH: Final[int] = 128
GUIDED_PRIOR_DEPTH: Final[int] = 2            # profondeur max où le prior guide le tri

# --- UI ---
CELL_SIZE: Final[int] = 28
VIEW_ROWS: Final[int] = 21                   # lignes visibles à l'écran
PANEL_WIDTH: Final[int] = 320                # panneau de statistiques (px)
FPS: Final[int] = 15
