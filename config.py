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

# --- Obstacles dynamiques (motifs de blocs qui défilent en boucle) ---
# Chaque ligne mobile porte un motif de blocs (position, longueur) qui glisse d'un
# bloc à l'autre ; le motif se reproduit à l'identique (périodicité = anticipation).
VEHICLE_LENGTHS: Final[tuple[int, ...]] = (2, 3)   # voiture = 2 cases, camion = 3
LOG_LENGTHS: Final[tuple[int, ...]] = (2, 3, 4)    # troncs de 2 à 4 cases, mélangés
ROAD_GAP: Final[tuple[int, int]] = (2, 5)          # trou entre véhicules (>= 2 => toujours franchissable)
RIVER_GAP: Final[tuple[int, int]] = (1, 2)         # eau entre troncs (dense => troncs atteignables)
LILY_PROB: Final[float] = 0.25                     # part des rivières éligibles en nénuphars
LILY_PADS: Final[tuple[int, int]] = (5, 7)         # nombre de nénuphars (fixes, isolés) par ligne
PERIODS: Final[tuple[int, ...]] = (1, 2, 3)  # 1 pas toutes les `period` ticks ; vitesse = dir/period

# --- Génération du monde ---
START_SAFE_ROWS: Final[int] = 3              # bande de départ sans danger
LINE_WEIGHTS: Final[dict[int, float]] = {SAFE: 0.30, ROAD: 0.45, RIVER: 0.25}
MAX_CONSECUTIVE_RIVERS: Final[int] = 3
MAX_CONSECUTIVE_SAFE: Final[int] = 1         # une seule ligne d'herbe d'affilée
MAX_ADJACENT_TREES: Final[int] = 2           # jamais plus de 2 arbres collés
TREE_DENSITY: Final[float] = 0.20
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

# --- Capteurs « vision grille » (spatio-temporels, égocentriques) ---
# Carte de praticabilité par cellule (19 colonnes centrées sur le joueur) pour
# chaque ligne perçue, PROJETÉE aux instants t+k (monde périodique : exact).
GRID_SENSOR_ROWS: Final[tuple[int, ...]] = (-1, 0, 1, 2, 3, 4)
GRID_TIME_PLANES: Final[tuple[int, ...]] = (0, 1, 2, 3)   # horizons de projection
# par ligne : 19 cases x plans temporels + one-hot type (3) + vitesse signée ;
# global : X normalisé + compteur de stagnation (complétude Markov)
GRID_SENSOR_SIZE: Final[int] = len(GRID_SENSOR_ROWS) * (
    GRID_WIDTH * len(GRID_TIME_PLANES) + 4
) + 2

# --- Neuroévolution ---
NN_HIDDEN_SIZE: Final[int] = 32
NN_OUTPUT_SIZE: Final[int] = len(ACTIONS)
POPULATION_SIZE: Final[int] = 100
ELITE_COUNT: Final[int] = 5
TOURNAMENT_SIZE: Final[int] = 5
CROSSOVER_RATE: Final[float] = 0.9
MUTATION_RATE: Final[float] = 0.10
MUTATION_SIGMA: Final[float] = 0.30
EPISODES_PER_EVAL: Final[int] = 3
# 10 graines : la sélection du « meilleur modèle » est moins sensible à la chance
VALIDATION_SEEDS: Final[tuple[int, ...]] = tuple(range(9001, 9011))
DEFAULT_GENERATIONS: Final[int] = 60
MODEL_PATH: Final[str] = "models/best.npz"
FITNESS_SURVIVAL_BONUS: Final[float] = 0.005  # shaping : bonus/tick survécu (<< 1 ligne)
TRAIN_TEMPERATURE: Final[float] = 0.5         # softmax d'exploration à l'entraînement
CURRICULUM_FRACTION: Final[float] = 0.4       # part des générations sur mondes sans rivière
CURRICULUM_WEIGHTS: Final[dict[int, float]] = {SAFE: 0.35, ROAD: 0.65, RIVER: 0.0}

# --- Options RL avancées, TOUTES MESURÉES par ablation (voir ANALYSE_IA.md) ---
# Défauts = la recette la plus performante à notre budget (~10^5 pas). Les briques
# « littérature » (mémoire, n-pas, curriculum, shaping, Double DQN) restent
# disponibles : à ce budget elles sont neutres à nuisibles, chiffres à l'appui.
FRAME_STACK: Final[int] = 1                   # pile des K dernières observations (1 = sans mémoire)
# shaping potentiel : "off", "plateforme" (mesuré neutre, Épisode 2) ou
# "alignement" (mesuré gagnant AVEC la vision grille, Épisode 3)
RL_SHAPING: Final[str] = "alignement"
PHI_PLATFORM: Final[float] = 0.3              # potentiel : vivant sur tronc/nénuphar
PHI_DANGER: Final[float] = 0.2                # potentiel : voiture imminente sur ma case
PHI_DANGER_HORIZON: Final[int] = 8            # ticks sous lesquels le danger « se sent »
PHI_ALIGN: Final[float] = 0.03                # potentiel : par case d'écart au passage suivant
PHI_ALIGN_CAP: Final[int] = 8                 # écart au-delà duquel le potentiel sature

# --- DQN ---
DQN_SENSOR: Final[str] = "grid"               # "grid" (vision grille) ou "full" (42 agrégés)
# Architecture (piste B.2, STRATEGIE.md Épisode 7) : un MLP dense doit
# réapprendre chaque motif de praticabilité à chaque colonne indépendamment ;
# une convolution 1D partage ses poids entre colonnes (équivariance par
# translation). Mesuré : conv 6 canaux/noyau 5 bat le MLP dense de +45 %
# (83,4 -> 120,6, 4000 ép./2 graines) ; plus de capacité (8 canaux, noyau 7)
# nuit, comme partout ailleurs à ce budget d'entraînement.
DQN_ARCHITECTURE: Final[str] = "conv"         # "conv" (mesuré gagnant) ou "mlp"
DQN_CONV_CHANNELS: Final[int] = 6
DQN_CONV_KERNEL: Final[int] = 5
DQN_HIDDEN_SIZE: Final[int] = 64
DQN_LR: Final[float] = 1e-3
DQN_GAMMA: Final[float] = 0.97
DQN_EPS_START: Final[float] = 1.0
DQN_EPS_END: Final[float] = 0.05
DQN_EPS_DECAY: Final[float] = 0.999           # décroissance par épisode
DQN_BUFFER_SIZE: Final[int] = 100_000
DQN_BATCH_SIZE: Final[int] = 64
DQN_TRAIN_EVERY: Final[int] = 2               # 1 mise à jour tous les 2 pas
DQN_TARGET_SYNC: Final[int] = 1000            # synchronisation du réseau cible (pas)
DQN_EPISODES: Final[int] = 12_000
DQN_DOUBLE: Final[bool] = False               # Double DQN (choix en ligne, évaluation cible)
DQN_NSTEP: Final[int] = 1                     # retours multi-pas (1 = TD classique)
DQN_CURRICULUM_PROB: Final[float] = 0.0       # part d'épisodes « rivières denses »
DQN_CURRICULUM_WEIGHTS: Final[dict[int, float]] = {SAFE: 0.35, ROAD: 0.10, RIVER: 0.55}
DQN_MODEL_PATH: Final[str] = "models/dqn.npz"
REWARD_PROGRESS: Final[float] = 1.0           # nouvelle ligne max franchie
REWARD_STEP: Final[float] = -0.01             # coût du temps
REWARD_DEATH: Final[float] = -1.0
REWARD_WIN: Final[float] = 5.0

# --- PPO ---
PPO_HIDDEN_SIZE: Final[int] = 64
PPO_LR: Final[float] = 3e-4
PPO_GAMMA: Final[float] = 0.97
PPO_LAMBDA: Final[float] = 0.95            # GAE
PPO_CLIP: Final[float] = 0.2               # epsilon du surrogate clippé
PPO_ROLLOUT: Final[int] = 4096             # pas collectés par itération
PPO_EPOCHS: Final[int] = 4                 # passes d'optimisation par rollout
PPO_BATCH: Final[int] = 256
PPO_ENTROPY: Final[float] = 0.01           # bonus d'exploration
PPO_ITERATIONS: Final[int] = 1500
PPO_MODEL_PATH: Final[str] = "models/ppo.npz"

# --- MCTS (robot) ---
MCTS_BUDGET_MS: Final[float] = 15.0
MCTS_UCT_C: Final[float] = 1.2
MCTS_ROLLOUT_DEPTH: Final[int] = 25
MCTS_FORWARD_BIAS: Final[float] = 0.5         # proba d'essayer AVANCER d'abord en rollout

# --- Imitation (clone du planificateur) + recherche guidée ---
POLICY_MODEL_PATH: Final[str] = "models/policy.npz"
IMITATION_SENSOR: Final[str] = "grid"         # vision grille : précision 73 -> 87 %
POLICY_HIDDEN_SIZE: Final[int] = 64
IMITATION_SAMPLES: Final[int] = 60_000
IMITATION_EPOCHS: Final[int] = 12
IMITATION_LR: Final[float] = 1e-3
IMITATION_BATCH: Final[int] = 128
DAGGER_ROUNDS: Final[int] = 2                 # tours DAgger (l'élève conduit, l'expert étiquette)
DAGGER_SAMPLES: Final[int] = 30_000           # exemples collectés par tour
GUIDED_PRIOR_DEPTH: Final[int] = 2            # profondeur max où le prior guide le tri

# --- Filet de sécurité (ShieldedAI, hybride) ---
# Vérifie avec le simulateur EXACT (pas d'apprentissage, pas d'approximation)
# qu'une suite d'actions survivante existe sur cet horizon avant de jouer le
# coup préféré de l'agent enveloppé. Voir STRATEGIE.md, piste A.
# Profondeur 7 mesurée optimale : dqn-shield 165 -> 178,5 (plateau dès d=5),
# clone-shield 153 -> 180,4 (gagne encore à d=7 — le réseau le plus faible
# profite le plus de la vérification profonde). Coût max mesuré : 4,7 ms / 20.
SHIELD_DEPTH: Final[int] = 7
# Sauvetage anti-stagnation (piste A'.3) : quand l'agent n'a plus progressé
# depuis SHIELD_RESCUE_AFTER ticks, le filet cherche (BFS exact borné) le
# premier coup d'un chemin qui atteint une NOUVELLE ligne max en au plus
# SHIELD_RESCUE_DEPTH ticks, et l'impose s'il existe.
SHIELD_RESCUE_AFTER: Final[int] = 60
SHIELD_RESCUE_DEPTH: Final[int] = 8
# Filet robuste au bruit (piste A'', monde --noise) : le filet déterministe
# suppose la turbulence actuelle figée, alors qu'Engine._apply_noise() en
# retire une nouvelle à CHAQUE tick réel (avant même la transition en cours).
# Sous --noise, la survie est donc vérifiée sur SHIELD_NOISE_SAMPLES tirages
# indépendants de turbulence future (expectimax local échantillonné),
# jusqu'à SHIELD_NOISE_DEPTH ticks (plus court que SHIELD_DEPTH : chaque
# tirage répète toute l'exploration, le coût multiplie par SHIELD_NOISE_SAMPLES).
SHIELD_NOISE_SAMPLES: Final[int] = 6
SHIELD_NOISE_DEPTH: Final[int] = 4
SHIELD_NOISE_THRESHOLD: Final[float] = 0.5  # part des tirages qui doit survivre

# --- UI ---
CELL_SIZE: Final[int] = 32
VIEW_ROWS: Final[int] = 21                   # lignes visibles à l'écran
PANEL_WIDTH: Final[int] = 340                # panneau de statistiques (px)
FPS: Final[int] = 15
