# Graph Report - .  (2026-07-10)

## Corpus Check
- Corpus is ~31,653 words - fits in a single context window. You may not need a graph.

## Summary
- 471 nodes · 900 edges · 50 communities (18 shown, 32 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 124 edges (avg confidence: 0.71)
- Token cost: 0 input · 217,963 output

## Community Hubs (Navigation)
- BaseAI/HeuristicAI interface
- Neuroevolution (GA) trainer
- DQN trainer and agent
- Search planner and world generation
- Pygame rendering and duel UI
- DQN/clone breakthrough narrative
- ShieldedAI safety net
- PPO trainer and agent
- Sensors and PPO docs
- Module architecture overview
- Reward-shaping literature and audit plan
- MCTS agent
- Neuroevolution agent (cross-doc)
- Sparse-reward exploration research
- Heuristic agent (cross-doc)
- MCTS agent (cross-doc)
- POMDP memory research
- PPO agent (cross-doc)
- Irreversibility cause and citation
- Elo tournament idea
- Train-line hazard idea
- Double DQN option
- N-step returns option
- Frame-stack memory option
- Reactive-learning limitation
- Shield depth parameter
- Heuristic tie-break bias
- CLI flags idea
- Obsolete spacing fix
- Missing test suite
- Replanning inefficiency
- Dead code finding
- Package metadata
- MCTS module doc
- PPO module doc
- DQN module doc
- Line dataclass doc
- Neural network module doc
- River-carrying mechanic doc
- Sensors module doc
- Reward-shaping module doc
- Aggregated sensors doc
- Bench protocol doc
- Directed exploration idea
- Full AlphaZero hybrid idea
- Game rules doc
- Stochastic noise mode doc
- Noise-robust planning idea
- Objective and metric doc
- PPO grid-vision TODO

## God Nodes (most connected - your core abstractions)
1. `Engine` - 95 edges
2. `BaseAI` - 40 edges
3. `MLP` - 40 edges
4. `Renderer` - 32 edges
5. `SearchAI` - 18 edges
6. `ShieldedAI` - 17 edges
7. `FrameStack` - 17 edges
8. `GeneticTrainer` - 16 edges
9. `DQNTrainer` - 16 edges
10. `PPOTrainer` - 15 edges

## Surprising Connections (you probably didn't know these)
- `M5: typing gaps, no mypy/ruff` --references--> `BaseAI`  [INFERRED]
  AUDIT.md → ai_base.py
- `M2: sequential GA evaluation, propose multiprocessing` --references--> `GeneticTrainer`  [INFERRED]
  AUDIT.md → ai_genetic.py
- `nn agent (neuroevolution)` --references--> `GeneticTrainer`  [INFERRED]
  ROBOTS.md → ai_genetic.py
- `H2: fixed locally-unsolvable stacked-river worlds` --references--> `SearchAI`  [INFERRED]
  AUDIT.md → ai_search.py
- `ai_search.py module` --references--> `SearchAI`  [INFERRED]
  README.md → ai_search.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Episode 3 grid-vision representation breakthrough (dqn + clone unlocked by sense_grid + alignment shaping)** — analyse_ia_dqn, analyse_ia_clone, analyse_ia_sense_grid, analyse_ia_alignment_shaping [INFERRED 0.90]
- **ShieldedAI safety-net pattern wrapping dqn and clone** — analyse_ia_shieldedai, analyse_ia_dqn_shield, analyse_ia_clone_shield [INFERRED 0.90]
- **Seed-variance-dominates methodology shared across research log and journal** — strategie_methodology, analyse_ia_multi_seed_training, analyse_ia_ref_henderson_rl_matters [INFERRED 0.85]

## Communities (50 total, 32 thin omitted)

### Community 0 - "BaseAI/HeuristicAI interface"
Cohesion: 0.06
Nodes (52): ABC, BaseAI, HeuristicAI, Action, Interface commune des IA et IA heuristique à horizon T+1., Contrat minimal d'un agent : un état de jeu -> une action par tick.      `family, Choisit une action pour le tick courant. Ne doit pas muter game_state., Réinitialise l'état interne entre deux épisodes. (+44 more)

### Community 1 - "Neuroevolution (GA) trainer"
Cohesion: 0.08
Nodes (29): _eval_genome(), GeneticTrainer, load_genome(), NeuralAI, Action, ndarray, Neuroévolution : MLP numpy piloté par capteurs + algorithme génétique.  Génotype, Algorithme génétique sur les génomes du réseau.      Chaque génération est évalu (+21 more)

### Community 2 - "DQN trainer and agent"
Cohesion: 0.07
Nodes (28): DQNAI, DQNTrainer, Action, ndarray, DQN : Q-learning profond en numpy pur (IA, apprentissage par renforcement).  Dif, Tampon circulaire numpy : insertion et échantillonnage sans copie de liste., Boucle d'apprentissage : collecte epsilon-greedy + descente TD., Monde standard, ou « rivières denses » pour une part des épisodes.          Curr (+20 more)

### Community 3 - "Search planner and world generation"
Cohesion: 0.07
Nodes (22): Action, Une passe de recherche.          Retourne (premier coup du meilleur chemin, Y fi, Line, Action, ndarray, Génère paresseusement les lignes jusqu'à l'index y inclus., Nénuphars : cases sûres FIXES, isolées (>= 2 d'écart), réparties., Motif d'obstacles (position, longueur) sur l'anneau, longueurs mélangées. (+14 more)

### Community 4 - "Pygame rendering and duel UI"
Cohesion: 0.08
Nodes (14): Font, DuelRenderer, Rendu de l'état final + bandeau de verdict superposé., Bloque sur l'écran de fin jusqu'à une touche ou la fermeture., Bas de la fenêtre de vision, ancré sur le meilleur Y atteint.          engine.sc, Plateau dessiné à l'offset horizontal x0 (permet plusieurs plateaux)., Véhicule vu du dessus, d'un seul tenant sur `length` cases.          length 2 =, Nénuphar fixe : disque vert avec encoche et petite fleur. (+6 more)

### Community 5 - "DQN/clone breakthrough narrative"
Cohesion: 0.07
Nodes (34): Alignment potential shaping (RL_SHAPING="alignement"), AlphaZero pattern: combine search and learning, clone agent (imitation), DAgger imitation correction, dqn agent (grid-vision Q-learning), guided hybrid agent, Multi-seed parallel training + validation selection, CrossyRoadPlayer: DDQN and A3C on Crossy Road (GitHub) (+26 more)

### Community 6 - "ShieldedAI safety net"
Cohesion: 0.08
Nodes (26): Enveloppe n'importe quel agent d'un filet de sécurité EXACT.      Réponse à l'ob, ShieldedAI, clone-shield hybrid agent, dqn-shield hybrid agent, search agent (planner), ShieldedAI safety net (Episode 4), ai_hybrid.py module (imitation/DAgger/guided/shield), Bench protocol (same seeds, mean/median/max/win-rate) (+18 more)

### Community 7 - "PPO trainer and agent"
Cohesion: 0.15
Nodes (12): PPOAI, PPOTrainer, Action, ndarray, Avantages GAE(lambda) et retours cibles du critique., Retourne (meilleure politique validée, score validé)., Un entraînement complet — fonction de module, compatible multiprocessing., `workers` entraînements indépendants en parallèle ; garde le mieux validé. (+4 more)

### Community 8 - "Sensors and PPO docs"
Cohesion: 0.16
Nodes (17): PPO : policy gradient actor-critic en numpy pur (IA, apprentissage par renforcem, _phase(), ndarray, Capteurs partagés par les agents apprenants (neuroévolution, DQN, PPO, imitation, (tto, ttf) : délais normalisés avant occupation puis libération de (x, ly)., Capteurs complets (SENSOR_FULL_SIZE) : base + phase des 5 lignes., Praticabilité de la ligne ly, vue égocentrique, aux instants t+k.      Retourne, Capteurs « vision grille » (GRID_SENSOR_SIZE) : la projection spatiale     et te (+9 more)

### Community 9 - "Module architecture overview"
Cohesion: 0.12
Nodes (18): ai_base.py module (BaseAI + HeuristicAI), ai_genetic.py module (GeneticTrainer), ai_search.py module (SearchAI), config.py module (constants & hyperparameters), engine.py module (Line + Engine), Evaluation loop (play/bench sequence diagram), main.py module (CLI: play/train/bench), Neuroevolution training loop diagram (+10 more)

### Community 10 - "Reward-shaping literature and audit plan"
Cohesion: 0.15
Nodes (13): DQN_CURRICULUM_PROB river curriculum, DISCOVER: Automated Curricula for Sparse-Reward RL (arXiv:2505.19850), Ng, Harada & Russell - Policy invariance under reward transformations (ICML 1999), From Sparse to Dense: Toddler-inspired Reward Transition in Goal-Oriented RL (arXiv:2501.17842), RL_SHAPING potential-based reward shaping, Prioritized action plan (lots 1-4), B1: fitness shaping (survival bonus) idea, B3: no-river-first curriculum idea (+5 more)

### Community 11 - "MCTS agent"
Cohesion: 0.31
Nodes (4): _Node, Action, Nœud d'arbre : état (x, y, profondeur) + statistiques UCT., Partie aléatoire biaisée survie depuis (x, y) à la profondeur donnée.          R

### Community 12 - "Neuroevolution agent (cross-doc)"
Cohesion: 0.29
Nodes (7): nn agent (neuroevolution), B2: stochastic softmax policy idea, Genetic algorithm (elitism + tournament + crossover + mutation), Numpy MLP architecture (32-16-5), nn agent, nn agent (neuroevolution), nn agent

### Community 13 - "Sparse-reward exploration research"
Cohesion: 0.40
Nodes (6): Cause 4: poor epsilon-greedy exploration, Exploration in Deep Reinforcement Learning: A Survey, Go-Explore: a New Approach for Hard-Exploration Problems (arXiv:1901.10995), Learning Montezuma's Revenge from a Single Demonstration, A Survey of Deep Reinforcement Learning in Video Games (arXiv:1912.10944), Cause 2: sparse/delayed reward for river crossings

### Community 14 - "Heuristic agent (cross-doc)"
Cohesion: 0.50
Nodes (4): heuristic agent, heuristic agent, heuristic agent (T+1 greedy), heuristic agent

### Community 15 - "MCTS agent (cross-doc)"
Cohesion: 0.50
Nodes (4): mcts agent, mcts agent, mcts agent (UCT rollouts), mcts agent

### Community 16 - "POMDP memory research"
Cohesion: 0.50
Nodes (4): Cause 3: reactive memoryless policy under partial observability, Influence-aware memory architectures for deep RL in POMDPs, Solving Deep Memory POMDPs with Recurrent Policy Gradients, B4: phase-of-next-gap sensor idea

### Community 17 - "PPO agent (cross-doc)"
Cohesion: 0.50
Nodes (4): ppo agent (policy gradient), ppo agent, ppo agent (proximal policy optimization), ppo agent

## Knowledge Gaps
- **65 isolated node(s):** `crossy-ia`, `dqn-shield hybrid agent`, `clone-shield hybrid agent`, `Learning to Undo: Rollback-Augmented RL with Reversibility Signals (arXiv:2510.14503)`, `A Survey of Deep Reinforcement Learning in Video Games (arXiv:1912.10944)` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Engine` connect `BaseAI/HeuristicAI interface` to `Neuroevolution (GA) trainer`, `DQN trainer and agent`, `Search planner and world generation`, `Pygame rendering and duel UI`, `ShieldedAI safety net`, `PPO trainer and agent`, `Sensors and PPO docs`, `MCTS agent`?**
  _High betweenness centrality (0.427) - this node is a cross-community bridge._
- **Why does `ShieldedAI` connect `ShieldedAI safety net` to `BaseAI/HeuristicAI interface`, `Neuroevolution (GA) trainer`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `BaseAI` connect `BaseAI/HeuristicAI interface` to `Neuroevolution (GA) trainer`, `DQN trainer and agent`, `Pygame rendering and duel UI`, `ShieldedAI safety net`, `PPO trainer and agent`, `Sensors and PPO docs`, `Reward-shaping literature and audit plan`, `MCTS agent`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `Engine` (e.g. with `BaseAI` and `HeuristicAI`) actually correct?**
  _`Engine` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `BaseAI` (e.g. with `Engine` and `GeneticTrainer`) actually correct?**
  _`BaseAI` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `MLP` (e.g. with `GeneticTrainer` and `NeuralAI`) actually correct?**
  _`MLP` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Renderer` (e.g. with `BaseAI` and `Engine`) actually correct?**
  _`Renderer` has 2 INFERRED edges - model-reasoned connections that need verification._