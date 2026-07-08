# Crossy IA — simulateur Crossy Road et comparateur de robots

## Description

Copie fonctionnelle de Crossy Road sur grille, conçue pour faire s'affronter et se comparer
plusieurs types d'intelligences artificielles sur un terrain identique :

| Agent | Famille | Principe |
|---|---|---|
| `heuristic` | robot | glouton T+1, priorités fixes + préférence centre |
| `search` | robot | best-first élagué avec mémoization (type A*), horizon réglable `--horizon` |
| `mcts` | robot | UCT mono-joueur, rollouts sous budget 15 ms |
| `nn` | IA | MLP entraîné par neuroévolution (shaping, softmax, curriculum) |
| `dqn` | IA | Q-learning profond, récompense dense, replay buffer |
| `ppo` | IA | policy gradient actor-critic (surrogate clippé, GAE) |
| `clone` | IA | imitation du planificateur (distillation de `search`) |
| `guided` | hybride | recherche A* départagée par la politique apprise |

**Robots** = algorithmes déterministes écrits à la main (ils calculent) ; **IA** =
paramètres appris (évolution, gradient, imitation) ; **hybride** = planification guidée
par apprentissage. Les règles du jeu (bords infranchissables, caméra qui n'avance que,
portage des troncs, espacements constants) et chaque agent sont expliqués de façon
vulgarisée dans [ROBOTS.md](ROBOTS.md), avec le tableau de l'explosion combinatoire et
les résultats.

Le but du projet est le **comparatif algorithme déterministe vs neuroévolution** : à monde
strictement identique (mêmes graines aléatoires), quelle approche franchit le plus vite et le
plus sûrement les 200 lignes cibles ?

La contrainte absolue est la **performance** : la simulation est totalement découplée de
l'affichage, et chaque IA doit décider en **moins de 20 ms par tick**. En pratique, l'IA de
recherche décide en ~0,4 ms en moyenne (maximum observé ~2 ms), grâce aux optimisations
décrites plus bas.

## Architecture du projet

```
Crossy_Road/
├── config.py        # constantes globales (grille, actions, physique, hyperparamètres)
├── engine.py        # moteur de simulation pur : ticks, obstacles, collisions, clone()
├── sensors.py       # capteurs partagés des agents apprenants (base + phase)
├── neural.py        # MLP numpy : forward, rétropropagation, Adam, génome plat
├── ai_base.py       # interface BaseAI (familles robot/ia/hybride) + heuristique T+1
├── ai_search.py     # recherche prédictive, horizon réglable (heapq + mémoization)
├── ai_mcts.py       # MCTS mono-joueur sous budget (robot)
├── ai_genetic.py    # neuroévolution : GA, shaping, softmax, curriculum, --workers
├── ai_qlearning.py  # DQN : replay buffer, réseau cible, epsilon-greedy
├── ai_ppo.py        # PPO : actor-critic, surrogate clippé, GAE
├── ai_hybrid.py     # imitation du planificateur (clone) + recherche guidée
├── ui.py            # monitoring Pygame (rendu 2D + HUD + duel), aucune logique de jeu
├── main.py          # CLI : play / duel / train* / bench (--noise, --workers)
├── AUDIT.md         # audit du code : schémas d'architecture et plan d'amélioration
├── ROBOTS.md        # robots & IA vulgarisés : stratégies, combinatoire, résultats
└── pyproject.toml   # dépendances gérées par uv (numpy, pygame)
```

Les schémas architectural et fonctionnel (Mermaid) ainsi que le plan d'amélioration priorisé
sont dans [AUDIT.md](AUDIT.md).

### Moteur découplé

`engine.py` ne dépend d'aucune bibliothèque graphique. La boucle est inversée par rapport à
un jeu classique : c'est l'orchestrateur qui demande un coup à l'IA, applique `step(action)`,
puis — *optionnellement* — fait dessiner l'état par `ui.py`. L'entraînement génétique tourne
ainsi en headless à plusieurs dizaines de milliers de ticks par seconde.

Principes clés :

- **Coordonnées entières** `(x, y)` sur une grille de 19 colonnes, hauteur infinie
  (les lignes sont générées paresseusement, à la demande, par un `random.Random(seed)` dédié —
  chaque partie est entièrement reproductible par sa graine).
- **Lignes immuables** (`dataclass frozen`) : type (`SAFE`/`ROAD`/`RIVER`) + paramètres
  cinématiques (direction, période, espacement 6 ou 12, longueur, phase initiale).
- **Arbres statiques dans un `set()`** : test de blocage en O(1).
- **Le monde est indépendant du joueur** : les IA projettent l'avenir via la fonction pure
  `next_state(x, y, tick, action)` sans cloner le moteur (un `clone()` complet existe
  néanmoins pour des projections indépendantes).
- **Mécanique rivière (V2)** : sur une ligne `RIVER`, le joueur doit être superposé à un
  tronc sous peine de noyade ; le tronc porteur applique sa dérive au joueur au tick suivant,
  avant l'action volontaire. Les troncs franchissent le bord par modulo, le joueur porté suit.

## Installation et Exécution

Prérequis : Python ≥ 3.10 et [uv](https://docs.astral.sh/uv/).

```bash
# installation (crée .venv et installe numpy + pygame d'après uv.lock)
uv sync

# session visuelle : n'importe lequel des 9 agents
uv run python main.py --mode play --ai search --seed 3
uv run python main.py --mode play --ai mcts

# duel : deux agents côte à côte sur la même graine
uv run python main.py --mode duel --ai search --ai2 mcts --seed 3

# entraînements headless (chacun sauvegarde son modèle dans models/)
uv run python main.py --mode train --generations 400 --curriculum --workers 4
uv run python main.py --mode train-dqn      # 4000 épisodes (le plus performant)
uv run python main.py --mode train-ppo      # 1500 itérations (~25 min)
uv run python main.py --mode train-clone

# comparatif headless multi-graines de tous les agents disponibles
uv run python main.py --mode bench --episodes 25
uv run python main.py --mode bench --episodes 25 --noise 0.1   # monde stochastique

# monde STOCHASTIQUE (turbulences ±1) : le match retour des planificateurs
uv run python main.py --mode bench --episodes 25 --noise 0.1
```

> Sans uv : `pip install numpy pygame` puis `python main.py ...` fonctionne aussi.

Arguments principaux :

| Argument | Modes | Rôle | Défaut |
|---|---|---|---|
| `--mode` | — | `play`, `train` ou `bench` | `play` |
| `--ai` | play | `heuristic`, `search`, `nn` | `search` |
| `--seed` | play, train | graine de la partie / du GA | `0` |
| `--model` | tous | chemin du modèle `.npz` | `models/best.npz` |
| `--generations` | train | générations du GA | `60` |
| `--population` | train | taille de population | `100` |
| `--episodes` | train, bench | épisodes par évaluation / par IA | `3` / `20` |

En mode `play`, la partie s'arrête à la mort du robot ou à `Y = 200` ; le HUD affiche l'IA
active, le score, le tick, le temps de calcul du dernier coup (coloré selon le budget de
20 ms) et les compteurs d'efficacité (nœuds explorés, branches élaguées…).

## Explication des Algorithmes

### Prédiction du monde en O(1) : la formule modulo

Les obstacles d'une ligne (voitures ou troncs) partagent une cinématique commune. La position
du premier obstacle au tick $t$ est :

$$X_t = \left(X_{initial} + v \cdot \left\lfloor \tfrac{t}{période} \right\rfloor\right) \bmod L$$

avec $v \in \{-1, +1\}$ la direction, la *période* le nombre de ticks entre deux pas
(vitesse effective $v/période$), et $L = 19$ la largeur du terrain. Les autres obstacles s'en
déduisent par décalages fixes de 6 ou 12 cases. **Aucune simulation n'est déroulée** pour
connaître l'état à $t+15$ : c'est un calcul direct.

L'occupation d'une ligne est de plus **mémoïsée** : elle est périodique de période exacte
$période \times L$ ticks, la clé de cache `(y, tick mod cycle)` garantit donc une mémoire
bornée et un accès O(1) (frozenset, test d'appartenance en O(1)).

### IA de recherche : best-first élagué à T+15

Le monde étant déterministe et indépendant du joueur, un état futur est entièrement décrit
par la clé $(x, y, profondeur)$ — la profondeur donnant le tick absolu $t_0 + profondeur$.

- **Mémoization cruciale** : un ensemble `visited` de clés $(x, y, profondeur)$. Toute
  branche retombant sur un état déjà exploré est coupée immédiatement (`continue`) : deux
  chemins différents menant au même état ont exactement le même avenir.
- **File de priorité** (`heapq`) ordonnée par
  $f = y + (horizon - profondeur)$, borne supérieure *admissible* du $Y$ atteignable depuis
  le nœud (le joueur avance d'au plus une ligne par tick). Comme dans A*, le premier nœud
  extrait à l'horizon est donc **garanti optimal en $Y$**, ce qui autorise une sortie
  anticipée. Départage par distance au centre en $X$.
- **Élagage par mort** : toute transition fatale (voiture, noyade, stagnation) tue la
  branche sur place.
- **Repli** : si aucun chemin ne survit 15 ticks, l'IA joue le premier coup de la branche
  qui survit le plus longtemps ; si rien ne survit, `ATTENDRE`.

Complexité : le nombre d'états est borné par $O(L \times horizon^2)$ ; en pratique quelques
centaines de nœuds explorés et un millier de branches élaguées par coup, pour ~0,4 ms de
décision — 50 fois sous le budget de 20 ms.

### Réseau de neurones et capteurs

Perceptron multicouche codé nativement en numpy :

```
entrée (32) ──► cachée (16, tanh) ──► sortie (5 logits) ──► argmax = action
```

Le vecteur d'entrée encode, pour chaque ligne relative $Y-1, Y, Y+1, Y+2, Y+3$ :

1. la **distance normalisée au premier obstacle à gauche** (balayage circulaire, 1.0 = rien) ;
2. la même **à droite** — l'obstacle étant une voiture, un tronc ou un arbre selon le type ;
3. le **type de ligne en one-hot** (`SAFE`, `ROAD`, `RIVER`) ;
4. la **vitesse signée** de la ligne ($direction/période$).

soit $5 \times 6 = 30$ valeurs, plus la position $X$ normalisée dans $[-1, 1]$ et un flag
« sur un tronc ». Le génotype = poids et biais aplatis, soit $32 \times 16 + 16 + 16 \times 5 + 5 = 613$ paramètres.

### Algorithme génétique

- **Population** : 100 individus, initialisation par couche en $1/\sqrt{fan_{in}}$.
- **Fitness** : $Y_{max}$ atteint, moyenné sur plusieurs épisodes dont les graines sont
  tirées au sort à chaque génération (anti-surapprentissage) ; le champion est revalidé sur
  des graines fixes pour suivre un « meilleur global » comparable entre générations.
- **Sélection** : élitisme (5 copies) + tournois de 5.
- **Croisement** : uniforme, probabilité 0,9.
- **Mutation** : bruit gaussien $\mathcal{N}(0, 0{,}3)$ sur 10 % des gènes.
- **Persistance** : meilleur génome sauvegardé/rechargé en `.npz` compressé.

## Protocole de test

Le mode `bench` rejoue les **mêmes graines** (donc des mondes strictement identiques) pour
chaque IA, sans affichage, et rapporte : score moyen, médiane, maximum, taux de victoire
(atteinte de $Y = 200$) et temps de décision moyen/maximum.

```bash
uv run python main.py --mode train --generations 60   # produit models/best.npz
uv run python main.py --mode bench --episodes 25      # tableau comparatif final
```

Résultats observés (25 graines, tous les tableaux détaillés dans
[ROBOTS.md](ROBOTS.md)) : en monde déterministe, les planificateurs exacts `search` et
`guided` gagnent **toutes** leurs parties, le MCTS environ la moitié pour un coût ~50
fois supérieur. Côté apprentissage, avec des réseaux musclés et un entraînement long, le
**DQN décolle nettement** (33 de moyenne, record 130 lignes — il traverse vraiment des
rivières), le PPO double (11), tandis que la neuroévolution et l'imitation restent basses.
Le plafond initial des IA tenait donc surtout à un sous-entraînement, pas à une limite de
principe — mais elles restent loin des 200 de la planification exacte.

En monde **stochastique** (`--noise 0.1`), les planificateurs s'effondrent (`search`
200 → 52, 0 victoire) et se retrouvent au coude-à-coude avec le MCTS (~50). Ils restent
néanmoins **devant toutes les IA** (meilleur apprenant : DQN 25,6) : leur robustesse à ce
type de perturbation est réelle. Contre-intuitivement, entraîner une IA *dans* le bruit
ne l'aide pas — le signal d'apprentissage s'y dégrade plus qu'il ne forge une robustesse.
Quand le futur est calculable, le calcul exact domine ; quand il ne l'est plus, tous
souffrent, et les planificateurs restent en tête.

Chaque partie étant reproductible (`--seed`), tout écart entre IA s'explique par la décision,
jamais par le tirage du monde.
