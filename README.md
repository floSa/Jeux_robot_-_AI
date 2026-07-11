# Crossy IA — simulateur Crossy Road et comparateur de robots

> 🧭 **Pour reprendre le projet** : [STRATEGIE.md](STRATEGIE.md) — la méthode, toutes les
> hypothèses testées (adoptées comme rejetées) avec leurs chiffres, l'état actuel, et la
> feuille de route argumentée vers « 200 à tous les coups ».
>
> 🕸️ **Carte du code** : [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) —
> graphe de connaissances du projet (code + docs), généré par le skill `graphify`. Ouvrir
> `graphify-out/graph.html` dans un navigateur pour l'explorer visuellement, ou poser une
> question en langage naturel sur le code — un assistant équipé du skill interroge le
> graphe directement au lieu de tout relire.

## Description

Copie fonctionnelle de Crossy Road sur grille, conçue pour faire s'affronter et se comparer
plusieurs types d'intelligences artificielles sur un terrain identique :

| Agent | Famille | Principe |
|---|---|---|
| `heuristic` | robot | glouton T+1, priorités fixes + préférence centre |
| `search` | robot | best-first élagué avec mémoization (type A*), horizon réglable `--horizon` |
| `mcts` | robot | UCT mono-joueur, rollouts sous budget 15 ms |
| `nn` | IA | MLP entraîné par neuroévolution (shaping, softmax, curriculum) |
| `dqn` | IA | Q-learning profond, **convolution 1D** sur la vision grille (praticabilité projetée t..t+3) |
| `ppo` | IA | policy gradient actor-critic (surrogate clippé, GAE) |
| `clone` | IA | imitation du planificateur en vision grille + tours DAgger |
| `guided` | hybride | recherche A* départagée par la politique apprise |
| `--shield` | hybride | enveloppe n'importe quel agent d'un filet de sécurité exact |

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
├── sensors.py       # capteurs des agents apprenants (agrégés + vision grille projetée)
├── neural.py        # MLP numpy : forward, rétropropagation, Adam, génome plat
├── ai_base.py       # interface BaseAI (familles robot/ia/hybride) + heuristique T+1
├── ai_search.py     # recherche prédictive, horizon réglable (heapq + mémoization)
├── ai_mcts.py       # MCTS mono-joueur sous budget (robot)
├── ai_genetic.py    # neuroévolution : GA, shaping, softmax, curriculum, --workers
├── ai_qlearning.py  # DQN : replay circulaire, réseau cible, options mesurées, --workers
├── conv_neural.py   # ConvQNet : architecture convolutive 1D (Épisode 7, +45 % vs MLP)
├── ai_ppo.py        # PPO : actor-critic, surrogate clippé, GAE
├── ai_hybrid.py     # imitation (clone, DAgger) + recherche guidée + filet de sécurité
├── shaping.py       # récompenses potentielles (plateforme, alignement)
├── ui.py            # monitoring Pygame (rendu 2D + HUD + duel), aucune logique de jeu
├── main.py          # CLI : play / duel / train* / bench (--noise, --workers)
├── AUDIT.md         # audit du code : schémas d'architecture et plan d'amélioration
├── ROBOTS.md        # robots & IA vulgarisés : stratégies, combinatoire, résultats
├── ANALYSE_IA.md    # l'enquête IA en 3 épisodes : causes, ablations, déblocage (sourcé)
├── STRATEGIE.md     # journal de recherche + guide de reprise + feuille de route vers 200
├── graphify-out/    # graphe de connaissances du projet (skill graphify) — GRAPH_REPORT.md, graph.html
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
- **Lignes immuables** (`dataclass frozen`) : type (`SAFE`/`ROAD`/`RIVER`) + cinématique
  (direction, période) + un **motif de blocs** `(position, longueur)` d'obstacles de
  longueurs variées (voitures 2 / camions 3 / troncs 2-4) qui défile en boucle.
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

# session visuelle : n'importe lequel des agents
uv run python main.py --mode play --ai search --seed 3
uv run python main.py --mode play --ai mcts
uv run python main.py --mode play --ai dqn --shield   # + filet de sécurité (146 -> 195 en moyenne)

# duel : deux agents côte à côte sur la même graine
uv run python main.py --mode duel --ai search --ai2 mcts --seed 3

# entraînements headless (chacun sauvegarde son modèle dans models/)
uv run python main.py --mode train --generations 400 --curriculum --workers 4
uv run python main.py --mode train-dqn --workers 10   # 10 graines en parallèle,
                                                      # garde la mieux validée (~5 min)
uv run python main.py --mode train-ppo --workers 6    # idem, 6 graines (~15 min)
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
| `--workers` | train, train-dqn, train-ppo | éval GA parallèle / entraînements multi-graines (garde le mieux validé) | `0` |
| `--noise` | play, duel, bench, train* | monde stochastique (proba de turbulence ±1) | `0.0` |
| `--horizon` | play, duel, bench | coups anticipés par `search` (1-30) | `15` |
| `--shield` | play, duel, bench | filet de sécurité exact (voir ANALYSE_IA.md, Épisode 4) | désactivé |
| `--shield-depth` | idem | profondeur du filet | `3` |

En mode `play`, la partie s'arrête à la mort du robot ou à `Y = 200` ; le HUD affiche l'IA
active, le score, le tick, le temps de calcul du dernier coup (coloré selon le budget de
20 ms) et les compteurs d'efficacité (nœuds explorés, branches élaguées…).

## Explication des Algorithmes

### Prédiction du monde en O(1) : la formule modulo

Les obstacles d'une ligne (voitures ou troncs) partagent une cinématique commune. La position
du premier obstacle au tick $t$ est :

$$X_t = \left(X_{initial} + v \cdot \left\lfloor \tfrac{t}{période} \right\rfloor\right) \bmod L$$

avec $v \in \{-1, +1\}$ la direction, la *période* le nombre de ticks entre deux pas
(vitesse effective $v/période$), et $L = 19$ la largeur du terrain. Chaque ligne porte un
**motif de blocs** de longueurs variées ; tout le motif subit le même décalage, donc les
autres obstacles s'en déduisent. **Aucune simulation n'est déroulée** pour connaître l'état
à $t+15$ : c'est un calcul direct.

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

Résultats observés (50 graines, monde complet avec routes, troncs mobiles et nénuphars
fixes ; tableaux détaillés dans [ROBOTS.md](ROBOTS.md)) : en monde déterministe, les
planificateurs exacts `search` et `guided` gagnent **toutes** leurs parties (50/50), le
MCTS la moitié pour un coût ~50 fois supérieur. Côté apprentissage, le **DQN — en réflexe
pur, sans recherche au moment de jouer — atteint 146 de moyenne et gagne 28 parties sur
50** (médiane 200, record 200). Le déblocage n'est venu ni d'un réglage ni de la seule
**représentation** (« vision grille » égocentrique projetée aux instants t..t+3, qui avait
déjà fait passer le DQN de 13 à 91), mais d'un changement d'**architecture** :
une convolution 1D qui partage ses poids entre colonnes, là où un MLP dense devait
réapprendre chaque motif à chaque position (+45 % en ablation contrôlée).

**Avec `--shield`** (filet de sécurité : le simulateur exact vérifie chaque coup avant de
le jouer, + sauvetage anti-stagnation), le **DQN monte à 195 de moyenne et 46 victoires sur
50 (92 %)** — à 5 points du plafond théorique de 200, sans changer un seul paramètre du
réseau. Ce gain fond entièrement sous bruit (le dqn+filet retombe exactement au niveau du
dqn seul), pour une raison précise et documentée (le filet suppose que la turbulence
actuelle persiste, comme les planificateurs).

L'enquête complète — pourquoi les IA plafonnaient, ce que les remèdes de la littérature ont
réellement donné (verdict d'ablation contre-intuitif), le déblocage par la vision puis par
l'architecture, et le filet de sécurité — est dans [ANALYSE_IA.md](ANALYSE_IA.md).

En monde **stochastique** (`--noise 0.1`), les planificateurs s'effondrent (`search`
200 → 28) et tout le monde se retrouve dans un mouchoir : cinq approches radicalement
différentes entre 17 et 22. Plus un agent exploitait le déterminisme (projections exactes,
plans à 15 coups), plus le bruit lui coûte ; quand le futur n'est plus calculable, ni la
planification ni le réflexe appris ne gardent d'avantage décisif.

Chaque partie étant reproductible (`--seed`), tout écart entre IA s'explique par la décision,
jamais par le tirage du monde.

---

## Licences & composants

| Composant | Rôle | Licence |
|---|---|---|
| NumPy | Calcul numérique (agents, projections) | BSD-3-Clause |
| pygame | Rendu du jeu | LGPL-2.1 |
| **Ce projet** | Code applicatif | MIT — Copyright (c) 2026 floSa `<à confirmer : aucun fichier LICENSE présent>` |
