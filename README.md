# Crossy IA — simulateur Crossy Road et comparateur de robots

## Description

Copie fonctionnelle de Crossy Road sur grille, conçue pour faire s'affronter et se comparer
plusieurs types d'intelligences artificielles sur un terrain identique :

| IA | Famille | Horizon | Principe |
|---|---|---|---|
| `heuristic` | déterministe gloutonne | T+1 | priorités fixes + survie immédiate |
| `search` | déterministe prédictive | T+15 | best-first élagué avec mémoization (type A*) |
| `nn` | neuroévolution | réactif | MLP numpy entraîné par algorithme génétique |

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
├── config.py       # constantes globales (grille, actions, physique, hyperparamètres)
├── engine.py       # moteur de simulation pur : ticks, obstacles, collisions, clone()
├── ai_base.py      # interface BaseAI + IA heuristique T+1
├── ai_search.py    # IA de recherche prédictive T+15 (heapq + mémoization)
├── ai_genetic.py   # MLP numpy, capteurs, algorithme génétique, persistance .npz
├── ui.py           # monitoring Pygame (rendu 2D + HUD), aucune logique de jeu
├── main.py         # orchestrateur CLI : play / train / bench
├── AUDIT.md        # audit du code : schémas d'architecture et plan d'amélioration
└── pyproject.toml  # dépendances gérées par uv (numpy, pygame)
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

# session visuelle : IA heuristique, de recherche ou réseau de neurones
uv run python main.py --mode play --ai heuristic
uv run python main.py --mode play --ai search --seed 3
uv run python main.py --mode play --ai nn            # nécessite un modèle entraîné

# entraînement génétique headless (sauvegarde models/best.npz)
uv run python main.py --mode train --generations 60

# comparatif headless multi-graines des trois IA
uv run python main.py --mode bench --episodes 25
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

Ordre de grandeur observé (20 graines, entraînement de 40 générations) : l'IA de recherche
gagne la grande majorité des parties (score médian 200), l'heuristique meurt typiquement
entre 20 et 80 lignes faute d'anticipation, et le réseau de neurones progresse avec le nombre
de générations — la neuroévolution reste très en deçà de la recherche exhaustive à budget de
calcul comparable, ce qui est précisément l'objet du comparatif.

Chaque partie étant reproductible (`--seed`), tout écart entre IA s'explique par la décision,
jamais par le tirage du monde.
