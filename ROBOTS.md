# Robots & IA — qui réfléchit comment

Le projet distingue trois familles d'agents. La frontière est nette et assumée :

| Famille | Définition | Agents |
|---|---|---|
| **Robot** | Algorithme déterministe écrit à la main. Il *calcule* — rien n'est appris. Un MCTS seul est un robot : c'est de la planification par échantillonnage, pas de l'apprentissage. | `heuristic`, `search`, `search+`, `mcts` |
| **IA** | Les paramètres de la politique sont *appris* (évolution, gradient, imitation). | `nn`, `dqn`, `clone` |
| **Hybride** | Planification guidée par un modèle appris. | `guided` |

Pour l'architecture technique, voir [AUDIT.md](AUDIT.md) ; pour les formules du moteur,
le [README](README.md).

## Les règles du jeu du point de vue de l'agent

À chaque tick, exactement **un coup parmi cinq** :

| Coup | Delta (dx, dy) |
|---|---|
| AVANCER | (0, +1) |
| RECULER | (0, −1) |
| GAUCHE | (−1, 0) |
| DROITE | (+1, 0) |
| ATTENDRE | (0, 0) |

Le monde est **déterministe et parfaitement prévisible** : la position de toute voiture
ou tronc à tout tick futur se calcule en O(1) (formule modulo). Toute la question du
projet est là : *quand le futur est calculable, que vaut l'apprentissage face au calcul ?*

---

# Première famille : les ROBOTS (algorithmes)

## `heuristic` — le glouton

Le robot le plus simple : il ne regarde que le coup suivant (T+1).

> Si on peut avancer, on avance. Sinon, parmi attendre / gauche / droite sans danger,
> on prend l'option qui **rapproche le plus du centre** (moins dangereux que les bords ;
> à égalité, on attend). En dernier recours, on recule.

Un coup impossible (arbre, bord) est écarté d'office. La préférence pour le centre n'est
pas un détail : une version antérieure plaçait ATTENDRE au-dessus des pas de côté, et
finissait bloquée derrière un arbre dans 64 % des parties (« avancer est impossible,
donc j'attends »… pour l'éternité). La fusion attendre/latéraux a doublé son score.

Limite indépassable : à T+1, impossible de *planifier* une traversée de rivière — la
plupart de ses parties finissent en mort par stagnation.

## `search` — le prévoyant

Les 5 coups du glouton sont des chemins de longueur 1. Rien n'empêche de considérer
tous les chemins de longueur N : on simule chacun tick par tick (le futur est connu !),
on garde ceux où l'on survit, on choisit celui qui avance le plus (à égalité : le plus
près du centre), on joue son **premier coup**, et on réévalue au tick suivant.

### L'explosion combinatoire et son remède

5 choix par tick → 5^N chemins de longueur N :

| N coups d'avance | chemins possibles (5^N) | cases-instants à visiter (avec mémoïzation) |
|---:|---:|---:|
| 1 | 5 | 6 |
| 2 | 25 | 19 |
| 3 | 125 | 44 |
| 5 | 3 125 | 146 |
| 7 | 78 125 | 344 |
| 10 | 9 765 625 | 891 |
| 15 | **30 517 578 125** (~30 milliards) | **2 736** |

L'argument-clé : si deux chemins A et B arrivent sur la **même case au même instant**,
leur avenir est identique. Si A (déjà simulé) est valide, B ne peut jamais faire mieux —
il est inutile de le simuler. Généralisation : dès qu'une case-instant `(x, y, tick)` a
été atteinte une fois, tout chemin qui y repasse autrement est coupé. Comme presque tous
les chemins repassent par des cases déjà vues, l'écrasante majorité disparaît :
croissance **cubique** (≈ ⅔·N³) au lieu d'**exponentielle** (5^N). En pratique ~400
états par coup (les cases mortelles élaguent leurs sous-arbres entiers).

Deux raffinements : une file de priorité sur la borne admissible
f = y + (coups restants) — le premier chemin complet extrait est *garanti optimal*,
arrêt anticipé — et un repli « survivre le plus longtemps » si rien n'atteint l'horizon.
Coût mesuré : **~0,3 ms par décision** (budget 20 ms).

## `search+` — l'horizon adaptatif

Si le meilleur chemin à T+15 fait du surplace, symptôme d'un alignement de troncs
au-delà de l'horizon, la recherche est relancée à T+30, T+45… jusqu'à T+90, sous une
garde stricte de 70 % du budget. Filet de sécurité de `search`.

## `mcts` — le joueur de hasard (Monte-Carlo Tree Search)

**Pourquoi l'inclure, et pourquoi ce n'est pas une IA.** Le MCTS ne contient aucun
paramètre appris : il explore l'arbre des coups en jouant des parties aléatoires
(rollouts) et concentre son budget sur les branches prometteuses (formule UCB1). C'est
un *robot* stochastique. Sa pertinence ici est celle d'un témoin expérimental : à budget
identique (15 ms/tick), que vaut un planificateur par échantillonnage face à la
recherche exhaustive dans un monde déterministe ? Réponse attendue et mesurée : il perd
— l'échantillonnage n'apporte rien quand on peut tout calculer ; il deviendrait pertinent
si le jeu devenait aléatoire (transitions bruitées), là où `search` s'effondre.

Implémentation : UCT mono-joueur, sélection UCB1, rollouts biaisés survie (AVANCER tenté
en premier une fois sur deux), récompense = progression en Y, garde temporelle adaptative.

---

# Deuxième famille : les IA (apprentissage)

Les trois IA lisent les mêmes **42 capteurs** : distances gauche/droite aux obstacles sur
5 lignes, type de ligne (one-hot), vitesse signée, position X, flag « sur un tronc », et
— nouveauté décisive — la **phase** : pour chaque ligne, le délai avant que la case du
joueur soit occupée puis libérée. C'est l'information temporelle que `search` calcule via
la formule modulo ; sans elle, un réseau ne peut pas savoir *quand* sauter.

## `nn` — la neuroévolution (sélection darwinienne)

100 réseaux (42 → 16 → 5, 773 paramètres) jouent ; les meilleurs se reproduisent
(croisement uniforme), mutent (bruit gaussien), et on recommence. Trois améliorations
issues du plateau mesuré (~7 lignes après 300 générations de la v1) :

- **Fitness shaping** : Y max + 0,005/tick survécu — densifie le signal sans récompenser
  le camping (une stagnation complète vaut 0,6, moins qu'une ligne).
- **Softmax à température (0,5)** pendant l'entraînement : casse les boucles
  comportementales des politiques argmax figées ; l'évaluation reste déterministe.
- **Curriculum** (`--curriculum`) : 40 % des générations sur mondes sans rivière, puis
  mondes complets.

## `dqn` — le Q-learning profond (gradient)

La comparaison canonique avec la neuroévolution : mêmes capteurs, même taille de réseau,
mais un **signal d'apprentissage dense** au lieu du seul score final. Le réseau apprend
la *valeur* Q(état, action) par équation de Bellman : +1 par ligne nouvelle, −0,01 par
tick, −1 à la mort, +5 à la victoire, propagés par récompense actualisée
(γ = 0,97). Ingrédients classiques, tous en numpy pur : replay buffer (50 000
transitions), réseau cible synchronisé, ε-greedy décroissant, Adam sur perte TD.

## `clone` — l'imitation du planificateur (distillation)

L'idée la plus « recherche » du lot : `search` est un expert parfait — peut-on le
**distiller** dans un réflexe ? On rejoue des parties de l'expert, on enregistre
(capteurs, coup choisi) à chaque tick, et on entraîne une politique par entropie croisée
à prédire le coup de l'expert. Le clone joue ensuite *sans aucune recherche* : ce que le
planificateur calcule en explorant ~400 états, le clone le devine en un forward de
réseau (~10 µs). La précision de prédiction et le score en jeu mesurent ce qui, dans la
décision du planificateur, est « compressible » en réflexe — et ce qui exige
irréductiblement du calcul.

---

# Troisième famille : l'HYBRIDE

## `guided` — la recherche guidée par politique

Le patron AlphaZero en miniature : l'A* garde sa borne admissible (l'optimalité), mais
le départage des ex æquo près de la racine suit la politique apprise (celle du clone) au
lieu de la distance au centre. Objectif mesuré : réduire les nœuds explorés à qualité
égale. Le prior n'est évalué qu'aux profondeurs ≤ 2 : plus profond, le coût d'un forward
par nœud dépasserait le gain — la recherche entière ne coûte que ~0,3 ms.

**Résultat mesuré (honnêteté scientifique)** : voir tableau ci-dessous. Si la réduction
de nœuds est marginale, l'explication est intéressante en soi : la borne admissible et
la mémoïzation éliminent déjà 99,99 % du travail — il ne reste presque rien à guider.
AlphaZero guide MCTS précisément parce que Go/échecs n'ont PAS de borne admissible
exploitable ; ici elle existe, et elle est plus forte qu'un prior appris.

---

# L'enquête : quand le coupable était le monde

Longtemps, `search` perdait ~20 % de ses parties par stagnation devant certaines
rivières. Un BFS exhaustif sans limite d'horizon depuis toutes les positions réelles du
robot bloqué a prouvé que le franchissement était *impossible* : deux rivières
adjacentes au tronc unique, de même direction et même vitesse — écart constant pour
l'éternité, le saut n'existait jamais. Le générateur garantit désormais des directions
opposées et un espacement dense pour les rivières empilées. `search` est passé de ~78 à
**100 victoires sur 100**. Moralité : avant d'accuser l'algorithme, prouver que le
problème a une solution.

---

# Résultats — bench officiel (25 graines identiques pour tous)

`uv run python main.py --mode bench --episodes 25` :

| Agent | Famille | Score moyen | Médiane | Victoires | ms/décision (moy) | ms max |
|---|---|---:|---:|---:|---:|---:|
| `heuristic` | robot | 63,6 | 53 | 0/25 | 0,003 | 0,32 |
| `search` | robot | **200** | 200 | **25/25** | 0,29 | 3,98 |
| `search+` | robot | **200** | 200 | **25/25** | 0,30 | 3,60 |
| `mcts` | robot | 174,4 | 200 | 16/25 | 14,0 | 18,4 |
| `nn` | IA | 6,0 | 4 | 0/25 | 0,019 | 0,27 |
| `dqn` | IA | 5,3 | 4 | 0/25 | 0,019 | 0,27 |
| `clone` | IA | 6,3 | 5 | 0/25 | 0,019 | 0,25 |
| `guided` | hybride | **200** | 200 | **25/25** | 0,57 | 4,43 |

Mesure dédiée guided vs search (15 graines) : 107,4 nœuds/coup contre 105,8, soit
**+1,5 % de nœuds** et un temps de décision quasi doublé (0,55 ms vs 0,29 ms).

Lecture — quatre enseignements :

1. **Dans un monde prévisible, la planification écrase l'apprentissage.** Même le MCTS,
   planificateur *approximatif*, gagne 16 parties sur 25 — loin devant toutes les IA —
   tout en étant 50 fois plus coûteux que `search` pour un résultat inférieur :
   l'échantillonnage n'apporte rien quand le futur se calcule exactement.

2. **Les trois IA convergent vers le même plafond (~5-6 lignes) par trois voies
   d'apprentissage totalement différentes** : évolution (nn, 6,0), gradient de valeur
   (dqn, 5,3), imitation supervisée d'un expert parfait (clone, 6,3). C'est le résultat
   le plus instructif du comparatif : le goulot n'est PAS le signal d'apprentissage —
   sinon ces trois signaux donneraient des plafonds différents — mais la **classe de
   politique réactive** elle-même (42 capteurs → une décision par réflexe). Le clone le
   prouve : même en copiant coup par coup le champion invaincu, le réflexe ne franchit
   pas les rivières, qui exigent des plans à plusieurs coups que les capteurs seuls ne
   résument pas.

3. **`guided` : le résultat négatif propre.** Qualité identique à `search` (25/25), mais
   aucune économie de nœuds (+1,5 %) pour un coût doublé (forwards du prior). La borne
   admissible f et la mémoïzation éliminent déjà l'essentiel du travail ; il ne reste
   rien à guider. AlphaZero guide un MCTS précisément parce que Go et échecs n'offrent
   aucune borne exploitable — ici elle existe, et elle est plus forte qu'un prior appris.

4. **La hiérarchie des robots est une hiérarchie d'horizon** : T+1 (heuristic, 63,6) <
   rollouts aléatoires (mcts, 174,4) < T+15 exact (search, 200). À budget de 20 ms, la
   profondeur d'anticipation exacte est la seule monnaie qui compte.

---

# Reproduire les entraînements

```bash
uv run python main.py --mode train --generations 100 --curriculum   # nn  (GA)
uv run python main.py --mode train-dqn --episodes 800               # dqn
uv run python main.py --mode train-clone                            # clone + guided
uv run python main.py --mode bench --episodes 25                    # comparatif
```

# Pistes restantes (non implémentées)

1. **PPO / policy gradient** — troisième voie d'apprentissage (après évolution et
   valeur), la plus utilisée en pratique moderne.
2. **Monde stochastique** — bruiter les transitions (troncs qui coulent, vitesses
   variables) : `search` perd son oracle, le MCTS et les IA prennent leur revanche ;
   c'est l'extension qui rendrait le comparatif complet.
3. **Vectorisation numpy des capteurs + évaluation parallèle** (multiprocessing) —
   accélérer d'un ordre de grandeur les entraînements.
4. **Duel en direct** — mode UI affichant deux agents côte à côte sur la même graine.
