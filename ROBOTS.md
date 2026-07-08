# Robots & IA — qui réfléchit comment

Le projet distingue trois familles d'agents. La frontière est nette et assumée :

| Famille | Définition | Agents |
|---|---|---|
| **Robot** | Algorithme déterministe écrit à la main. Il *calcule* — rien n'est appris. Un MCTS seul est un robot : c'est de la planification par échantillonnage, pas de l'apprentissage. | `heuristic`, `search`, `search+`, `mcts` |
| **IA** | Les paramètres de la politique sont *appris* (évolution, gradient de valeur, policy gradient, imitation). | `nn`, `dqn`, `ppo`, `clone` |
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

## `ppo` — le policy gradient (l'algorithme de référence du RL moderne)

Quatrième voie d'apprentissage, qui complète le carré : au lieu de faire évoluer la
politique (nn), d'apprendre une valeur (dqn) ou de copier un expert (clone), PPO optimise
**directement la politique par gradient**, chaque pas d'optimisation étant borné par le
*surrogate clippé* (le ratio politique nouvelle/ancienne est contraint à ±20 % — c'est le
« proximal » de Proximal Policy Optimization). Acteur softmax et critique de valeur
séparés, avantages estimés par GAE(λ), bonus d'entropie contre l'effondrement prématuré.
Mêmes 42 capteurs, même récompense dense que le DQN : les quatre signaux sont comparables
à armes strictement égales.

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
| `heuristic` | robot | 63,6 | 53 | 0/25 | 0,003 | 0,21 |
| `search` | robot | **200** | 200 | **25/25** | 0,29 | 2,09 |
| `search+` | robot | **200** | 200 | **25/25** | 0,30 | 3,60 |
| `mcts` | robot | 149,7 | 193 | 12/25 | 14,0 | 15,9 |
| `nn` | IA | 6,0 | 4 | 0/25 | 0,21 | 1,31 |
| `dqn` | IA | 5,3 | 4 | 0/25 | 0,12 | 1,14 |
| `ppo` | IA | 5,1 | 4 | 0/25 | 0,34 | 0,90 |
| `clone` | IA | 6,3 | 5 | 0/25 | 0,29 | 1,03 |
| `guided` | hybride | **200** | 200 | **25/25** | 0,77 | 2,85 |

Mesure dédiée guided vs search (15 graines) : 107,4 nœuds/coup contre 105,8, soit
**+1,5 % de nœuds** et un temps de décision quasi doublé (0,55 ms vs 0,29 ms).

Lecture — quatre enseignements :

1. **Dans un monde prévisible, la planification écrase l'apprentissage.** Même le MCTS,
   planificateur *approximatif*, gagne 12 parties sur 25 — loin devant toutes les IA —
   tout en étant ~50 fois plus coûteux que `search` pour un résultat inférieur :
   l'échantillonnage n'apporte rien quand le futur se calcule exactement.

2. **Les quatre IA convergent vers le même plafond (~5-6 lignes) par quatre voies
   d'apprentissage totalement différentes** : évolution (nn, 6,0), gradient de valeur
   (dqn, 5,3), policy gradient (ppo, 5,1), imitation supervisée d'un expert parfait
   (clone, 6,3). C'est le résultat le plus instructif du comparatif : le goulot n'est
   PAS le signal d'apprentissage — sinon ces quatre signaux donneraient des plafonds
   différents — mais la **classe de politique réactive** elle-même (42 capteurs → une
   décision par réflexe). Le clone le prouve : même en copiant coup par coup le champion
   invaincu, le réflexe ne franchit pas les rivières, qui exigent des plans à plusieurs
   coups que les capteurs seuls ne résument pas.

3. **`guided` : le résultat négatif propre.** Qualité identique à `search` (25/25), mais
   aucune économie de nœuds (+1,5 %) pour un coût doublé (forwards du prior). La borne
   admissible f et la mémoïzation éliminent déjà l'essentiel du travail ; il ne reste
   rien à guider. AlphaZero guide un MCTS précisément parce que Go et échecs n'offrent
   aucune borne exploitable — ici elle existe, et elle est plus forte qu'un prior appris.

4. **La hiérarchie des robots est une hiérarchie d'horizon** : T+1 (heuristic, 63,6) <
   rollouts aléatoires (mcts, 149,7) < T+15 exact (search, 200). À budget de 20 ms, la
   profondeur d'anticipation exacte est la seule monnaie qui compte.

---

# Le match retour : le monde stochastique

Toute la domination des planificateurs repose sur un privilège : le futur se calcule.
`--noise p` le leur retire. À chaque tick, chaque ligne mobile proche du joueur a une
probabilité p de subir une **turbulence** — un glissement persistant de ±1 case, tiré
d'un flux aléatoire dédié (reproductible par graine). Les agents prédisent avec les
décalages *connus à l'instant t* : leurs plans peuvent être démentis au tick suivant.
La voiture « prévue » à deux cases peut être sur vous ; le tronc visé peut s'être dérobé.

Protocole : mêmes graines que le bench déterministe, bruit 0,1, mêmes modèles entraînés
(en monde déterministe — le test mesure aussi la *robustesse au changement de régime*).

Résultats (bruit 0,1, mêmes 25 graines) — à comparer colonne à colonne avec le tableau
déterministe ci-dessus :

| Agent | Famille | Score moyen (dét. → bruité) | Médiane | Victoires |
|---|---|---:|---:|---:|
| `heuristic` | robot | 63,6 → 24,9 | 14 | 0/25 |
| `search` | robot | **200 → 47,0** | 28 | 0/25 |
| `search+` | robot | 200 → 47,0 | 28 | 0/25 |
| `mcts` | robot | 149,7 → 44,4 | 40 | 0/25 |
| `nn` | IA | 6,0 → **6,2** | 4 | 0/25 |
| `dqn` | IA | 5,3 → 4,9 | 4 | 0/25 |
| `ppo` | IA | 5,1 → **5,4** | 4 | 0/25 |
| `clone` | IA | 6,3 → **6,6** | 6 | 0/25 |
| `guided` | hybride | 200 → 43,6 | 29 | 0/25 |

Trois enseignements :

1. **L'oracle brisé anéantit les rois.** `search` perd 76 % de son score (200 → 47) et
   la totalité de ses victoires : ses plans à 15 coups sont démentis par les turbulences.
   Plus personne ne gagne — le jeu bruité est objectivement plus dur pour tous.

2. **Le MCTS ne prend PAS sa revanche** (44,4 ≈ search 47,0) — résultat contre-intuitif
   et instructif : ses rollouts interrogent le *même* modèle de prédiction trompé que
   l'A*. L'échantillonnage ne protège pas du bruit quand c'est le simulateur interne qui
   est faux ; il faudrait un planificateur qui *modélise* l'incertitude (expectimax,
   marges de sécurité — voir pistes). En revanche, l'écart entre planification exacte et
   échantillonnée disparaît : sous incertitude, calculer juste ne rapporte plus.

3. **Les IA traversent le changement de régime sans broncher** (nn 6,0 → 6,2 ; clone
   6,3 → 6,6 ; ppo 5,1 → 5,4 — trois sur quatre font même mieux), alors qu'elles ont été
   entraînées en monde déterministe. Leurs réflexes statistiques locaux n'ont jamais
   reposé sur la précision du futur. L'écart planificateurs/IA fond de ~33x à ~7x : la
   robustesse est bien le terrain naturel de l'apprentissage — c'est en l'entraînant
   directement en monde bruité (piste 3) qu'on saura s'il peut combler le reste.

# Reproduire les expériences

```bash
uv run python main.py --mode train --generations 100 --curriculum --workers 4   # nn (GA)
uv run python main.py --mode train-dqn --episodes 800                # dqn
uv run python main.py --mode train-ppo                               # ppo
uv run python main.py --mode train-clone                             # clone + guided
uv run python main.py --mode bench --episodes 25                     # comparatif
uv run python main.py --mode bench --episodes 25 --noise 0.1         # match retour
uv run python main.py --mode duel --ai search --ai2 mcts --seed 3    # duel visuel
uv run python main.py --mode duel --ai search --ai2 heuristic --noise 0.1
```

Infrastructure : capteurs en accès tabulaires O(1) (tables précalculées par ligne sur
son cycle exact, équivalence bit à bit vérifiée avec l'implémentation de référence) ;
évaluation du GA parallélisable (`--workers`, résultats strictement identiques au
séquentiel).

# Pistes restantes (non implémentées)

1. **Politique à mémoire** (réseau récurrent ou pile d'observations) — le plafond des
   IA réactives vient peut-être de l'absence d'état interne : impossible de « compter »
   l'attente devant une rivière.
2. **Planification robuste au bruit** — expectimax borné ou replanification avec marge
   de sécurité (éviter les cases adjacentes aux voitures) : rendre aux robots leur
   couronne en monde stochastique.
3. **Entraîner les IA directement en monde bruité** — mesurer si l'apprentissage,
   naturellement statistique, y rattrape son retard sur les planificateurs aveuglés.
4. **Tournoi complet** — matrice duel de tous les agents sur N graines, classement Elo.
