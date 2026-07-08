# Robots & IA — qui réfléchit comment

Le projet distingue trois familles d'agents. La frontière est nette et assumée :

| Famille | Définition | Agents |
|---|---|---|
| **Robot** | Algorithme déterministe écrit à la main. Il *calcule* — rien n'est appris. Un MCTS seul est un robot : c'est de la planification par échantillonnage, pas de l'apprentissage. | `heuristic`, `search`, `mcts` |
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

## Les règles, fidèles à Crossy Road

- **On avance ou on meurt.** Le but est de monter (Y croissant) jusqu'à la ligne 200.
  Reculer est autorisé, mais la **caméra ne redescend jamais** : elle reste calée sur le
  point le plus haut atteint. Trop tarder (120 ticks sans nouveau record) = mort.
- **Les bords sont des murs.** On ne peut pas sortir par la gauche ou la droite, et **on
  ne réapparaît jamais de l'autre côté** (pas de téléportation façon Snake). Un déplacement
  volontaire vers un bord est simplement bloqué.
- **Route :** être sur la case d'une voiture = mort.
- **Rivière :** il faut être sur un tronc, sinon noyade. Le tronc **porte** le joueur
  horizontalement au tick suivant. **S'il vous porte hors de l'écran, vous mourez
  immédiatement** — aucune action ne vous rattrape ce tick-là. Il faut sauter avant.
- **Trottoir :** sûr, mais des arbres bloquent certaines cases.
- **Espacements constants.** Sur une même ligne, l'écart entre les véhicules (ou les
  troncs) est toujours le même (6 ou 12 cases) et le défilement est régulier. C'est ce qui
  rend le futur *anticipable* : un robot peut calculer où sera chaque obstacle, une IA peut
  apprendre le rythme.

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

Le paramètre se règle au lancement : `--horizon 3` pour un robot myope, `--horizon 15`
pour le voir loin. Mesuré : à T+3 il rate quelques rivières (13/15), à T+5 et au-delà il
gagne tout — l'horizon est le curseur exact entre myopie et clairvoyance.

## `mcts` — jouer des milliers de parties imaginaires (Monte-Carlo Tree Search)

**Comment il décide, concrètement.** À chaque tick, le MCTS dispose de ~15 ms. Pendant ce
temps, il ne calcule pas la meilleure suite de coups comme `search` : il **imagine des
milliers de parties jouées presque au hasard** à partir de la position actuelle. Une
« partie imaginaire » (un *rollout*) part de maintenant, joue une trentaine de coups plus
ou moins aléatoires, et note jusqu'où elle est montée avant de mourir. En répétant ça des
centaines de fois, il apprend quel **premier coup** mène en moyenne le plus loin — et
c'est ce premier coup qu'il joue. La finesse (formule UCB1) est qu'il ne tire pas au sort
uniformément : il rejoue plus souvent les premiers coups qui ont déjà bien marché, tout
en gardant un peu d'exploration sur les autres. « Garder la meilleure », c'est donc :
**garder le premier coup dont les parties imaginaires ont le mieux fini en moyenne.**

**Est-ce pertinent pour ce jeu ? Franchement, non — et c'est justement l'intérêt de le
tester.** Le MCTS est fait pour les jeux où l'on ne PEUT PAS tout calculer (le Go a plus
de positions que d'atomes dans l'univers) : on échantillonne parce que l'exhaustif est
hors de portée. Or ici, l'exhaustif EST à portée — `search` explore tous les futurs utiles
en 0,3 ms. Utiliser du hasard là où le calcul exact tient dans le budget, c'est se priver
volontairement de la bonne réponse. Résultat mesuré : le MCTS fait moins bien que `search`
pour 50 fois plus de temps de calcul. **Il ne devient pertinent que si le monde cesse
d'être calculable** — c'est exactement ce que teste le mode `--noise` plus bas, où il
prend une revanche instructive : sous incertitude, moyenner sur mille futurs échantillonnés
bat un plan exact unique et fragile. C'est pour l'incertain que le MCTS est fait.

Implémentation : UCT mono-joueur, sélection UCB1, rollouts biaisés survie (le coup AVANCER
est tenté en premier une fois sur deux), récompense = nombre de lignes gagnées, garde
temporelle adaptative pour ne jamais dépasser le budget.

---

# Deuxième famille : les IA (apprentissage)

Les quatre IA lisent les mêmes **42 capteurs** : distances gauche/droite aux obstacles sur
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

## `guided` — la recherche aidée par l'IA (le patron AlphaZero en miniature)

**L'idée en une phrase :** c'est `search`, mais quand il doit choisir par quel coup
commencer son exploration, il demande son avis à l'IA `clone`.

**Ce que « trier les coups » veut dire, concrètement.** `search` explore l'arbre des
futurs coup par coup. À chaque étape, il a jusqu'à 5 coups à examiner, et il doit décider
**dans quel ordre** les explorer. Un bon ordre — commencer par les coups les plus
prometteurs — permet de trouver la solution plus vite et d'élaguer le reste. `search`
seul trie par une règle simple (« reste près du centre »). `guided` remplace cette règle,
près de la racine, par la **recommandation de l'IU** : le réseau `clone` regarde la
position et dit « à ta place, je jouerais plutôt AVANCER, puis DROITE… » ; `search`
explore alors les coups dans cet ordre-là. L'IA ne décide pas du coup final — elle ne fait
que *suggérer l'ordre d'exploration*. `search` garde le dernier mot et sa garantie de
trouver le chemin optimal ; l'IA sert juste à ce qu'il y arrive en explorant moins.

C'est exactement le principe d'AlphaZero (un réseau qui souffle au moteur de recherche
quels coups regarder en priorité), en modèle réduit.

**Résultat mesuré, en toute honnêteté :** ça ne sert quasiment à rien ici (voir tableau) —
la suggestion de l'IA ne fait pas explorer moins de nœuds. Et l'explication est le vrai
enseignement : `search` est déjà tellement efficace (sa borne mathématique + sa mémoire
des cases déjà vues éliminent 99,99 % du travail) qu'il ne reste presque rien à optimiser.
AlphaZero aide un moteur de recherche parce qu'au Go il n'existe aucune borne pour élaguer ;
ici cette borne existe, et elle est plus forte que n'importe quel conseil appris.

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
| `heuristic` | robot | 63,6 | 53 | 0/25 | 0,003 | 0,17 |
| `search` | robot | **200** | 200 | **25/25** | 0,31 | 5,73 |
| `mcts` | robot | 157,2 | 200 | 14/25 | 14,0 | 17,1 |
| `nn` | IA | 8,1 | 7 | 0/25 | 0,26 | 1,70 |
| `dqn` | IA | 4,6 | 4 | 0/25 | 0,09 | 1,26 |
| `ppo` | IA | 5,1 | 4 | 0/25 | 0,36 | 1,11 |
| `clone` | IA | 7,1 | 6 | 0/25 | 0,26 | 1,06 |
| `guided` | hybride | **200** | 200 | **25/25** | 0,83 | 5,60 |

Mesure dédiée guided vs search (15 graines) : 113,9 nœuds/coup contre 112,5, soit
**+1,2 % de nœuds** et un temps de décision plus que doublé (l'IA est interrogée en plus) —
l'aide de l'IA fait explorer *plus*, pas moins.

Lecture — quatre enseignements :

1. **Dans un monde prévisible, la planification écrase l'apprentissage.** Même le MCTS,
   planificateur *approximatif*, gagne 14 parties sur 25 — loin devant toutes les IA —
   tout en étant ~50 fois plus coûteux que `search` pour un résultat inférieur :
   l'échantillonnage n'apporte rien quand le futur se calcule exactement.

2. **Les quatre IA convergent vers le même plafond (~5-8 lignes) par quatre voies
   d'apprentissage totalement différentes** : évolution (nn, 8,1), gradient de valeur
   (dqn, 4,6), policy gradient (ppo, 5,1), imitation supervisée d'un expert parfait
   (clone, 7,1). C'est le résultat le plus instructif du comparatif : le goulot n'est
   PAS le signal d'apprentissage — sinon ces quatre signaux donneraient des plafonds
   très différents — mais la **classe de politique réactive** elle-même (42 capteurs → une
   décision par réflexe). Le clone le prouve : même en copiant coup par coup le champion
   invaincu, le réflexe ne franchit pas les rivières, qui exigent des plans à plusieurs
   coups que les capteurs seuls ne résument pas.

3. **`guided` : le résultat négatif propre.** Qualité identique à `search` (25/25), mais
   aucune économie de nœuds (+1,2 %) pour un coût plus que doublé (l'IA est interrogée à
   chaque nœud proche de la racine). La borne admissible f et la mémoïzation éliminent
   déjà l'essentiel du travail ; il ne reste rien à guider. AlphaZero guide un MCTS
   précisément parce que Go et échecs n'offrent aucune borne exploitable — ici elle
   existe, et elle est plus forte qu'un conseil appris.

4. **La hiérarchie des robots est une hiérarchie d'horizon** : T+1 (heuristic, 63,6) <
   rollouts aléatoires (mcts, 157,2) < T+15 exact (search, 200). À budget de 20 ms, la
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
| `search` | robot | **200 → 52,2** | 41 | 0/25 |
| `mcts` | robot | 157,2 → **67,6** | 41 | **2/25** |
| `nn` | IA | 8,1 → **8,7** | 8 | 0/25 |
| `dqn` | IA | 4,6 → 4,2 | 4 | 0/25 |
| `ppo` | IA | 5,1 → **5,4** | 4 | 0/25 |
| `clone` | IA | 7,1 → 6,2 | 6 | 0/25 |
| `guided` | hybride | 200 → 49,9 | 41 | 0/25 |

Trois enseignements :

1. **L'oracle brisé détrône les rois.** `search` perd 74 % de son score (200 → 52) et la
   totalité de ses victoires : ses plans à 15 coups sont démentis par les turbulences. Le
   jeu bruité est objectivement plus dur pour tous.

2. **Cette fois, le MCTS prend sa revanche — modestement mais nettement.** Il devient le
   meilleur agent du monde bruité (67,6 contre 52,2 pour `search`) et remporte même 2
   parties quand plus aucun autre n'y arrive. La raison est belle : le MCTS **moyenne sur
   des milliers de futurs échantillonnés**, alors que `search` mise tout sur *un* plan
   exact — donc fragile dès que le futur dévie. Sous incertitude, l'agent qui répartit ses
   paris bat celui qui parie juste mais tout sur un seul scénario. C'est exactement pour ça
   que le MCTS existe : il est fait pour l'incertain, pas pour le calculable.

3. **Les IA traversent le changement de régime sans broncher** (nn 8,1 → 8,7 ; ppo
   5,1 → 5,4 font même mieux ; les autres bougent à peine), alors qu'elles ont été
   entraînées en monde déterministe. Leurs réflexes statistiques locaux n'ont jamais
   reposé sur la précision du futur. L'écart planificateurs/IA fond de ~25x à ~8x : la
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
