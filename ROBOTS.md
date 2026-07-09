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
- **Route :** être sur la case d'un véhicule = mort. Les **voitures occupent 2 cases**, les
  **camions 3**, de longueurs mélangées sur une même ligne, séparés par des trous.
- **Rivière :** il faut être sur une plateforme, sinon noyade. Deux sortes de lignes d'eau :
  - **Troncs mobiles** (2 à 4 cases, longueurs variées) : ils **portent** le joueur
    horizontalement au tick suivant ; **s'ils vous portent hors de l'écran, mort immédiate**
    (« sortie d'écran »), aucune action ne vous rattrape ce tick-là — il faut sauter avant.
  - **Nénuphars** : plateformes **fixes** et isolées (5-7 par ligne), à franchir de saut en
    saut comme des pierres de gué immobiles.
- **Trottoir :** sûr, mais des arbres bloquent certaines cases — **jamais plus de 2 collés**,
  et il n'y a **qu'une seule ligne d'herbe à la fois**.
- **Le motif défile en boucle.** Sur une ligne donnée, l'agencement des obstacles (leurs
  positions et longueurs) est **fixe et se reproduit à l'identique** en glissant : c'est la
  seule règle du défilement, et c'est elle qui rend le futur *anticipable*. Un robot peut
  calculer où sera chaque obstacle à n'importe quel instant, une IA peut apprendre le rythme.

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

100 réseaux (42 → 32 → 5) jouent ; les meilleurs se reproduisent (croisement uniforme),
mutent (bruit gaussien), et on recommence. À noter : agrandir le réseau n'a pas aidé
l'évolution (elle a même un peu régressé) — une sélection aveugle explore mal un grand
espace de poids, contrairement aux méthodes par gradient. Trois améliorations issues du
plateau mesuré (~7 lignes après 300 générations de la v1) :

- **Fitness shaping** : Y max + 0,005/tick survécu — densifie le signal sans récompenser
  le camping (une stagnation complète vaut 0,6, moins qu'une ligne).
- **Softmax à température (0,5)** pendant l'entraînement : casse les boucles
  comportementales des politiques argmax figées ; l'évaluation reste déterministe.
- **Curriculum** (`--curriculum`) : 40 % des générations sur mondes sans rivière, puis
  mondes complets.

## `dqn` — le Q-learning profond (gradient)

La comparaison canonique avec la neuroévolution : mêmes capteurs, mais un **signal
d'apprentissage dense** au lieu du seul score final. Le réseau (42 → 64 → 5) apprend la
*valeur* Q(état, action) par équation de Bellman : +1 par ligne nouvelle, −0,01 par tick,
−1 à la mort, +5 à la victoire, propagés par récompense actualisée (γ = 0,97). Ingrédients
classiques, tous en numpy pur : replay buffer circulaire (100 000 transitions), réseau
cible synchronisé, ε-greedy décroissant, Adam sur perte TD.

**C'est la seule IA qui traverse réellement des rivières**, et elle a bénéficié de deux
chantiers successifs. D'abord la capacité et l'entraînement long (déjà ~2,5× la v1).
Ensuite un chantier « littérature » complet : récompense potentielle, mémoire par pile
d'observations, retours multi-pas, curriculum rivières, Double DQN — tous implémentés,
tous pilotables par la config… et tous mesurés par **ablation** comme neutres à nuisibles
à notre budget d'entraînement (l'histoire complète, contre-intuitive et sourcée, est dans
[ANALYSE_IA.md](ANALYSE_IA.md)). Ce qui a réellement payé : constater que **la graine
d'entraînement fait varier le score final du simple au double**, et donc entraîner
**10 graines en parallèle** (10 000 épisodes chacune, `--workers 10`) en gardant la mieux
validée sur des graines fixes.

## `ppo` — le policy gradient (l'algorithme de référence du RL moderne)

Quatrième voie d'apprentissage, qui complète le carré : au lieu de faire évoluer la
politique (nn), d'apprendre une valeur (dqn) ou de copier un expert (clone), PPO optimise
**directement la politique par gradient**, chaque pas d'optimisation étant borné par le
*surrogate clippé* (le ratio politique nouvelle/ancienne est contraint à ±20 % — c'est le
« proximal » de Proximal Policy Optimization). Acteur softmax et critique de valeur
séparés, avantages estimés par GAE(λ), bonus d'entropie contre l'effondrement prématuré.
Mêmes 42 capteurs, même récompense dense que le DQN : les quatre signaux sont comparables
à armes strictement égales.

## `clone` — copier le maître (imitation / distillation)

**Ce que `clone` n'est PAS.** Il ne mémorise aucune grille et ne compare pas la situation
présente à des situations déjà vues. Aucune base de données, aucun « si je revois cette
grille, je rejoue ce coup ». Ce serait impossible : il y a bien trop de grilles différentes.

**Ce que `clone` EST.** Un réseau de neurones (le même genre que `nn`, `dqn`, `ppo`),
entraîné à **imiter `search`** comme un élève copie un maître. La recette, en trois temps :

1. On regarde `search` jouer des milliers de parties. À chaque tick, on note une fiche :
   « voilà ce que les 42 capteurs voyaient → voilà le coup que `search` a joué ».
2. On entraîne le réseau sur ces dizaines de milliers de fiches à **prédire le coup du
   maître à partir des seuls capteurs** (apprentissage supervisé, entropie croisée).
3. Une fois entraîné, le réseau **généralise** : face à une situation *nouvelle* (jamais
   dans les fiches), il propose le coup que `search` aurait probablement joué — en un seul
   calcul de réseau (~10 µs), **sans aucune recherche**.

L'intérêt scientifique : `search` explore ~400 états par décision ; le clone tente de
compresser tout ce raisonnement en un réflexe instantané. Sa précision (il rejoue ~78 % des
coups du maître) et son score en jeu mesurent la part de la décision de `search` qui est
« compressible » en réflexe — et la part qui exige irréductiblement de calculer. Réponse :
il imite bien coup par coup mais reste faible en jeu (~6 lignes), car une seule erreur
devant une rivière est fatale, et le réflexe ne rattrape pas comme le fait la recherche.

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

# Résultats — bench officiel (50 graines identiques pour tous)

`uv run python main.py --mode bench --episodes 50` (réseaux : nn 32, dqn/ppo 64 neurones
cachés ; entraînements : dqn **10 graines × 10 000 épisodes** puis sélection sur validation,
ppo 6 graines × 1500 itérations, GA 400 générations) :

Sur **50 graines**, monde complet (routes, troncs mobiles **et nénuphars fixes**) :

| Agent | Famille | Score moyen | Médiane | Record | Victoires | ms/décision |
|---|---|---:|---:|---:|---:|---:|
| `heuristic` | robot | 51,4 | 38 | 200 | 1/50 | 0,003 |
| `search` | robot | **200** | 200 | 200 | **50/50** | 0,31 |
| `mcts` | robot | 134,3 | 200 | 200 | 25/50 | 14,0 |
| `nn` | IA | 4,7 | 4 | 10 | 0/50 | 0,45 |
| `dqn` | IA | **29,8** | 22 | **168** | 0/50 | 0,10 |
| `ppo` | IA | 7,0 | 5 | 33 | 0/50 | 0,38 |
| `clone` | IA | 7,8 | 6 | 32 | 0/50 | 0,30 |
| `guided` | hybride | **200** | 200 | 200 | **50/50** | 0,87 |

*(`dqn` et `ppo` : modèles réentraînés par la procédure multi-graines — voir
[ANALYSE_IA.md](ANALYSE_IA.md), Épisode 2. Les autres agents sont inchangés.)*

Lecture — quatre enseignements :

1. **Dans un monde prévisible, la planification écrase l'apprentissage.** `search` et
   `guided` gagnent **tout** (50/50) ; même le MCTS, planificateur *approximatif*, gagne
   25/50 — loin devant toutes les IA — tout en étant ~50 fois plus coûteux que `search` pour
   un résultat inférieur. L'échantillonnage n'apporte rien quand le futur se calcule exactement.

2. **La difficulté est une question d'anticipation, pas de richesse.** L'heuristique (T+1)
   se débrouille bien sur les routes (trous larges → 51 de moyenne, un record à 200), mais
   trébuche dès qu'il faut *planifier* : rivières à troncs et surtout **nénuphars**, où sauter
   de plateforme fixe en plateforme fixe demande de viser plusieurs coups à l'avance. Ajouter
   des nénuphars a d'ailleurs fait redescendre son score (97 → 51) : ce qui coûte à un agent
   myope, c'est tout ce qui exige un plan, pas la variété en soi.

3. **Le DQN est de loin le seul apprenant qui décolle** (**29,8**, record **168**) — il
   traverse réellement des rivières, là où `nn` (4,7), `ppo` (7,0) et `clone` (7,8) stagnent.
   La leçon tient : les apprenants par **gradient** profitent de la capacité et de
   l'entraînement long, l'**évolution** (nn) n'en profite pas, l'**imitation** (clone)
   plafonne au niveau du réflexe. Le bond de 13 à 30 ne vient pas d'une brique sophistiquée
   mais d'un entraînement plus long **et** du contrôle de la variance inter-graines (le même
   entraînement varie de ~50 % selon la graine — validations de 25 à 39 sur la campagne, d'où
   10 graines en parallèle et on garde la mieux validée). « Inutile » est devenu « joueur
   solide » — mais loin des 200 exacts.

4. **La hiérarchie reste une hiérarchie d'horizon** : T+1 (heuristic, 51) < rollouts
   aléatoires (mcts, 134) < T+15 exact (search, 200). Plus on anticipe loin et juste, plus on
   va loin ; `guided` confirme le résultat négatif (même qualité que `search` pour ~+1 % de
   nœuds et un coût doublé — il ne reste rien à guider quand la recherche est déjà si efficace).

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

Sous bruit 0,1 (mêmes 50 graines, mêmes modèles qu'en déterministe) :

| Agent | Famille | Score moyen (dét. → bruité) | Médiane | Record | Victoires |
|---|---|---:|---:|---:|---:|
| `heuristic` | robot | 51,4 → 18,3 | 13 | 75 | 0/50 |
| `search` | robot | **200 → 27,8** | 20 | 115 | 0/50 |
| `guided` | hybride | 200 → 23,1 | 17 | 108 | 0/50 |
| `mcts` | robot | 134,3 → 17,3 | 12 | 75 | 0/50 |
| `dqn` | IA | 29,8 → **14,4** | 12 | 44 | 0/50 |
| `clone` | IA | 7,8 → 8,0 | 6 | 27 | 0/50 |
| `ppo` | IA | 7,0 → 6,3 | 5 | 18 | 0/50 |
| `nn` | IA | 4,7 → 4,8 | 4 | 10 | 0/50 |

Trois enseignements :

1. **L'oracle brisé fait s'effondrer les planificateurs.** `search` perd 86 % de son score
   (200 → 28), `guided` de même (→ 23) ; leurs plans à 15 coups sont démentis par les
   turbulences. Le MCTS ne prend pas l'avantage (17,3 < 27,8) : sous incertitude, calculer
   juste reste légèrement devant l'échantillonnage, mais tout le monde souffre.

2. **Sous bruit, l'écart planificateurs/IA se referme presque entièrement.** En déterministe,
   le meilleur robot (200) écrase la meilleure IA (dqn 30) d'un facteur ~7. Sous bruit, ce
   facteur tombe à **moins de 2** : `search` 28, `dqn` **14,4** devant le `mcts` (17,3 — le DQN
   n'est plus qu'à quelques longueurs) et `heuristic` (18,3). Autrement dit, **quand le futur
   n'est plus calculable, le DQN devient un vrai concurrent des planificateurs** — c'est le
   régime où l'apprentissage a le plus de valeur relative.

3. **Le DQN encaisse mieux le changement de régime qu'avant**, mais il a désormais plus à
   perdre : entraîné en monde propre, il passe de 29,8 à 14,4 (les rivières qu'il franchissait
   par timing exact deviennent des pièges — la part de « noyade » explose). Les IA plus faibles
   restent, elles, imperturbables (clone 7,8 → 8,0 ; nn 4,7 → 4,8), parce que leurs réflexes
   n'ont jamais reposé sur la précision du futur. (Sur un monde précédent on avait aussi testé
   un entraînement *dans* le bruit : il n'aidait pas — le bruit dégrade le signal d'apprentissage
   plus qu'il ne forge une robustesse.)

# Reproduire les expériences

```bash
uv run python main.py --mode train --generations 400 --curriculum --workers 4   # nn (GA)
uv run python main.py --mode train-dqn --workers 10    # dqn : 10 graines en parallèle,
                                                       # garde la mieux validée (~5 min)
uv run python main.py --mode train-ppo --workers 6     # ppo : 6 graines (~15 min)
uv run python main.py --mode train-clone --samples 120000     # clone + guided
uv run python main.py --mode bench --episodes 25              # comparatif
uv run python main.py --mode bench --episodes 25 --noise 0.1  # match retour
# entraîner une IA directement dans le bruit (résultat : n'aide pas)
uv run python main.py --mode train-dqn --noise 0.1 --model models/dqn_noisy.npz
uv run python main.py --mode duel --ai search --ai2 dqn --seed 3   # duel visuel
```

Infrastructure : capteurs en accès tabulaires O(1) (tables précalculées par ligne sur
son cycle exact, équivalence bit à bit vérifiée avec l'implémentation de référence) ;
évaluation du GA parallélisable (`--workers`, résultats strictement identiques au
séquentiel) ; entraînements DQN/PPO **multi-graines** (`--workers N` : N entraînements
indépendants, sélection sur validation — l'issue d'un entraînement RL varie du simple
au double selon la graine, voir ANALYSE_IA.md) ; le bruit du monde (`--noise`)
s'applique aussi à l'entraînement. Les threads BLAS sont épinglés à 1 (matrices
minuscules : le multi-threading coûtait plus qu'il ne rapportait).

# Comment on évalue les agents

Les scores ci-dessus viennent du **mode bench** : chaque agent joue les **mêmes N graines**
(mondes identiques pour tous, pour que l'écart vienne de la décision et non du tirage), et on
rapporte la moyenne, la médiane, le record et le taux de victoire de son score final `Y`. Par
défaut on prend 25 graines ; les tableaux ci-dessus sont sur **50 graines**
(`--mode bench --episodes 50`). Le mode `play` sert à *regarder* une partie (graine
aléatoire à chaque lancement), pas à évaluer : une seule partie ne dit rien de la moyenne.

**La métrique qui compte est le score MOYEN** — jusqu'où l'agent va en moyenne. La « victoire »
(atteindre Y = 200) n'est qu'un **plafond de simulation** : un indicateur binaire que la
solution fonctionne, pas une fin en soi. Un agent qui fait 51 de moyenne sans jamais « gagner »
est plus intéressant qu'un détail de taux de victoire.

Pourquoi les IA (`dqn`, `ppo`, `nn`, `clone`) plafonnent-elles si bas ? L'analyse détaillée et
sourcée (environnement mortel, signal rare, politique sans mémoire, exploration faible) et le
plan d'amélioration priorisé sont dans **[ANALYSE_IA.md](ANALYSE_IA.md)**.

# Pistes restantes (non implémentées)

*(La mémoire, le shaping, le curriculum, les retours multi-pas, le Double DQN et
l'entraînement long ont été implémentés et mesurés — voir
[ANALYSE_IA.md](ANALYSE_IA.md), Épisode 2. Restent :)*

1. **Lignes de train** — des rails où un train traverse périodiquement et **bloque toute la
   ligne** deux ticks, avec un signal d'alerte quelques coups à l'avance (la phase des
   capteurs le donnerait déjà). Thématiquement fidèle mais plus lourd à rendre et à équilibrer.
2. **Exploration dirigée** (curiosité, Go-Explore) — la piste littérature restante la plus
   crédible contre le verrou des rivières.
3. **Planification robuste au bruit** — expectimax borné ou marges de sécurité, pour rendre
   aux robots une avance nette en monde stochastique plutôt que le coude-à-coude actuel.
4. **Hybride AlphaZero complet** — une fonction de valeur apprise qui guide vraiment la
   recherche (pas seulement l'ordre des coups).
5. **Tournoi complet** — matrice duel de tous les agents, classement Elo.
