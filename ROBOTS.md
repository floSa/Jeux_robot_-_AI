# Les robots — comment ils réfléchissent

Vulgarisation des stratégies de chaque robot, de leurs coûts de calcul et des pistes
d'amélioration. Pour l'architecture technique, voir [AUDIT.md](AUDIT.md) ; pour les
formules exactes, le [README](README.md).

## Les règles du jeu du point de vue du robot

À chaque tick, le robot doit choisir **exactement un coup parmi cinq** :

| Coup | Delta (dx, dy) |
|---|---|
| AVANCER | (0, +1) |
| RECULER | (0, −1) |
| GAUCHE | (−1, 0) |
| DROITE | (+1, 0) |
| ATTENDRE | (0, 0) |

Le monde est entièrement **déterministe et prévisible** : la position de n'importe quelle
voiture ou tronc à n'importe quel tick futur se calcule instantanément (formule modulo).
Toute l'intelligence consiste à exploiter — ou non — cette prévisibilité.

---

## Robot 1 — le glouton (`heuristic`)

Le robot le plus simple possible : il ne regarde que **le coup suivant** (T+1).

> Si on peut avancer, on avance. Sinon, on choisit entre attendre, aller à gauche ou
> aller à droite : parmi ces options sans danger, on prend celle qui **rapproche le plus
> du centre du terrain** (le centre est en général moins dangereux que les bords ; à
> distance égale, on attend). Et si vraiment tout le reste est mortel, on recule.

Un coup « impossible » (arbre devant, bord du terrain) est écarté d'office : le jouer
gaspillerait un tick sans bouger.

**Pourquoi la préférence pour le centre est-elle importante ?** Une première version
plaçait ATTENDRE strictement au-dessus de gauche/droite. Résultat mesuré sur 100 parties :
le robot finissait bloqué derrière un arbre dans 64 % des cas (« avancer est impossible,
donc j'attends »… pour l'éternité). En fusionnant attendre/gauche/droite avec la
préférence centre, le robot contourne l'arbre naturellement : son score moyen a doublé
(32 → ~68) et il gagne ses premières parties.

**Sa limite indépassable** : avec un seul coup d'avance, il ne peut pas *planifier* la
traversée d'une rivière. Il saute sur un tronc si l'occasion se présente, mais n'anticipe
jamais l'alignement suivant — la plupart de ses parties finissent en mort par stagnation
(120 ticks sans progresser).

---

## Robot 2 — le prévoyant (`search`)

Le glouton regarde les 5 coups possibles et prend le meilleur. Ces 5 coups, on peut les
voir comme des **chemins de longueur 1**. Rien n'empêche de considérer des chemins plus
longs : la liste de tous les chemins de longueur 2, ou de longueur N. Pour chacun, on
simule tick par tick ce qui se passe (les voitures et troncs futurs sont connus !) et on
vérifie qu'on ne meurt nulle part. Parmi les chemins où l'on survit, on choisit **celui
qui avance le plus** ; à égalité, celui qui finit le plus près du centre. On joue alors
le **premier coup** du meilleur chemin, puis on réévalue tout au tick suivant, etc.

### Le problème : l'explosion combinatoire

À chaque tick il y a 5 choix, donc 5^N chemins de longueur N. Ça devient vite ingérable :

| N coups d'avance | chemins possibles (5^N) | cases distinctes à visiter (avec l'amélioration) |
|---:|---:|---:|
| 1 | 5 | 6 |
| 2 | 25 | 19 |
| 3 | 125 | 44 |
| 5 | 3 125 | 146 |
| 7 | 78 125 | 344 |
| 10 | 9 765 625 | 891 |
| 15 | **30 517 578 125** (~30 milliards) | **2 736** |

### L'amélioration : ignorer les chemins redondants

Considérons deux chemins A et B qui commencent différemment mais **arrivent sur la même
case au même moment**. Si on a déjà simulé A et qu'il est valide (on atteint cette case
sans mourir), a-t-on besoin de simuler B ? Non : si B est mortel, on ne le prendra pas ;
et si B est sans danger, il arrive au même endroit au même tick que A — son avenir est
**exactement le même**. Dans tous les cas, B est inutile.

Généralisation : dès qu'on a trouvé *une* façon valide d'atteindre une case à un instant
donné, **tous les autres chemins qui repassent par cette case-instant peuvent être
ignorés**. Le robot garde donc en mémoire l'ensemble des états `(x, y, profondeur)` déjà
atteints, et coupe immédiatement toute branche qui y retombe. Comme presque tous les
chemins repassent par des cases déjà explorées, l'écrasante majorité est éliminée : à
T+15, il ne reste que ~2 700 états à visiter au lieu de 30 milliards de chemins — et en
pratique **encore moins** (~400 en moyenne en partie réelle), car les cases mortelles
élaguent leurs sous-arbres entiers.

Mathématiquement : le nombre d'états croît en **polynôme cubique** (≈ ⅔·N³) au lieu d'une
**exponentielle** (5^N). C'est toute la différence entre 30 milliards et 3 000.

Deux raffinements complètent le robot :

- **File de priorité** : les états sont explorés par « potentiel » décroissant
  f = y + (coups restants), le meilleur Y encore atteignable depuis cet état. Le premier
  chemin complet extrait est alors *garanti optimal* — on peut s'arrêter là.
- **Repli** : si aucun chemin ne survit 15 ticks, on joue le premier coup de la branche
  qui survit le plus longtemps.

Coût mesuré : **~0,4 ms par décision** (budget : 20 ms).

---

## Robot 2+ — l'horizon adaptatif (`search+`)

Que se passe-t-il si le meilleur chemin à T+15 fait du surplace (Y final ≤ Y actuel) ?
C'est le symptôme d'une rivière dont l'alignement des troncs dépasse l'horizon. `search+`
relance alors la recherche avec un horizon élargi — T+30, T+45… jusqu'à T+90 — sous une
garde stricte : jamais plus de 70 % du budget de 20 ms. Une passe interrompue par le
chrono n'est retenue que si elle a trouvé mieux. Pire décision mesurée : 5,2 ms.

---

## Robot 3 — le cerveau évolué (`nn`)

Aucune règle codée à la main : un petit réseau de neurones (32 capteurs → 16 neurones →
5 sorties) lit son environnement proche (distances aux obstacles sur 5 lignes, types de
lignes, vitesses) et « vote » pour un coup. Ses 613 paramètres sont réglés par **sélection
darwinienne** : 100 individus jouent, les meilleurs se reproduisent (croisement), mutent
(bruit gaussien), et on recommence. Fitness = Y maximal atteint.

C'est l'approche la plus fascinante et la moins efficace ici : le signal « Y max » est
pauvre, et le réseau ne voit pas la *phase* des troncs (quand le trou passera). Il
plafonne à ~10 lignes là où la recherche gagne systématiquement — c'est précisément
l'objet du comparatif : **quand le monde est parfaitement prévisible, calculer bat
apprendre**.

---

## L'enquête : quand le coupable était le monde

Longtemps, `search` perdait ~20 % de ses parties par stagnation devant certaines
rivières. Un BFS exhaustif **sans limite d'horizon** depuis toutes les positions réelles
du robot bloqué a prouvé que le franchissement était *impossible* : deux rivières
adjacentes portaient chacune un unique tronc, de même direction et même vitesse — leur
écart relatif restait constant pour l'éternité, le saut n'existait jamais.

Le générateur garantit désormais que deux rivières consécutives ont des **directions
opposées** et un espacement dense : le mouvement relatif force un alignement périodique
(≤ ~29 ticks). Résultat : `search` est passé de ~78 à **100 victoires sur 100**.

Moralité de chercheur : avant d'accuser l'algorithme, prouver que le problème a une
solution.

---

## Résultats (bench officiel, 25 graines identiques pour tous)

Voir le tableau ci-dessous, reproductible via
`uv run python main.py --mode bench --episodes 25` :

| Robot | Score moyen | Médiane | Victoires | ms/décision (moy) | ms max |
|---|---:|---:|---:|---:|---:|
| `heuristic` | 63,6 | 53 | 0/25 | 0,003 | 0,54 |
| `search` | 200 | 200 | **25/25** | 0,29 | 2,21 |
| `search+` | 200 | 200 | **25/25** | 0,30 | 5,19 |
| `nn` (100 générations) | 4,0 | 3 | 0/25 | 0,009 | 0,14 |

(Sur les graines d'entraînement 0-99, l'heuristique décroche en plus ~7 % de victoires.)

---

## Pistes d'IA à tester (proposées)

Par ordre d'intérêt décroissant à mes yeux :

1. **Q-learning / DQN** — apprendre une *valeur* d'état plutôt qu'une politique directe ;
   comparaison passionnante avec la neuroévolution (même capteurs, autre signal
   d'apprentissage : récompense par ligne franchie au lieu du seul Y final).
2. **Fitness shaping + softmax** (améliorer le GA existant) — bonus de survie
   (+0,01/tick), pénalité d'immobilisme, choix stochastique à température décroissante
   pour sortir des boucles comportementales. Le plus petit effort / plus grand gain.
3. **Curriculum** — entraîner le réseau d'abord sur des mondes sans rivière (Phase 1),
   puis mixtes : la génération par graine le permet sans toucher au moteur.
4. **Capteurs de phase** — donner au réseau le *délai avant le prochain alignement* des
   troncs plutôt que la seule distance instantanée : c'est exactement l'information que
   `search` exploite et que `nn` ne voit pas.
5. **MCTS (Monte-Carlo Tree Search)** — surdimensionné pour un monde déterministe, mais
   deviendrait pertinent si on rendait le jeu stochastique (pannes de troncs, voitures à
   vitesse variable).
6. **Hybride réseau + recherche** — un réseau qui ordonne les coups prometteurs pour
   guider la file de priorité de `search` : le patron AlphaZero, en miniature.

`search+` (n° 0 de cette liste) est déjà implémenté et sert de filet de sécurité.
