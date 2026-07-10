# Pourquoi les IA sont faibles — analyse, pistes, et verdict expérimental

Ce document explique, mesures et littérature à l'appui, **pourquoi les agents appris**
(`dqn`, `ppo`, `nn`, `clone`) plafonnent très bas à ce jeu, et **comment les améliorer**.
La seconde partie (« Épisode 2 ») rapporte ce qui s'est passé quand on a réellement
implémenté les remèdes de la littérature — avec un verdict d'ablation qui surprend —
et l'Épisode 3 le déblocage par la représentation. La synthèse « reprendre le projet »
et la feuille de route sont dans [STRATEGIE.md](STRATEGIE.md).

> Note de métrique : ici la « victoire » (atteindre Y = 200) n'est qu'un **plafond de
> simulation** — un indicateur que la solution fonctionne. La vraie métrique est le
> **score moyen** : *jusqu'où l'agent va en moyenne*. C'est sous cet angle qu'il faut lire
> ce qui suit.

## Ce qu'on mesure (bench, 50 graines, monde complet)

| Agent | Score moyen | Record | Ce qu'il fait |
|---|---:|---:|---|
| `search` / `guided` | **200** | 200 | franchit tout (planification exacte) |
| `mcts` | 134 | 200 | planification approximative |
| `dqn` | **91** | 200 | vision grille : traverse tout, gagne 8/50 |
| `heuristic` | 51 | 200 | glouton T+1, cale sur ce qui exige un plan |
| `clone` | **51** | 200 | imitation + DAgger en vision grille |
| `ppo` | 7 | 33 | policy gradient |
| `nn` | 5 | 10 | neuroévolution |

*(Chiffres `dqn`/`clone` = après l'Épisode 3 (vision grille) ; `ppo` = après l'Épisode 2.
Ceux d'origine, qui ont motivé l'analyse ci-dessous : dqn 13, clone 8, ppo 6.)*

Fait central au départ : **quatre méthodes d'apprentissage très différentes plafonnaient
toutes très bas** (dqn 13, clone 8, ppo 6, nn 5), alors que la planification atteint 200.
Ce n'était pas un bug de réglage : ce jeu réunit, d'un coup, la plupart des difficultés
connues de l'apprentissage par renforcement (RL). On les passe en revue — puis les
Épisodes 2 et 3 racontent ce que chaque remède a réellement donné, jusqu'au déblocage.

## Pourquoi — cinq raisons, chacune documentée

### 1. Un environnement *mortel et irréversible* → apprentissage à haute variance

Ici, une seule mauvaise case = mort immédiate, fin de l'épisode. Il n'y a pas de « rattrapage ».
La littérature montre que les états **irréversibles** déstabilisent fortement les agents à
base de valeur (comme le DQN) : ils provoquent une **surestimation des valeurs** et une
variance élevée, et l'on gagne beaucoup à *pénaliser explicitement* les états irréversibles
pour stabiliser l'apprentissage (réductions de « catastrophes » de >99 % dans des
benchmarks type CliffWalking) [1]. Conséquence concrète chez nous : l'agent apprend surtout
à **éviter le risque** — il reste sur le trottoir, tergiverse devant une rivière, et meurt
de stagnation plutôt que de tenter une traversée.

### 2. Un signal *rare et différé* pour les sous-tâches dures (rivières, nénuphars)

Traverser une rivière ou une ligne de nénuphars demande une **séquence précise de plusieurs
coups justes** ; une seule erreur tue. Or par exploration aléatoire (ε-greedy), l'agent ne
réussit presque **jamais** une traversée complète par hasard → il reçoit très peu d'exemples
positifs pour apprendre *comment* traverser. C'est exactement le régime « récompense creuse /
exploration difficile » où « des séquences précises de nombreuses actions doivent être
exécutées entre deux récompenses », et où les méthodes d'essai-erreur classiques comme le DQN
échouent [2][3]. Le cas d'école est *Montezuma's Revenge*, où le DQN marque… **0**, précisément
parce qu'il faut enchaîner une longue série d'actions avant la moindre récompense [4][5]. Notre
jeu est plus doux (on gagne un peu de récompense à chaque ligne), mais les **rivières** en
reproduisent le cœur : localement, la traversée est un mini-Montezuma.

### 3. Une politique *réactive, sans mémoire* face à un monde partiellement observé

Nos IA sont des réseaux **feedforward** : elles font `observation → action`, sans état interne.
Or timing d'un saut sur un tronc ou un nénuphar dépend de *quand* la plateforme arrivera —
une information temporelle qu'une politique sans mémoire capte mal. La théorie est nette : une
politique réactive qui mappe directement l'observation à l'action est **sous-optimale pour tout
sauf les problèmes les plus simples** en observabilité partielle ; l'optimum dépend en général
de **tout l'historique**, ce que seules des architectures à **mémoire** (réseaux récurrents,
LSTM) approchent, avec des gains « constants » sur les bases feedforward [6][7]. Nos capteurs de
phase (délai avant qu'une case soit occupée/libérée) atténuent le problème mais ne le suppriment
pas : le réseau n'a pas d'état pour « compter » l'attente.

### 4. Une exploration *pauvre* (ε-greedy)

L'exploration par bruit aléatoire est « **inefficace en échantillons** et peine dans les
environnements à récompense différée ou creuse » [3]. Sans mécanisme d'exploration dirigée
(curiosité, nouveauté, comptage d'états, ou approche *Go-Explore* qui mémorise et re-visite
les états prometteurs [2]), l'agent ne découvre quasiment jamais les rares trajectoires qui
franchissent une rivière — donc ne peut pas les renforcer.

### 5. L'apprentissage réactif *jette* la structure que la recherche exploite

Le monde est **périodique et parfaitement prévisible**. `search` en profite pour tout calculer.
Une politique réactive, elle, doit **ré-apprendre** ce rythme à partir d'exemples bruités, sans
jamais le calculer explicitement. C'est le pari inverse de la planification — et c'est pour ça
que les systèmes les plus forts (AlphaZero, MuZero) **combinent** recherche *et* apprentissage
plutôt que d'opposer les deux. Notre `guided` est une première marche dans cette direction.

## En résumé

Le plafond des IA n'est ni un bug ni une fatalité : c'est la **superposition** d'un
environnement mortel (variance), d'un signal rare pour les sous-tâches dures (crédit à long
terme), d'une politique sans mémoire (observabilité partielle) et d'une exploration faible.
Chacune de ces difficultés est un domaine de recherche actif — et chacune suggère un levier.

## Comment les améliorer — plan priorisé (chaque item = une des cinq causes)

*(Statuts mis à jour après mise en œuvre — le détail est dans l'Épisode 2 ci-dessous.)*

1. **Récompense mieux façonnée (dense, potentielle)** — au lieu de « +1 par ligne », ajouter
   un signal *potentiel* : bonus pour se rapprocher de la prochaine plateforme sûre, malus
   pour longer une case mortelle, bonus d'être *sur* un tronc/nénuphar. Le passage
   « creux → dense » est l'un des remèdes les mieux établis [8]. *(vise la cause 2)* —
   **✅ implémenté (`RL_SHAPING`), mesuré neutre à notre budget.**
2. **Mémoire dans la politique** — réseau récurrent (LSTM/GRU) ou pile des N dernières
   observations, pour capter le rythme des plateformes [6][7]. *(cause 3)* —
   **✅ implémenté (`FRAME_STACK`), mesuré NÉGATIF à notre budget.**
3. **Exploration dirigée** — récompense intrinsèque de nouveauté/curiosité, ou boucle façon
   *Go-Explore* (mémoriser les états d'où l'on a progressé, y retourner, explorer de là) [2].
   *(cause 4)* — **⏳ non implémenté.**
4. **Curriculum ciblé rivières** — entraîner spécifiquement à traverser des rivières de plus
   en plus larges, plutôt que « routes d'abord » ; les curricula automatiques sont efficaces
   sur les récompenses creuses [8][9]. *(cause 2)* —
   **✅ implémenté (`DQN_CURRICULUM_PROB`), mesuré négatif à notre budget.**
5. **Pénaliser explicitement l'irréversible / replay priorisé** — donner plus de poids aux
   transitions rares (traversées réussies, morts évitables) et rendre les états mortels
   explicitement « pires », pour stabiliser le DQN [1]. *(cause 1)* —
   **🟡 partiellement : Double DQN implémenté (`DQN_DOUBLE`), mesuré neutre ; replay priorisé non fait.**
6. **Aller au bout de l'hybride (AlphaZero-like)** — apprendre une *fonction de valeur* qui
   guide vraiment la recherche (pas seulement l'ordre des coups), pour combiner la robustesse
   de l'apprentissage et la précision de la planification. *(cause 5)* — **⏳ non implémenté.**
7. **Simplement entraîner beaucoup plus longtemps** — le DQN progressait encore à 4000
   épisodes (record 75) ; c'est le levier le moins malin mais réel. —
   **✅ fait : 10 000 épisodes × 10 graines avec sélection sur validation — c'est CE levier
   (plus le contrôle de la variance inter-graines) qui a réellement payé.**

Le meilleur rapport effort/gain attendu *a priori* : **(1) récompense potentielle +
(2) mémoire**, qui attaquent les deux causes les plus lourdes (signal rare et absence de
mémoire). La suite du document raconte ce que la mesure en a fait.

---

# Épisode 2 — on a implémenté les remèdes : verdict d'ablation

Les pistes 1 à 5 ont été **implémentées** (elles sont dans le code, pilotées par la
config) : récompense potentielle (`RL_SHAPING`, module `shaping.py`, construction de
Ng et al. [10] qui garantit de ne pas changer la politique optimale), mémoire par pile
d'observations (`FRAME_STACK`), retours multi-pas (`DQN_NSTEP`), curriculum rivières
(`DQN_CURRICULUM_PROB`), Double DQN (`DQN_DOUBLE`), plus un réseau doublé (128 neurones).

Le premier entraînement « tout activé » a stagné (validation ~5 là où la recette simple
atteignait ~14). Plutôt que de deviner, on a mené **deux ablations** (4000 épisodes,
score moyen sur 25 graines de bench jamais vues à l'entraînement) :

## Ablation soustractive — « tout v2, moins une brique » (graine 0)

| Variante | Score bench (25 graines) |
|---|---:|
| **v1-exact** (réseau 64, rien d'activé) | **13,9** |
| v1 + réseau 128 + Double DQN | 9,6 |
| tout v2 sauf mémoire | 6,5 |
| tout v2 sauf shaping | 6,2 |
| tout v2 sauf curriculum | 5,6 |
| tout v2 sauf n-pas | 5,0 |
| tout v2 | 4,2 |

## Ablation additive — « la recette simple, plus une brique » (2 graines, moyenne)

| Variante | Score bench (moyenne 2 graines) |
|---|---:|
| **base simple** | **13,0** |
| + shaping potentiel | 12,3 |
| + Double DQN | 12,3 |
| + curriculum rivières | 9,3 |
| + mémoire (pile ×4) | 7,9 |
| + retours 3 pas | 7,7 |

## Lecture — trois leçons qui valent la peine d'être retenues

1. **Les remèdes de la littérature supposent un budget d'échantillons qu'on n'a pas.**
   Frame stacking, n-pas, curriculum sont validés sur des régimes Atari de 10⁷-10⁸ pas
   d'environnement ; notre entraînement complet en fait ~10⁶. À ce budget, multiplier
   l'entrée par 4 (mémoire) ou doubler le réseau **ralentit l'apprentissage plus que
   l'information ajoutée ne l'accélère** ; et les retours multi-pas, calculés sur des
   trajectoires majoritairement exploratoires (epsilon élevé), injectent un biais
   hors-politique connu. Un remède juste *en théorie* peut être nuisible *à petit budget*.

2. **La graine d'entraînement pèse autant que ces hyperparamètres.** La *même* recette,
   relancée sur dix graines dans la campagne finale, donne des validations de **25 à 39**
   (~50 % d'écart) ; et dans les ablations, l'écart entre graines d'une variante donnée est
   du même ordre que l'écart *entre* variantes — autant dire que comparer deux méthodes sur
   une seule graine ne prouve rien. C'est un résultat classique du RL profond (reproductibilité
   fragile, la variance inter-graines domine souvent l'effet mesuré des méthodes [11]), et il
   transforme la conclusion : le levier le plus rentable n'est aucune des briques
   sophistiquées, mais **entraîner N graines en parallèle et garder la mieux validée**.

3. **Le shaping potentiel tient sa promesse théorique — et c'est tout.** Neutre en
   mesure (12,3 vs 13,0, dans le bruit inter-graines) : il ne casse rien (la garantie
   d'invariance [10] fonctionne), mais ne débloque pas les rivières à lui seul.

## Ce qui est livré

- Les cinq briques restent **disponibles dans la config** (`FRAME_STACK`, `RL_SHAPING`,
  `DQN_NSTEP`, `DQN_CURRICULUM_PROB`, `DQN_DOUBLE`), défauts alignés sur la mesure.
- **Entraînement multi-graines** : `--mode train-dqn --workers 10` (idem `train-ppo`)
  entraîne 10 graines en parallèle (1 thread BLAS chacune) et garde la mieux validée
  sur 10 graines fixes.
- Le modèle `dqn` livré vient de cette procédure : 10 graines × 10 000 épisodes, en
  ~3 minutes au total. Validations obtenues : 25,2 / 31,7 / 31,8 / 32,8 / 33,2 / 36,1 /
  36,5 / 36,7 / 38,2 / **38,8** (retenue) — à comparer aux **21** de l'ancienne recette
  (4 000 épisodes, 1 graine). Deux effets cumulés : allonger l'entraînement déplace
  *toute* la distribution (la pire des 10 graines bat l'ancien modèle), et la sélection
  prend la queue haute. Résultats officiels sur graines de bench dans [ROBOTS.md](ROBOTS.md).

---

# Épisode 3 — la vision en grille : la représentation était LE verrou

L'Épisode 2 se concluait sur un constat un peu frustrant : les remèdes
*algorithmiques* de la littérature ne payaient pas. L'hypothèse suivante ne
portait plus sur l'algorithme mais sur **ce que l'agent perçoit** : il faut
qu'il puisse *se projeter sur la grille* — voir où sont les passages, et où
ils seront.

## Le diagnostic qui a tout déclenché

Au bench, le DQN 42-capteurs mourait à **66 % par stagnation** (33/50), pas
écrasé (28 %), pas noyé. Il ne mourait pas d'imprudence : il mourait **bloqué**.
Or ses capteurs agrégés (distance au plus proche obstacle à gauche/droite,
par ligne) ont deux angles morts exactement là :

1. **Cécité latérale** : si le trou dans la ligne d'arbres est à 5 colonnes,
   « distance au plus proche obstacle » ne dit pas de quel côté ni où aller.
2. **État non markovien** : la mort par stagnation (120 ticks sans progrès)
   était *invisible* dans l'observation — une politique argmax bloquée restait
   donc bloquée pour l'éternité, sans même « sentir » l'urgence.

## Le nouveau capteur : praticabilité projetée dans l'espace-temps

`sense_grid` remplace les agrégats par la donnée brute organisée : pour 6
lignes autour du joueur, une carte égocentrique de **praticabilité** des 19
colonnes (herbe sans arbre, route sans véhicule, rivière AVEC tronc/nénuphar),
calculée aux instants **t, t+1, t+2, t+3** — le monde est périodique, la
projection est *exacte* et de la même classe d'information que les capteurs
de phase (tto/ttf) déjà présents. S'y ajoutent le type et la vitesse de chaque
ligne, et le **compteur de stagnation** normalisé (complétude Markov). Au
total 482 entrées au lieu de 42 ; la décision reste un unique passage avant
du réseau (~0,07 ms) — **aucune recherche, aucune triche** : c'est la
représentation en grille standard des travaux RL sur Frogger/Crossy Road
[12][13], enrichie de la projection temporelle que notre monde périodique
permet de rendre exacte.

Chaque brique a été validée par invariants sur 1446 états (la case sous le
joueur vivant est toujours praticable ; cohérence bit à bit avec les tables
occ/tto du moteur).

## Verdict expérimental (4000 épisodes, 2 graines, bench 25 graines)

| Variante | Score bench (moyenne 2 graines) |
|---|---:|
| base 42 capteurs (référence) | 15,9 |
| + compteur de stagnation seul | 13,4 |
| vision grille, praticabilité à t seul | 22,4 |
| vision grille projetée t..t+3 | 78,0 |
| **vision projetée + potentiel d'alignement** | **83,4** |
| vision projetée, 128 neurones | 66,6 |

Quatre leçons :

1. **La représentation était le verrou — pas l'algorithme.** Même DQN, mêmes
   hyperparamètres, même budget : ×5 en changeant uniquement ce que le réseau
   *voit*. À 4000 épisodes seulement, l'agent vision-grille dépasse déjà
   l'`heuristic` (51) et signe les **premières victoires (Y = 200) d'une IA**
   du projet.
2. **La projection temporelle est l'ingrédient décisif** : vision spatiale
   seule 22, spatio-temporelle 78. Voir où sont les trous ne suffit pas ;
   il faut voir où ils *seront*.
3. **Un signal de récompense ne vaut que si l'agent peut voir ce qu'il
   récompense.** Le shaping potentiel, neutre avec les capteurs agrégés
   (Épisode 2), devient utile et STABILISANT (83,4, écart inter-graines
   réduit) une fois la cible visible : le potentiel d'alignement pousse vers
   le passage praticable de la ligne suivante — l'anti-stagnation exact.
   Interaction représentation x récompense.
4. **La capacité double nuit toujours** (66,6) — cohérent avec l'Épisode 2 :
   à petit budget, les paramètres supplémentaires coûtent plus qu'ils ne
   rapportent.

## Campagne finale et résultats officiels

Recette livrée : `DQN_SENSOR="grid"`, `RL_SHAPING="alignement"`, 64 neurones,
et la procédure multi-graines de l'Épisode 2 (10 graines × 12 000 épisodes,
sélection sur validation — 15 minutes de calcul). Les dix graines valident
entre **131 et 153,5** ; la veille, même procédure avec les capteurs agrégés :
25 à 38,8. Bench officiel (50 graines jamais vues à l'entraînement) :

| `dqn` | 42 capteurs, 4 000 ép. | 42 capteurs, multi-graines | **vision grille** |
|---|---:|---:|---:|
| score moyen | 13,1 | 29,8 | **91,3** |
| médiane | 12 | 22 | **78** |
| record | 75 | 168 | **200** |
| victoires | 0/50 | 0/50 | **8/50** |

Le diagnostic initial est **validé par les causes de mort** : les stagnations
passent de 33/50 à **3/50**. L'agent qui voit où est le passage ne reste plus
bloqué devant — il meurt désormais en jouant (collisions, noyades), pas en
hésitant.

## Bonus — l'imitation explose aussi (clone en vision grille + DAgger)

Le supervisé profite encore plus de la représentation que le RL (il est bien
plus efficace en échantillons). Trois paliers mesurés :

| `clone` | bench | précision |
|---|---:|---:|
| imitation pure, 42 capteurs | 7,8 | 73 % |
| imitation pure, vision grille | 28,8 | 87 % |
| + 2 tours **DAgger** [14] | **51,4** (record 200, 1 victoire) | 86 % (équilibrée 65 → 77 %) |

DAgger corrige LE défaut structurel de l'imitation : l'élève n'apprenait que
sur les états que le maître visite, donc jamais à se rattraper quand il
s'égare. En rejouant la politique de l'ÉLÈVE et en faisant étiqueter ses
états par l'expert, le clone égale désormais l'`heuristic` (51) — un pur
réflexe, sans une seule recherche au moment de jouer.

## La leçon de l'Épisode 3

Les cinq causes de l'analyse initiale étaient réelles, mais la hiérarchie
était ailleurs : **avant l'algorithme, la représentation**. Un DQN standard,
sans une seule brique exotique, passe de 13 à 91 quand on lui donne à voir ce
que le problème demande de voir. Reste l'écart avec `search` (200) : le
réflexe voit 4 ticks devant, la recherche 15 — combler cet écart demanderait
d'élargir le cône visible ou d'hybrider (piste AlphaZero, toujours ouverte).

## Références

1. *Learning to Undo: Rollback-Augmented RL with Reversibility Signals* — [arXiv:2510.14503](https://arxiv.org/abs/2510.14503v1)
2. *Go-Explore: a New Approach for Hard-Exploration Problems* — [arXiv:1901.10995](https://arxiv.org/pdf/1901.10995) · [blog Uber](https://www.uber.com/us/en/blog/go-explore/)
3. *Exploration in Deep Reinforcement Learning: A Survey* — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253522000288)
4. *A Survey of Deep Reinforcement Learning in Video Games* — [arXiv:1912.10944](https://arxiv.org/pdf/1912.10944)
5. *Learning Montezuma's Revenge from a Single Demonstration* — [Semantic Scholar](https://www.semanticscholar.org/paper/Learning-Montezuma's-Revenge-from-a-Single-Salimans-Chen/5d4ebbaa1ef8feeac885e2869f45c0276c18834f)
6. *Solving Deep Memory POMDPs with Recurrent Policy Gradients* — [IDSIA](https://people.idsia.ch/~alexander/2007/2/icann2007.pdf)
7. *Influence-aware memory architectures for deep RL in POMDPs* — [Springer](https://link.springer.com/article/10.1007/s00521-022-07691-7)
8. *From Sparse to Dense: Toddler-inspired Reward Transition in Goal-Oriented RL* — [arXiv:2501.17842](https://arxiv.org/pdf/2501.17842)
9. *DISCOVER: Automated Curricula for Sparse-Reward RL* — [arXiv:2505.19850](https://arxiv.org/html/2505.19850v2)
10. Ng, Harada & Russell, *Policy invariance under reward transformations: Theory and application to reward shaping* (ICML 1999) — le shaping potentiel F = γΦ(s′) − Φ(s) ne change pas la politique optimale.
11. Henderson et al., *Deep Reinforcement Learning that Matters* — [arXiv:1709.06560](https://arxiv.org/abs/1709.06560) : la variance entre graines d'entraînement domine souvent l'effet des méthodes comparées.
12. *Comparing Learning Algorithms for Crossy Road* (Stanford CS221) — [poster](https://web.stanford.edu/class/archive/cs/cs221/cs221.1192/2018/restricted/posters/nwrubin/poster.pdf) : le défi central est une représentation d'état « assez riche sans exploser l'espace d'états ».
13. *CrossyRoadPlayer : DDQN et A3C sur Crossy Road* — [GitHub](https://github.com/Introduction-to-Machine-Learning-Team4/CrossyRoadPlayer) : observation en grille (7 lignes × 21 cases) autour du joueur, l'approche standard.
14. Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger) — [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) : faire étiqueter par l'expert les états visités par l'élève corrige le décalage de distribution de l'imitation.
