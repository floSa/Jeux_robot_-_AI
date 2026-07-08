# Pourquoi les IA sont faibles — analyse et pistes d'amélioration

Ce document explique, mesures et littérature à l'appui, **pourquoi les agents appris**
(`dqn`, `ppo`, `nn`, `clone`) plafonnent très bas à ce jeu, et **comment les améliorer**.

> Note de métrique : ici la « victoire » (atteindre Y = 200) n'est qu'un **plafond de
> simulation** — un indicateur que la solution fonctionne. La vraie métrique est le
> **score moyen** : *jusqu'où l'agent va en moyenne*. C'est sous cet angle qu'il faut lire
> ce qui suit.

## Ce qu'on mesure (bench, 50 graines, monde complet)

| Agent | Score moyen | Record | Ce qu'il fait |
|---|---:|---:|---|
| `search` / `guided` | **200** | 200 | franchit tout (planification exacte) |
| `mcts` | 134 | 200 | planification approximative |
| `heuristic` | 51 | 200 | glouton T+1, cale sur ce qui exige un plan |
| `dqn` | **13** | 75 | seule IA qui traverse vraiment des rivières |
| `clone` | 8 | 32 | imite `search`, plafonne au réflexe |
| `ppo` | 6 | 23 | policy gradient |
| `nn` | 5 | 10 | neuroévolution |

Fait central : **quatre méthodes d'apprentissage très différentes plafonnent toutes très
bas**, alors que la planification atteint 200. Ce n'est pas un bug de réglage : c'est que
ce jeu réunit, d'un coup, la plupart des difficultés connues de l'apprentissage par
renforcement (RL). On les passe en revue.

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

1. **Récompense mieux façonnée (dense, potentielle)** — au lieu de « +1 par ligne », ajouter
   un signal *potentiel* : bonus pour se rapprocher de la prochaine plateforme sûre, malus
   pour longer une case mortelle, bonus d'être *sur* un tronc/nénuphar. Le passage
   « creux → dense » est l'un des remèdes les mieux établis [8]. *(vise la cause 2)*
2. **Mémoire dans la politique** — réseau récurrent (LSTM/GRU) ou pile des N dernières
   observations, pour capter le rythme des plateformes [6][7]. *(cause 3)*
3. **Exploration dirigée** — récompense intrinsèque de nouveauté/curiosité, ou boucle façon
   *Go-Explore* (mémoriser les états d'où l'on a progressé, y retourner, explorer de là) [2].
   *(cause 4)*
4. **Curriculum ciblé rivières** — entraîner spécifiquement à traverser des rivières de plus
   en plus larges, plutôt que « routes d'abord » ; les curricula automatiques sont efficaces
   sur les récompenses creuses [8][9]. *(cause 2)*
5. **Pénaliser explicitement l'irréversible / replay priorisé** — donner plus de poids aux
   transitions rares (traversées réussies, morts évitables) et rendre les états mortels
   explicitement « pires », pour stabiliser le DQN [1]. *(cause 1)*
6. **Aller au bout de l'hybride (AlphaZero-like)** — apprendre une *fonction de valeur* qui
   guide vraiment la recherche (pas seulement l'ordre des coups), pour combiner la robustesse
   de l'apprentissage et la précision de la planification. *(cause 5)*
7. **Simplement entraîner beaucoup plus longtemps** — le DQN progressait encore à 4000
   épisodes (record 75) ; c'est le levier le moins malin mais réel.

Le meilleur rapport effort/gain attendu : **(1) récompense potentielle + (2) mémoire**, qui
attaquent les deux causes les plus lourdes (signal rare et absence de mémoire).

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
