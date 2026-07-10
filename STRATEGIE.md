# Stratégie du projet — journal de recherche et guide de reprise

Ce document est le **point d'entrée pour reprendre le projet** : la méthode de
travail, toutes les hypothèses testées avec leurs résultats (y compris les
échecs — ils ont autant de valeur), l'état actuel, et la feuille de route
argumentée vers l'objectif « 200 à tous les coups » pour une IA apprise.

Compléments : [ANALYSE_IA.md](ANALYSE_IA.md) (l'enquête détaillée et sourcée,
en 3 épisodes), [ROBOTS.md](ROBOTS.md) (chaque agent vulgarisé + tableaux de
résultats), [AUDIT.md](AUDIT.md) (architecture du code).

---

## 1. L'objectif et la métrique

Faire jouer des agents à un clone de Crossy Road (grille 19 colonnes, hauteur
infinie, monde **déterministe et périodique**, victoire à Y = 200), et comparer
trois familles : **robots** (algorithmes écrits à la main), **IA** (paramètres
appris), **hybrides**.

**La métrique qui compte est le score MOYEN** sur 50 graines de bench jamais
vues à l'entraînement (graines 10000-10049) — jusqu'où l'agent va en moyenne.
La « victoire » (Y = 200) n'est qu'un plafond de simulation. Protocole :
`--mode bench --episodes 50`, mêmes graines pour tous les agents.

Contrainte absolue : **décision < 20 ms par tick**, simulation découplée de
l'affichage. Règle du jeu de la recherche : **pas de triche** — une IA reste
une politique apprise ; les capteurs peuvent encoder toute fonction de l'état
courant (le monde est déterministe), mais la *décision* ne fait pas de
recherche.

## 2. La méthode de travail (ce qui a fait avancer le projet)

1. **Diagnostiquer avant d'optimiser** : les *causes de mort* (collision,
   noyade, sortie d'écran, stagnation) sont l'outil de diagnostic n°1. C'est
   le comptage « 66 % de stagnations » qui a orienté vers la représentation.
2. **Une hypothèse = une expérience contrôlée** : budget court (4000 épisodes),
   2 graines minimum, ablation *soustractive* (tout activé moins une brique)
   ET *additive* (base simple plus une brique), évaluation sur graines de
   bench jamais vues.
3. **Jamais une seule graine** : l'issue d'un entraînement RL varie de ±50 %
   selon la graine (Henderson et al. 2017). Toute comparaison mono-graine est
   du bruit. En production : N entraînements en parallèle, on garde le mieux
   validé (`--workers N`).
4. **La littérature propose, la mesure dispose** : cinq briques « état de
   l'art » se sont révélées neutres à nuisibles à notre budget. L'inverse
   existe aussi : le shaping d'alignement, neutre en épisode 2, devient utile
   en épisode 3 — un remède ne vaut que dans son contexte.
5. **Vérifier par invariants** tout nouveau capteur (ex. « la case sous un
   joueur vivant est praticable », cohérence avec les tables du moteur).

## 3. Historique des hypothèses testées

### Épisode 1 — capacité et entraînement (13 de moyenne)

| Hypothèse | Test | Résultat | Décision |
|---|---|---|---|
| Les IA sont sous-entraînées | réseaux élargis + entraînements longs | dqn ~5 → 13,1 | ✅ adopté |
| S'entraîner dans le bruit rend robuste au bruit | train-dqn --noise 0.1 | moins bon que le modèle propre | ❌ rejeté |

### Épisode 2 — les remèdes de la littérature (13 → 30)

Cinq briques implémentées (toutes restent dans la config) : récompense
potentielle « plateforme », mémoire par pile d'observations (FRAME_STACK),
retours multi-pas (DQN_NSTEP), curriculum rivières, Double DQN.

| Hypothèse | Test (4000 ép., bench 25 graines) | Résultat | Décision |
|---|---|---|---|
| Mémoire ×4 aide (POMDP) | ablation add./soustr. | 13,0 → 7,9 | ❌ nuit à ce budget |
| Retours 3 pas aident | idem | 13,0 → 7,7 | ❌ nuit (biais hors-politique) |
| Curriculum rivières aide | idem | 13,0 → 9,3 | ❌ nuit |
| Shaping « plateforme » aide | idem | 13,0 → 12,3 | ➖ neutre |
| Double DQN + réseau 128 aident | idem | 13,9 → 9,6 | ❌ capacité nuit, double neutre |
| **La graine domine tout** | même recette, graines multiples | 6,5 à 13,9 (×2) | ✅ **multi-graines `--workers`** |
| Entraîner plus longtemps | 4000 → 10 000 ép. × 10 graines | validations 25-38,8 (avant : 21) | ✅ adopté |

Bench officiel : **dqn 29,8** (record 168). Leçon : à ~10⁶ pas
d'environnement, les remèdes calibrés pour 10⁷-10⁸ pas ne paient pas ;
le levier réel est le contrôle de la variance + la durée.

### Épisode 3 — la représentation (30 → 91)

Diagnostic : 33/50 morts par **stagnation** → deux angles morts : cécité
latérale (les distances agrégées ne disent pas OÙ est le passage) et état non
markovien (le compteur de stagnation était invisible).

Nouveau capteur `sense_grid` (482 entrées) : praticabilité des 19 colonnes ×
6 lignes, **projetée aux instants t..t+3** (exact, monde périodique), + type
et vitesse par ligne + compteur de stagnation.

| Hypothèse | Test (4000 ép., 2 graines) | Résultat (bench) | Décision |
|---|---|---|---|
| Le timer de stagnation seul suffit | 42 capteurs + timer | 15,9 → 13,4 | ❌ l'info seule ne sert pas |
| La vision spatiale seule aide | grille à t seulement | 15,9 → 22,4 | ✅ mais insuffisant |
| **La projection temporelle est la clé** | grille t..t+3 | 15,9 → **78,0** | ✅✅ adopté |
| Le shaping d'« alignement » aide une fois la cible visible | grille + potentiel d'alignement | 78,0 → **83,4** + stabilité | ✅ adopté (RL_SHAPING="alignement") |
| Plus de capacité aide avec la grande entrée | grille, 128 neurones | 78,0 → 66,6 | ❌ toujours non |
| L'imitation profite encore plus de la vision | clone grille (BC pur) | 7,8 → 28,8 (précision 73 → 87 %) | ✅ adopté |
| **DAgger** corrige le décalage de distribution | + 2 tours (l'élève conduit, l'expert étiquette) | 28,8 → **56,1** | ✅ adopté (train-clone) |

Campagne finale (10 graines × 12 000 ép., 15 min) : validations **131-153,5**.

### Résultats officiels actuels (bench 50 graines, déterministe)

| Agent | Score moyen | Médiane | Record | Victoires | Note |
|---|---:|---:|---:|---:|---|
| `search` / `guided` | 200 | 200 | 200 | 50/50 | planification exacte |
| `mcts` | 134 | 200 | 200 | 25/50 | |
| **`dqn`** | **91,3** | 78 | 200 | **8/50** | stagnations 33 → 3/50 |
| `heuristic` | 51,4 | 38 | 200 | 1/50 | |
| **`clone`** | **51,4** | 40 | 200 | 1/50 | égale l'heuristique en réflexe pur |
| `ppo` | 7,0 | 5 | 33 | 0/50 | resté aux capteurs agrégés |
| `nn` | 4,7 | 4 | 10 | 0/50 | idem |

Sous bruit 0,1 : tout le monde entre 17 et 28 — plus un agent exploite le
déterminisme, plus le bruit lui coûte.

## 4. Feuille de route vers « 200 à tous les coups »

Le constat de départ : le dqn meurt désormais **en jouant** (collisions 19,
noyades 11, sorties 9), plus en hésitant. Or gagner exige ~2000 décisions
consécutives sans une seule erreur fatale : même 99,9 % de décisions justes
→ ~87 % de parties perdues. Deux chemins, du plus sûr au plus pur :

### Piste A — le filet de sécurité (quasi-garantie, famille « hybride »)

Garder la politique apprise, mais **vérifier chaque coup avant de le jouer** :
simuler l'action argmax 1 à 3 ticks avec `engine.next_state` (pur, ~µs) ; si
elle est fatale ou mène à une impasse, prendre la meilleure action sûre
suivante. C'est le « shield » du Safe RL — le réseau décide, le filet ne fait
qu'interdire le prouvé-mortel. Coût : quelques appels moteur par tick, budget
20 ms intact. Gain attendu : élimine l'essentiel des 39 morts « d'inattention »
→ très proche de 200/200. À assumer dans la taxonomie : `dqn+shield` est un
hybride (comme `guided`), pas une IA pure — c'est LE compromis à discuter.
Variante aboutie : l'hybride AlphaZero complet (réseau de valeur + recherche
minuscule de 5-15 nœuds), déjà en piste depuis l'épisode 1.

### Piste B — pousser le réflexe pur (sans garantie, mais le vrai défi)

Dans l'ordre de rapport gain/effort attendu :

1. **Élargir le cône de vision** (GRID_TIME_PLANES jusqu'à t+7, lignes
   jusqu'à +6) : les erreurs actuelles ressemblent à des choix localement
   corrects mais perdants à 5-10 ticks — exactement ce que le cône actuel
   (t+3, +4 lignes) ne montre pas. Expérience directe avec le harnais
   existant (~30 min). Attention mesurée : entrée plus grande = apprentissage
   plus lent, à compenser par plus d'épisodes.
2. **Architecture convolutionnelle 1D sur les colonnes** : un MLP dense doit
   réapprendre chaque motif à chaque position ; une convolution partage les
   poids entre colonnes (équivariance par translation — la situation « trou à
   gauche » est la même partout). C'est LE remplacement de capacité qui a des
   chances de payer là où « plus de neurones » a toujours échoué. À écrire en
   numpy (forward + backward, ~100 lignes, vérification par gradient numérique
   comme pour le MLP).
3. **DQfD / initialisation par le clone** : pré-remplir le replay avec des
   transitions de `search` (ou initialiser le réseau Q depuis la politique du
   clone-DAgger) — le RL part alors d'un comportement déjà compétent au lieu
   de redécouvrir la marche. Littérature : Hester et al. 2018.
4. **Masquage des actions fatales PENDANT l'entraînement** : ne jamais
   laisser epsilon jouer un coup prouvé-mortel → toute l'exploration se
   concentre sur les vrais choix. (Si on garde le masque au jeu, on retombe
   dans la piste A — à trancher.)
5. **Encore plus long + replay priorisé sur les quasi-morts** : le dqn
   progressait toujours à 12 000 épisodes ; viser 50 000+ (l'entraînement est
   parallèle et ne coûte que quelques dizaines de minutes).

Pronostic honnête : la piste B peut viser 120-160 de moyenne et des victoires
majoritaires ; le **200/200 strict** passera très probablement par la piste A
— et c'est un beau résultat en soi : c'est exactement la conclusion AlphaZero
(le réflexe appris fait 95 % du travail, la vérification fait la perfection).

### Chantiers annexes (indépendants)

- **PPO en vision grille** : ppo est resté aux 42 capteurs (7,0) ; lui donner
  `sense_grid` + alignement est un test à ~1 h qui dirait si le policy
  gradient profite autant que la valeur.
- **Lignes de train** (gameplay), **tournoi Elo**, **planification robuste au
  bruit** : voir « Pistes restantes » de ROBOTS.md.

## 5. Reprendre le projet en 5 minutes

```bash
uv sync                                                  # environnement
uv run python main.py --mode play --ai dqn               # voir jouer le champion
uv run python main.py --mode bench --episodes 50         # reproduire les tableaux
uv run python main.py --mode train-dqn --workers 10      # réentraîner dqn (~15 min)
uv run python main.py --mode train-clone                 # réentraîner clone (BC+DAgger)
```

Fichiers clés : `config.py` (tous les choix mesurés sont les défauts ; les
briques rejetées restent activables : FRAME_STACK, DQN_NSTEP, DQN_DOUBLE,
DQN_CURRICULUM_PROB, RL_SHAPING, DQN_SENSOR), `sensors.py` (les deux
générations de capteurs + dispatch automatique par taille d'entrée du modèle),
`shaping.py` (potentiels), `ai_qlearning.py` (DQN + multi-graines),
`ai_hybrid.py` (imitation + DAgger + recherche guidée).

Détail d'exécution : les threads BLAS sont épinglés à 1 dans `main.py`
(matrices minuscules : le multi-threading coûtait jusqu'à 10× ; ne pas retirer).
