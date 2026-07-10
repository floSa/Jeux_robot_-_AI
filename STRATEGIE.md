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

## 4. Épisode 4 — le filet de sécurité (piste A implémentée et mesurée)

Question posée : peut-on FAIRE COMPRENDRE à l'agent qu'une case tue, de façon
fiable ? Réponse : on n'a pas besoin qu'il l'apprenne — on a un simulateur
**exact** qui le sait déjà avec certitude (`engine.next_state`, pur, quelques
microsecondes). `ShieldedAI` (`ai_hybrid.py`) enveloppe n'importe quel agent :
avant de jouer son coup préféré, elle vérifie récursivement (profondeur
`SHIELD_DEPTH=3`) qu'une suite d'actions survivante existe ; sinon elle
remplace le coup par le meilleur repli sûr (même priorité que `HeuristicAI` :
AVANCER > proximité du centre > RECULER). C'est un filet, pas un planificateur :
il n'interdit que le prouvé-mortel, il ne cherche pas le meilleur chemin.

### Résultat (bench 50 graines, déterministe)

| Agent | Score moyen | Médiane | Victoires | Collision | Noyade |
|---|---:|---:|---:|---:|---:|
| `dqn` | 91,3 | 78 | 8/50 | 19 | 11 |
| **`dqn-shield`** | **165,1** | **200** | **35/50** | 2 | 2 |
| `clone` | 51,4 | 40 | 1/50 | 31 | 14 |
| **`clone-shield`** | **153,4** | **200** | **29/50** | 8 | 2 |

Hypothèse confirmée sans ambiguïté : les collisions et noyades s'effondrent
(19→2, 31→8), la médiane saute directement à 200, et **70 % des parties du
dqn deviennent des victoires** (35/50). Temps de décision toujours sous
budget (max 9,5 ms sur 20 ms — le pire cas, une position acculée qui demande
d'explorer plus de branches avant de trouver une issue). Il reste 15 parties
perdues sur 50 : 7 stagnations (le filet ne couvre pas ce cas, non local),
4 sorties d'écran + 2-3 collision/noyade — des positions où **aucune** suite
de 3 coups ne survit (le filet le confirme honnêtement plutôt que de
prétendre sauver l'insauvable).

### La limite honnête : le filet s'effondre sous bruit

| Agent | bruit 0 → 0,1 | Victoires |
|---|---:|---:|
| `dqn` | 91,3 → 18,0 | 8/50 → 0/50 |
| `dqn-shield` | 165,1 → **19,6** | 35/50 → 0/50 |
| `clone` | 51,4 → 19,2 | 1/50 → 0/50 |
| `clone-shield` | 153,4 → **28,2** | 29/50 → 0/50 |

Sous bruit, le gain du filet fond quasiment (dqn-shield à peine au-dessus de
dqn brut). **Explication précise, pas une vague « le bruit perturbe tout »** :
`next_state` calcule la survie en supposant que la turbulence *actuellement
enregistrée* (`_noise_offsets`) reste valable au tick suivant — or
`Engine._apply_noise()` re-tire une turbulence aléatoire à CHAQUE tick réel.
Le filet vérifie donc une survie sous une hypothèse qui sera probablement
fausse une fraction de seconde plus tard : une confiance non garantie, pas
une absence de bénéfice. C'est très exactement la même faiblesse qui fait
s'effondrer `search` (200→28) et `guided` (200→22) sous bruit — tout
mécanisme qui suppose un futur connu perd cette hypothèse quand le monde la
retire à chaque tick. Un filet correct sous bruit devrait vérifier la survie
sur plusieurs tirages de turbulence (expectimax local) — piste ouverte,
jointe à la « planification robuste au bruit » déjà identifiée.

### Statut mis à jour de la piste A

**✅ Implémentée et mesurée : gain massif en monde déterministe (91→165,
51→153 ; victoires 8→35 et 1→29 sur 50), gain marginal sous bruit (limite
comprise et documentée).** Reproductible : `--shield` (`--shield-depth` pour
ajuster l'horizon) sur `play`, `duel`, `bench`. Taxonomie : `dqn-shield` et
`clone-shield` sont des **hybrides** (comme `guided`) — le réseau décide, le
filet vérifie ; ce n'est plus une IA pure sans aide extérieure au jeu.

## 5. Feuille de route vers « 200 à tous les coups »

Le constat de départ (avant le filet) : le dqn mourait **en jouant**
(collisions 19, noyades 11, sorties 9), plus en hésitant. Or gagner exige
~2000 décisions consécutives sans une seule erreur fatale : même 99,9 % de
décisions justes → ~87 % de parties perdues. La piste A vient de combler
l'essentiel de cet écart en monde déterministe (35/50 victoires) ; il reste
deux chantiers, du plus mûr au plus exploratoire :

### Piste A′ — approfondir le filet (prolongement direct, déjà mesuré comme rentable)

- **Filet robuste au bruit** : vérifier la survie sur plusieurs tirages de
  turbulence plutôt qu'un seul (expectimax local sur `_survives`) — cible
  directement la limite mesurée ci-dessus.
- **Profondeur > 3** : le budget est à peine entamé (9,5 ms sur 20) ; tester
  `--shield-depth 5` ou 7 pourrait rattraper une partie des 7 stagnations et
  4 sorties d'écran restantes (impasses qu'un horizon plus long verrait venir).
- **Couvrir la stagnation** : ajouter un critère « pas de progrès en Y depuis
  N ticks » à la vérification, pour que le filet agisse aussi contre les
  parties qui trament sans mourir.

### Piste B — pousser le réflexe pur, sans filet (le vrai défi, statut inchangé)

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

Pronostic mis à jour : la piste A a déjà démontré qu'un « réseau + filet »
gagne 70 % des parties en monde déterministe — au-delà de ça (impasses à
3+ coups, robustesse au bruit), la piste A′ est le prolongement le plus
rentable. La piste B (réflexe pur, sans aucune vérification externe) reste
le vrai défi scientifique : peut-elle un jour se passer du filet ? Pronostic
honnête : viser 120-160 de moyenne en pur réflexe est réaliste ; approcher
200/200 **sans filet ni recherche** demanderait probablement une architecture
qualitativement différente (piste B.2) plutôt qu'un réglage de plus.

### Chantiers annexes (indépendants)

- **PPO en vision grille** : ppo est resté aux 42 capteurs (7,0) ; lui donner
  `sense_grid` + alignement est un test à ~1 h qui dirait si le policy
  gradient profite autant que la valeur.
- **Lignes de train** (gameplay), **tournoi Elo** : voir « Pistes restantes »
  de ROBOTS.md.

## 6. Reprendre le projet en 5 minutes

```bash
uv sync                                                  # environnement
uv run python main.py --mode play --ai dqn --shield      # voir jouer le champion (avec filet)
uv run python main.py --mode bench --episodes 50 --shield # reproduire les tableaux du filet
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
