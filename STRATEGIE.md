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
| Shaping « plateforme » aide | idem | 13,0 → 12,3 | neutre |
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

## 5. Épisode 5 — piste A′ : le filet approfondi passe à 43/50, piste B.1 réfutée

Deux expériences lancées en parallèle dès la reprise de session.

### Sweep de profondeur (3 / 5 / 7)

| Agent | d=3 | d=5 | d=7 |
|---|---:|---:|---:|
| `dqn-shield` | 165,1 (35/50) | **178,5 (39/50)** | 178,5 (39/50, plateau) |
| `clone-shield` | 153,4 (29/50) | 167,6 (34/50) | **180,4 (40/50)** |

Le budget de décision reste large (max 4,7 ms sur 20). Le dqn plafonne à
d=5 ; le clone (réseau plus faible au départ) continue de profiter d'une
vérification plus profonde jusqu'à d=7. **`SHIELD_DEPTH` passé à 7.** Effet
collatéral net : collisions et noyades quasi éliminées des deux agents — les
échecs restants (9 et 8 sur 50) sont presque tous des **stagnations**, pas
des impasses locales. Diagnostic confirmé : le filet de survie ne peut rien
contre un agent qui ne meurt pas mais tourne en rond.

### Sauvetage anti-stagnation (nouvelle brique)

Ajout à `ShieldedAI` : quand `ticks_since_progress >= SHIELD_RESCUE_AFTER`
(60), un BFS exact borné sur `next_state` (largeur × lignes × profondeur
états, dédupliqués) cherche le premier coup d'un chemin qui atteint une
NOUVELLE ligne max en ≤ `SHIELD_RESCUE_DEPTH` (8) ticks, et l'impose s'il
est sûr. Résultat (50 graines, déterministe, profondeur 7) :

| Agent | sans sauvetage | **avec sauvetage** | stagnations |
|---|---:|---:|---:|
| `dqn-shield` | 178,5 (39/50) | **187,9 (43/50)** | 9 → 4 |
| `clone-shield` | 180,4 (40/50) | **185,0 (42/50)** | 8 → 6 |

**Le dqn atteint 86 % de victoires en monde déterministe.** Coût : pire temps
de décision mesuré 18,0 ms (budget 20 ms — marge devenue mince, à surveiller
si on pousse encore la profondeur). Limite assumée pour les stagnations
restantes : certains sauvetages exigent d'attendre le retour d'un tronc sur
son cycle complet (jusqu'à ~57 ticks), hors de portée d'un BFS à 8 coups —
aller plus loin reviendrait à réimplémenter `search` en moins bien, ce qui
n'est pas l'esprit d'un filet.

### Piste B.1 réfutée : élargir le cône de vision dégrade, sans exception

Hypothèse testée : les erreurs restantes du réflexe pur viennent d'un cône
trop court (t..t+3, 6 lignes). Quatre variantes, 4000 épisodes × 2 graines,
bench 25 graines :

| Variante | Entrée | Bench (moyenne 2 graines) |
|---|---:|---:|
| **base (t..t+3, 6 lignes)** | 482 | **83,4** |
| + 2 lignes (rows6) | 642 | 61,6 |
| + 2 ticks (t5) | 710 | 49,3 |
| + 4 ticks (t7) | 938 | 36,9 |
| cumul (t7-rows6) | 1250 | 28,0 |

**Dégradation monotone, sans une seule exception sur 8 runs.** Même
diagnostic que la mémoire par pile et le réseau à 128 neurones (Épisode 2) :
à budget d'entraînement fixe, chaque paramètre d'entrée supplémentaire coûte
plus qu'il n'apporte. **Piste B.1 classée non retenue** ; le cône actuel
(t+3, 6 lignes) reste la config optimale mesurée. La piste B reste ouverte
via B.2 (convolution 1D) et B.3 (DQfD/init clone) — des changements
d'architecture, pas de taille d'entrée.

## 6. Épisode 6 — piste A″ testée et réfutée : le filet robuste au bruit ne suffit pas

Implémenté `_survives_noisy` : au lieu de figer la turbulence courante,
chaque tick simulé re-tire une turbulence stochastique (miroir exact
d'`Engine._apply_noise`, sur une copie locale des offsets — l'état réel
n'est jamais modifié). La vérification exige qu'une fraction
(`SHIELD_NOISE_THRESHOLD`) de `SHIELD_NOISE_SAMPLES` tirages indépendants
trouve une suite survivante sur `SHIELD_NOISE_DEPTH` ticks. Activé
automatiquement sous `--noise` (le mode déterministe est inchangé).

**Résultat (50 graines, bruit 0,1) : aucun gain.**

| Agent | déterministe (filet exact) | + filet robuste au bruit | sans filet |
|---|---:|---:|---:|
| `dqn-shield` | — | 19,6 | 18,0 |
| `clone-shield` | — | 28,2 | 19,2 |

Chiffres identiques (aux arrondis près) à ceux du filet simple déjà mesurés.
**Diagnostic précis, pas un simple « ça n'a pas aidé »** : instrumentation du
taux d'intervention — **0 % sur 207 ticks réels**, y compris en relevant le
seuil d'exigence à 0,9 (quasi tous les tirages doivent survivre). Le
mécanisme « existe-t-il une suite qui survit » est presque toujours vrai dès
qu'on autorise 5 actions sur quelques ticks (rester sur place suffit souvent
à survivre localement) — ce n'est donc PAS un problème de réglage de seuil
ou de profondeur : la question posée est la mauvaise question. Un filet
réellement robuste au bruit devrait comparer la VALEUR espérée des actions
(expectimax complet sur plusieurs tirages), pas leur existence — une
reconstruction, pas un ajustement. **Classé non retenu pour l'instant** ;
le code reste dans `ai_hybrid.py` (inerte sous bruit, sans coût en
déterministe) pour une reprise éventuelle avec la bonne formulation.

## 7. Épisode 7 — piste B.2 : la convolution 1D bat le MLP dense de +45 %

Diagnostic de départ : B.1 (Épisode 5) a prouvé que le problème n'est pas
« voir plus loin » mais « mieux exploiter ce qui est déjà vu ». Un MLP dense
doit réapprendre chaque motif de praticabilité à chaque position de colonne
indépendamment (482 entrées, aucun paramètre partagé) ; une convolution 1D
partage ses poids entre colonnes — « un trou est un trou, où qu'il soit ».

### Implémentation (`conv_neural.py`)

`ConvQNet` : la carte de praticabilité de `sense_grid` (6 lignes × 4 plans
temporels × 19 colonnes) est vue comme un signal 1D de 24 canaux (ligne ×
plan) sur 19 positions ; convolution à noyau impair, padding « same »
(poids partagés entre colonnes), tanh, aplatissement + concaténation des
scalaires par ligne (type, vitesse) qui contournent la convolution (aucune
structure spatiale à exploiter pour eux), puis tête dense classique
(cachée tanh → logits). Même API que `neural.MLP`
(forward/backward/adam_step/get_flat/copy/save/load) — substituable partout
où `MLP` est utilisée. **Gradients vérifiés numériquement sur les 6
groupes de paramètres (conv et denses) : erreur relative maximale 7,75e-10**
(précision machine), avant tout entraînement.

### Ablation (4000 épisodes, 2 graines, bench 25 graines)

| Config | Bench moyen | vs MLP |
|---|---:|---:|
| MLP dense (référence) | 83,4 | — |
| **conv 6 canaux, noyau 5** | **120,6** | **+45 %** |
| conv 8 canaux, noyau 5 | 100,7 | +21 % |
| conv 8 canaux, noyau 7 | 76,2 | −9 % |

Verdict net et reproductible sur les deux graines de chaque variante :
**la convolution bat le MLP dense**, et **le même pattern qu'aux Épisodes
2/3/5 se reproduit une fois de plus** — plus de capacité (8 canaux, noyau 7)
nuit à budget d'entraînement fixe ; la config la plus légère (6 canaux,
noyau 5) gagne. C'est la première fois qu'un changement d'ARCHITECTURE bat
la simple représentation — pas un réglage de plus.

### Intégration production

`config.DQN_ARCHITECTURE = "conv"` (nouveau défaut, mesuré gagnant) pilote
`ai_qlearning._build_net()` ; `load_dqn_net()` charge indifféremment un
ancien modèle MLP ou un modèle conv (marqueur `kind` dans le `.npz`,
rétro-compatible). Vérifié : `--shield` fonctionne tel quel avec un DQN
conv (le filet est agnostique de l'architecture interne de l'agent enveloppé).

**Campagne de production** (10 graines × 12 000 épisodes, config conv
6 canaux/noyau 5) : validations remarquablement stables et hautes —
165,0 / 165,0 / 172,8 / 173,5 / 176,6 / 179,4 / 181,4 / 183,2 / 183,4 /
**190,3** (retenue). À comparer à la campagne MLP-grid de l'Épisode 2
(25,2 à 38,8) : non seulement le sommet est plus haut, mais **toute la
distribution est décalée** — la pire graine conv (165,0) bat de loin la
meilleure graine MLP (38,8).

**Bench final officiel (50 graines) :**

| Agent | Déterministe | Bruit 0,1 |
|---|---:|---:|
| `dqn` (conv, sans filet) | **146,3**, méd 200, **28/50 victoires** | 19,2, 0/50 |
| **`dqn-shield` (conv + filet)** | **194,9**, méd 200, **46/50 (92 %)** | 19,2, 0/50 |

**Le réflexe pur seul (146,3, 28/50) dépasse déjà l'ancien DQN+filet MLP
(91,3 → 165,1)** — la représentation en vision grille était bonne, mais le
MLP dense la gâchait par-dessous. Combiné au filet (déjà mesuré comme
rentable, Épisode 5), le nouveau DQN atteint **194,9 de moyenne et 92 % de
victoires** — à 5 points du plafond théorique de 200. Il ne reste que
4 défaites sur 50 (3 stagnations, 1 collision) : la piste B.2 + A′ ensemble
sont, de loin, le résultat le plus fort du projet côté apprentissage. Sous
bruit, le filet reste inerte comme diagnostiqué à l'Épisode 6 (dqn-shield =
dqn, 19,2 dans les deux cas) — la limite de robustesse au bruit demeure
entière et n'est pas résolue par ce changement d'architecture.

## 8. Feuille de route vers « 200 à tous les coups »

Le constat de départ (avant le filet) : le dqn mourait **en jouant**
(collisions 19, noyades 11, sorties 9), plus en hésitant. Les pistes A, A′,
puis B.2 ont comblé l'essentiel de cet écart en monde déterministe :
**46/50 victoires, 194,9 de moyenne, à 5 points du plafond théorique.**
A″ (bruit) a été testée et n'a pas suffi en l'état — une reconstruction,
pas un réglage, la reporterait, et le nouveau DQN conv n'y change rien
(19,2 avec ou sans filet sous bruit, identique à l'ancien MLP).

### Piste B — pousser le réflexe pur, sans filet (le vrai défi)

1. ~~Élargir le cône de vision~~ — **testé et réfuté** (Épisode 5).
2. ~~Architecture convolutionnelle 1D~~ — **testée, confirmée, en
   production** (Épisode 7). Seule ou combinée au filet, c'est le
   meilleur résultat du projet côté apprentissage : réflexe pur 146,3
   (28/50), + filet 194,9 (46/50, 92 %).
3. **Reste à fermer (4 défaites/50, déterministe)** : 3 stagnations que ni
   le réflexe conv ni le sauvetage BFS à 8 coups ne rattrapent (cycles de
   tronc trop longs), 1 collision résiduelle. Marge de progression faible
   mais non nulle — `SHIELD_RESCUE_DEPTH` adaptatif, ou plus d'épisodes
   d'entraînement conv (12 000 → 20 000+, coût modéré).
4. **DQfD / initialisation par le clone** : pré-remplir le replay avec des
   transitions de `search` (ou initialiser le réseau Q depuis la politique du
   clone-DAgger) — le RL part alors d'un comportement déjà compétent au lieu
   de redécouvrir la marche. Littérature : Hester et al. 2018.
5. **Masquage des actions fatales PENDANT l'entraînement** : ne jamais
   laisser epsilon jouer un coup prouvé-mortel → toute l'exploration se
   concentre sur les vrais choix. (Si on garde le masque au jeu, on retombe
   dans la piste A — à trancher.)

Le vrai chantier restant est **la robustesse au bruit** (piste A″, non
résolue par aucun des deux leviers testés) — c'est la seule zone où le
projet est encore loin d'une solution, déterministe ou apprise.

### Chantiers annexes (indépendants)

- **PPO / clone en convolution** : seul le dqn a été testé ; étendre
  `ConvQNet` à `ai_ppo.py` et `ai_hybrid.py` (imitation) est un prolongement
  direct à faible risque (l'architecture a déjà fait ses preuves).
- **Lignes de train** (gameplay), **tournoi Elo** : voir « Pistes restantes »
  de ROBOTS.md.

## 9. Reprendre le projet en 5 minutes

```bash
uv sync                                                  # environnement
uv run python main.py --mode play --ai dqn --shield      # voir jouer le champion (avec filet)
uv run python main.py --mode bench --episodes 50 --shield # reproduire les tableaux du filet
uv run python main.py --mode train-dqn --workers 10      # réentraîner dqn (~15 min)
uv run python main.py --mode train-clone                 # réentraîner clone (BC+DAgger)
```

Fichiers clés : `config.py` (tous les choix mesurés sont les défauts ; les
briques rejetées restent activables : FRAME_STACK, DQN_NSTEP, DQN_DOUBLE,
DQN_CURRICULUM_PROB, RL_SHAPING, DQN_SENSOR, DQN_ARCHITECTURE=mlp pour
revenir au dense), `sensors.py` (les deux générations de capteurs + dispatch
automatique par taille d'entrée du modèle), `shaping.py` (potentiels),
`conv_neural.py` (`ConvQNet`, l'architecture convolutive de l'Épisode 7),
`ai_qlearning.py` (DQN + multi-graines + `_build_net`/`load_dqn_net` — bascule
mlp/conv), `ai_hybrid.py` (imitation + DAgger + recherche guidée + `ShieldedAI`
filet de sécurité + sauvetage anti-stagnation).

Détail d'exécution : les threads BLAS sont épinglés à 1 dans `main.py`
(matrices minuscules : le multi-threading coûtait jusqu'à 10× ; ne pas retirer).
