# Journal des résultats

Validation toujours via `train.py` → `predict.py` → `evaluate.py` (script `valider.py`).

- **Directe** : entraînement sur annee1, test sur `validation/` (annee2).
- **Inverse** : entraînement sur annee2 seul, test sur `validation_inverse/` (annee1).
- **Score de référence** = moyenne des deux RMSE. Un changement n'est gardé que s'il la fait baisser.

Note : les `power.csv` bruts ne sont pas versionnés. `reconstruire_power.py` les recrée à
partir de `data/prepared/*/y.csv` (une ligne par heure). Contrôle : la validation directe
régénérée ainsi est identique à `validation/` (écart max 4e-22).

| Étape | Changement | RMSE directe | RMSE inverse | Moyenne | Durée (train + predict, par direction) |
|---|---|---:|---:|---:|---|
| E0 | Constante (moyenne par circuit de l'année d'entraînement) | 0.1058 | 0.1252 | 0.1155 | < 1 s |
| E1 | Boosting 300 it., 31 feuilles, sans recalage | 0.0981 | 0.1095 | 0.1038 | 20 s + 5 s |
| E1 | + recalage constant sur power_init, poids 0.25 (meilleur de 0.25/0.5/0.75/1) | 0.0970 | 0.1073 | 0.1021 | idem |
| E1 | + recalage qui s'estompe, poids 1, demi-vie 1440 h (testé 24 h → 4000 h) | 0.0974 | 0.1020 | 0.0997 | idem |
| E1 | Boosting régularisé 150 it., 15 feuilles, ≥ 200 obs./feuille | 0.0959 | 0.0994 | 0.0977 | 12 s + 5 s |
| E1 | Boosting 600 it., 63 feuilles (plus gros) | 0.0994 | 0.1041 | 0.1018 | 33 s + 5 s |
| **E1** | **Boosting 100 it., 7 feuilles, ≥ 500 obs./feuille + recalage (retenu)** | **0.0951** | **0.0999** | **0.0975** | 9 s + 5 s |

E1 : cible < 0.095 manquée de peu après 3 essais (0.0951 en directe). Constat : un
recalage constant sur toute l'année dégrade l'inverse (0.139 avec poids 1) ; le niveau
des 7 premiers jours ne vaut que pour quelques semaines. Les modèles plus simples
généralisent mieux d'une année à l'autre.

| Étape | Changement | RMSE directe | RMSE inverse | Moyenne | Durée (train + predict, par direction, CPU 4 cœurs) |
|---|---|---:|---:|---:|---|
| E2 | GRU selon le plan : fenêtres 7 j + 60 j, 400 pas (inverse non mesurée) | 0.1148 | – | – | 680 s + 6 s |
| E2 | + épisodes d'un an, BPTT tronquée par 30 j, entrées bornées à la plage d'entraînement | 0.1094 | – | – | 485 s + 7 s |
| E2 | 60 / 150 pas ; sans ltdata ; plus régularisé (taille 64, dropout 0.3) | 0.107 à 0.133 | – | – | – |
| **E2** | **+ variables quasi constantes écartées (mode ≥ 99,5 %), z-scores bornés à ±5, 300 pas (retenu)** | **0.0901** | **0.0989** | **0.0945** | 250 s + 7 s |

E2 : diagnostic de l'échec initial. L'erreur montait au fil de l'année (0.075 sur les
premières semaines, 0.14 à 0.17 en fin de période). Un MLP sans mémoire faisait pire (0.156) :
le problème venait donc des entrées, pas de la récurrence. Cause : 17 variables presque
constantes sur annee1 (`saaf_coverage_fraction`, `saaf_missing`, `*_unknown`, `*_outside_recorded_range`…)
avaient un écart-type minuscule ; dès qu'elles bougent sur annee2 (61 % des heures pour la
couverture saaf), les entrées normalisées explosent. Après filtrage, 48 variables sur 65 restent.
Écart au plan : les fenêtres de 60 jours sont remplacées par des épisodes d'un an coupés en
morceaux de 30 jours (l'état caché passe d'un morceau à l'autre). Le simulateur s'entraîne
ainsi sur l'horizon réel de la simulation.

| Étape | Changement | RMSE directe | RMSE inverse | Moyenne | Durée |
|---|---|---:|---:|---:|---|
| E2 | Même config, graine 1 (mesure du bruit de graine : ≈ 0.0015 sur la moyenne) | 0.0878 | 0.0982 | 0.0930 | 420 s + 7 s |
| E3 | Variables v2 complètes (399 : fenêtres 3/6/12/24 h passées et futures sur éclipses, comms, 14 modes ftl, 12 comptes dmop ; depuis/jusqu'à ; phase orbitale ; cos/sin saaf) | 0.0990 | 0.1061 | 0.1025 | 450 s + 7 s |
| E3 | v1 + état « dernière commande » des 12 sous-systèmes dmop (136 variables) | 0.0973 | – | – | – |
| E3 | v2 réduite (v2r) : fenêtres et depuis/jusqu'à seulement sur éclipses + comms, phase orbitale, cos/sin saaf | 0.0878 | 0.0987 | 0.0933 | 330 s + 7 s |

E3 : avec une seule trajectoire annuelle pour apprendre, ajouter beaucoup de variables
fait surapprendre (v2 complète : +0.008). L'état des sous-systèmes dmop corrèle à 0.75
avec NPWD2551 sur annee1 (AXXX301A) mais dégrade la validation directe : relation
propre à une année. Seule la version réduite reste au niveau de v1.
| E3 | Ensemble de 5 graines, variables v1 (moyenne = prédiction, écart-type = incertitude) | 0.0876 | 0.0975 | **0.0925** | 950 s (2 directions en parallèle) + 10 s |
| E3 | Ensemble de 5 graines, variables v2r | 0.0877 | 0.0980 | 0.0928 | 1070 s + 10 s |

Entraînement CPU : quatre processus mono-thread (un par graine) donnent 2,5 fois le débit
d'un processus à 4 threads ; `valider.py` lance aussi les deux directions en parallèle.

Analyse de l'ensemble v1 : 2551 et 2451 font 52 % de l'erreur en directe, 34 % en inverse.
Plusieurs circuits sont prédits plus mal qu'une constante à leur vraie moyenne : 2531
(RMSE 0.167 pour un écart-type de 0.081 en directe), 2851, 2881 et 2561. Leur niveau change
d'une année à l'autre (2531 : moyenne 0.125 sur annee1, 0.013 sur annee2).
| It. 1 | Niveau moyen de power_init donné au simulateur à chaque heure (« statique »), 5 graines | 0.0886 | 0.1009 | 0.0948 | 1100 s + 10 s |

Itération 1 non retenue : dans une même année, le niveau initial reste informatif et le
modèle s'y fie ; d'une année à l'autre, il trompe (inverse +0.0034).

Ablations (`ablations.py`, 2 graines par direction, détail dans `ABLATIONS.md`) :
retirer la mémoire coûte +0.0049, les éclipses +0.0033, saaf +0.0029, dmop +0.0009.
| It. 2 | 200 pas au lieu de 300, 5 graines | 0.0876 | 0.0989 | 0.0933 | 640 s + 12 s |
| **It. 3** | **Bruit gaussien σ = 0.3 sur les entrées normalisées pendant l'entraînement, 5 graines (retenu, final)** | **0.0872** | **0.0971** | **0.0921** | 1060 s + 11 s |

Arrêt : trois itérations de suite sans gain > 0.001 (statique −0.0023, 200 pas −0.0008,
bruit +0.0004). Le bruit est gardé, car il améliore la moyenne, même si le gain reste
sous le bruit de graine.

Modèle final (`poids/`) : même configuration, entraîné sur annee1 + annee2. Entraînement
717 s sur CPU 4 cœurs ; prédiction 10 s ; poids 3,4 Mo.
| It. 4 | Spécialiste 2551 + 2451 (perte sur ces 2 circuits seulement) remplaçant le généraliste sur ces colonnes | 0.0877 | 0.0985 | 0.0931 | +1030 s |
| **It. 5** | **Ensemble mixte global : moyenne à poids égal (tous circuits) de l'ensemble v1 + bruit et de l'ensemble v2r (retenu, final)** | **0.0863** | **0.0964** | **0.0914** | 2 × train + 18 s |
| It. 6 | + 3e membre (v1 sans bruit) | 0.0864 | 0.0963 | 0.0913 | non retenu : gain 0.00005 pour +50 % d'entraînement |
| It. 7 | 2531, 2851, 2881, 2561 remplacés par leur moyenne sur power_init | 0.0816 | 0.1275 | 0.1046 | non retenu |

Itérations sur 2551 et 2451. Le spécialiste fait moins bien que le généraliste sur ses
propres circuits (2551 : 0.280 contre 0.277 en directe) : apprendre les 33 circuits
ensemble aide. Le mélange v1/v2r améliore surtout 2551 (0.277 → 0.271 en directe,
0.236 → 0.228 en inverse). Il est appliqué globalement, avec le même poids pour chaque
circuit, et améliore les deux directions (directe 0.0872 → 0.0863, inverse 0.0971 → 0.0964).
La persistance du niveau initial aide en directe mais ruine l'inverse.

Modèle final (`poids/`, `mixte.py`) : 2 membres × 5 graines, entraînés sur annee1 + annee2.
Entraînement 2640 s sur CPU 4 cœurs (ralenti par des tâches concurrentes ; ≈ 25 min seul).
Prédiction 18 s. Poids : 7 Mo. Temps sur GPU T4 non mesuré (pas de GPU ici) ; sur Colab,
`python train.py` utilise CUDA automatiquement, à lancer par toi pour vérifier.

## Session « améliorations » (branche `ameliorations`)

Protocole : chaque essai via `valider.py` (ensemble mixte complet, 2 membres × 5 graines,
deux directions). Un changement est gardé si la moyenne baisse de plus de 0.0015 sans
dégrader aucune direction ; entre 0.0005 et 0.0015, confirmation avec les graines 5 à 9.
Uniquement des réglages globaux (aucun choix circuit par circuit). Machine : CPU 16 cœurs,
pas de GPU ; durée = entraînement des deux directions en parallèle + prédictions.

| Essai | Changement | RMSE directe | RMSE inverse | Moyenne | Durée |
|---|---|---:|---:|---:|---|
| Réf. | Ensemble mixte actuel, relancé sur cette machine (reproduit à l'identique) | 0.0863 | 0.0964 | 0.0914 | 1097 s |
| A1 | Moyenne exponentielle des poids (EMA 0.98), 300 pas | 0.0863 | 0.0965 | 0.0914 | 1035 s |
| A2 | EMA 0.99 + 600 pas (au lieu de 300) | 0.0855 | 0.0961 | 0.0908 | 2703 s |
| A2 bis | Confirmation, graines 5 à 9 : référence | 0.0869 | 0.0966 | 0.0917 | 2583 s (en parallèle d'A2 ter) |
| **A2 ter** | **Confirmation, graines 5 à 9 : EMA 0.99 + 600 pas (retenu)** | **0.0851** | **0.0963** | **0.0907** | 4800 s (en parallèle) |

A1/A2 : l'EMA seule ne change rien à 300 pas (le OneCycle termine déjà avec un taux
d'apprentissage quasi nul, ce qui lisse les derniers pas). Doubler les pas avec une
EMA plus longue aide un peu : −0.0006 sur les graines 0 à 4, −0.0011 sur les graines
5 à 9, les deux directions s'améliorent dans les deux cas. Retenu par défaut
(`pas` = 600, `ema` = 0.99). L'entraînement dure deux fois plus longtemps.

Pour économiser le calcul, les pistes suivantes sont d'abord criblées à 300 pas contre
la référence à 300 pas (0.0914) ; un changement retenu est ensuite vérifié à 600 pas.
| B1 | Regard appris : Conv1d non causale ±24 h (32 canaux) avant le GRU, sur les deux membres, 300 pas | 0.0868 | 0.0986 | 0.0927 | 2476 s (en parallèle d'A2 ter) |
| C1 | Correction de niveau issue de power_init qui s'estompe : écart normalisé de chaque circuit sur les 7 jours × poids appris × 2^(−t / demi-vie apprise), 300 pas | 0.0860 | 0.0941 | 0.0900 | 844 s |
| C1 bis | Idem, graines 5 à 9 (référence 300 pas graines 5 à 9 : 0.0917) | 0.0864 | 0.0945 | 0.0905 | 1869 s (en parallèle) |
| **C2** | **Correction de niveau + EMA 0.99 + 600 pas, graines 0 à 4 (retenu)** | **0.0853** | **0.0940** | **0.0897** | 2978 s (en parallèle) |

C : le biais résiduel venait surtout des circuits dont le niveau change d'une année à
l'autre (biais de l'ensemble de référence : 2531 +0.11 en directe, −0.095 en inverse ;
2851 −0.089 / +0.099). Le niveau des 7 premiers jours prédit bien les 2 à 4 premiers mois
(inverse : 2531 vaut 0.57 sur power_init et ≈ 0.5 pendant 4 mois, contre 0.013 en moyenne
sur l'année d'entraînement), puis plus du tout. La correction est un terme additif de la
sortie, global (même forme pour tous les circuits) : poids et demi-vie par circuit sont
des paramètres du réseau, appris par descente de gradient sur l'entraînement seul,
initialisés à 0 et 30 jours. Les poids appris les plus forts vont à 2531, 2851, 2881 ;
les demi-vies restent proches de 30 jours (660 à 750 h). Gain surtout en inverse
(−0.0024). Confirmé sur les graines 5 à 9 (−0.0013), puis à 600 pas (−0.0011).
| D1 | Régularisation : taille 96, dropout 0.25, weight_decay 1e-3 (avec correction de niveau, 300 pas ; à comparer à C1) | 0.0867 | 0.0937 | 0.0902 | 877 s (en parallèle de C2) |
| D2 | Bruit 0.3 aussi sur le membre v2r (300 pas ; à comparer à C1) | 0.0860 | 0.0941 | 0.0900 | 862 s |
| E1 | 3e membre de forme différente (v2r, bruit 0.3, taille 64, morceaux de 15 jours), poids égal (300 pas ; à comparer à C1) | 0.0861 | 0.0942 | 0.0902 | 1008 s |
| F1 | Bruit « lent » σ = 0.2 : décalage constant par épisode et par variable sur les entrées normalisées (300 pas ; à comparer à C1) | 0.0860 | 0.0947 | 0.0904 | 840 s |

D, E, F : aucune régularisation supplémentaire ni membre de plus n'aide. Arrêt de la
session après quatre essais de suite sans gain retenu (B1 compté avant C ; D1, D2, E1, F1).

Modèle final (`poids/`) : ensemble mixte (v1 + bruit 0.3, v2r ; 2 × 5 graines), 600 pas,
EMA 0.99, correction de niveau apprise. Pour le modèle final seulement, annee1 et annee2
(contiguës, à une heure près) sont fusionnées en une seule période : les variables
glissantes et les épisodes d'entraînement traversent la frontière (piste non mesurable
en validation, où une seule année sert à l'entraînement). Entraînement 1453 s sur CPU
16 cœurs (en parallèle des ablations) ; prédiction 8 s sur 4 cœurs ; poids 6,9 Mo.
Démo régénérée avec les poids de C2 directe (entraînés sur annee1 seul).

Ablations relancées sur la nouvelle configuration (`ABLATIONS.md`, v1 seul, 2 graines) :
mémoire +0.0071, saaf +0.0027, dmop +0.0019, éclipses +0.0016. Durée 2495 s.

## Session « commandes » : AOOO, ATTT, ATMB

Même protocole (ensemble mixte complet, 2 membres × 5 graines, 600 pas, deux directions ;
gardé si la moyenne baisse de plus de 0.0015 sans dégrader aucune direction ; entre 0.0005
et 0.0015, confirmation avec les graines 5 à 9). Calculs sur GPU T4 Kaggle
(`lancer_kaggle.py`, deux sessions au plus en parallèle) ; la référence est donc relancée
sur GPU. Durée = entraînement des deux directions en parallèle sur un même GPU.

Constat de départ : prepare.py ne compte que les 10 commandes dmop les plus fréquentes
(AAAA, APSF, AVVV) ; les commandes AOOO et ATTT n'arrivent au modèle que fondues dans
`dmop_other_count`. Or NPWD2881 s'allume avec ATTTF321A et s'éteint avec ATTTF321D ou
dans l'heure qui précède AOOOF01D0, AOOOF15A0, AOOOF02A1 (annee1 et annee2). Nouveau
module `commandes.py` : pour chaque commande AOOO/ATTT vue au moins 20 fois à
l'entraînement, une impulsion (nombre dans l'heure) et un état (1 tant que c'est la
dernière commande reçue par sa famille, ex. ATTTF321A/D), chacun lu à t + délai, le délai
(−12 h à +12 h) étant celui qui corrèle le mieux avec NPWD2881 sur l'année
d'entraînement seule.

| Essai | Changement | RMSE directe | RMSE inverse | Moyenne | Durée |
|---|---|---:|---:|---:|---|
| Réf. GPU | Configuration finale actuelle, relancée sur GPU T4 | 0.0851 | 0.0941 | 0.0896 | 465 s |
| G1 | + impulsions et états décalés des commandes AOOO, en entrée du GRU | 0.0802 | 0.0966 | 0.0884 | 548 s |
| G1b | Mêmes variables AOOO, mais dans une branche à part : petit réseau heure par heure (32 neurones) dont la sortie s'ajoute au seul NPWD2881 ; le GRU ne les voit pas | 0.0842 | 0.0918 | 0.0880 | 546 s |
| G2 | Impulsions et états AOOO + ATTT en entrée du GRU (sans branche) | 0.0819 | 0.0951 | 0.0885 | 652 s |
| **G2b** | **Impulsions et états AOOO + ATTT dans la branche vers NPWD2881 (retenu)** | **0.0832** | **0.0914** | **0.0873** | 651 s |
| G3b | G2b + regroupement des commandes inconnues de l'entraînement sur la commande connue la plus proche | 0.0827 | 0.0915 | 0.0871 | 655 s |
| G4 | G2b + consigne ATMBdddK (ddd lu comme un nombre, maintenu jusqu'à la suivante), en entrée du GRU | 0.0843 | 0.0915 | 0.0879 | 653 s |
| G5 | G2b + masque ATTT305C/305O/305P/306C/306P (1 pendant les K heures suivantes, K = 4 h appris sur NPWD2551), en entrée du GRU | 0.0832 | 0.0921 | 0.0877 | 660 s |

G1 : NPWD2881 gagne dans les deux directions (directe 0.143 → 0.115, inverse 0.173 → 0.143),
mais en entrée du GRU les commandes AOOO servent aussi à NPWD2451, 2551, 2532, avec des
relations propres à annee2 : l'inverse se dégrade (2451 : 0.213 → 0.242). La branche
(G1b) isole l'usage des commandes au circuit visé : les deux directions s'améliorent.
Ajouter ATTT (ATTTF321A/D allument et éteignent NPWD2881) amplifie le gain : avec G2b,
NPWD2881 passe de 0.143 à 0.110 en directe et de 0.173 à 0.118 en inverse ; les autres
circuits bougent à peine. Retenu : −0.0023 sur la moyenne, les deux directions baissent.

G3b (regroupement) : rattachements obtenus sans la puissance, par ressemblance des
commandes voisines dans la table dmop du test (annee2 : AOOOF01D1 → F01D0, F22R1 → F22A1,
ATTTF321P/R → F321D, F33A0/F34A0 → F63A0, F32A0/R0 → F02A0, ATTT309B → ATTTF310A…).
Gain de 0.0005 en directe, rien en inverse (annee1 n'a presque pas de commandes nouvelles
pour annee2) : −0.0002 sur la moyenne, sous le seuil, non retenu (`regroupement` reste une
option de `commandes.py`). G4 : l'hypothèse « ATMB = température » n'aide pas. G5 : le
masque n'existe pas sur annee1 (ATTT305/306 n'apparaissent qu'à partir d'août 2010) : en
directe la variable est constante à l'entraînement, donc écartée ; en inverse elle est
apprise mais absente du test. Effet non mesurable ici, et léger bruit en inverse.
| H1 | G2b, branche avec fenêtre : Conv1d non causale ±6 h sur les commandes au lieu de l'heure t seule | 0.0833 | 0.0912 | 0.0873 | 672 s |
| H2 | Idem, fenêtre ±12 h | 0.0833 | 0.0916 | 0.0874 | 680 s |
| H3 | G2b, branche vers les 33 circuits au lieu du seul NPWD2881 | 0.0819 | 0.0935 | 0.0877 | 650 s |
| H4 | G2b avec impulsions seules (sans états) | 0.0836 | 0.0916 | 0.0876 | 535 s |

H : les délais appris suffisent, une fenêtre ne change rien (H1 : −0.00004). Ouvrir la
branche à tous les circuits refait l'erreur de G1 en inverse. Les états (ATTTF321A/D)
comptent : sans eux, +0.0003. G2b devient la configuration par défaut (`monde.py` :
`commandes` = AOOO + ATTT, `branche_commandes` = NPWD2881).
| Contrôle | G2b relancé avec la configuration par défaut (aucune option) : identique, l'entraînement GPU est déterministe | 0.0832 | 0.0914 | 0.0873 | 658 s |
| I1 | G2b + 2e terme de correction de niveau, demi-vie initialisée à 180 jours | 0.0835 | 0.0919 | 0.0877 | 661 s |
| I2 | G2b, 900 pas et EMA 0.995 | 0.0838 | 0.0929 | 0.0884 | 961 s |
| I3 | G2b, 10 graines par membre au lieu de 5 | 0.0835 | 0.0908 | 0.0872 | 1287 s |
| I4 | G2b + correction de niveau sur le résidu : écart moyen entre power_init et la prédiction du simulateur sur ces 7 jours (état nul, sans gradient, borné à ±5), au lieu de l'écart à la moyenne d'entraînement | 0.0829 | 0.0901 | 0.0865 | 650 s |
| I5 | G2b, lots de 64 épisodes au lieu de 32 | 0.0834 | 0.0920 | 0.0877 | 1279 s |
| I6 | G2b, la branche lit aussi la sortie du GRU à l'heure t | 0.0835 | 0.0912 | 0.0874 | 648 s |
| I7 | G2b, taux d'apprentissage 2e-3 au lieu de 3e-3 | 0.0833 | 0.0918 | 0.0876 | 643 s |
| J1 | G2b, cibles de la branche choisies sur l'entraînement : circuits (écart-type ≥ 0.01) dont la corrélation maximale avec une variable de commande atteint 0.5 (annee1 : 2531 + 2881 ; annee2 : 2881) | 0.0827 | 0.0914 | 0.0870 | 647 s |
| J2 | J1 + regroupement des commandes nouvelles | 0.0821 | 0.0915 | 0.0868 | 647 s |
| I4 bis | Confirmation, graines 5 à 9 : G2b | 0.0840 | 0.0905 | 0.0873 | 675 s |
| **I4 ter** | **Confirmation, graines 5 à 9 : G2b + correction sur le résidu (retenu)** | **0.0833** | **0.0897** | **0.0865** | 680 s |

I4 : la correction de niveau recopiait l'écart de power_init à la moyenne d'entraînement,
y compris la part que le contexte explique déjà (Soleil, éclipses, saison). Sur le résidu,
elle ne garde que l'écart inexpliqué. Gain sur les circuits dont le niveau change d'une
année à l'autre (inverse : 2531 0.240 → 0.229, 2851 0.178 → 0.169). −0.0008 sur les
graines 0 à 4 et sur les graines 5 à 9, les deux directions baissent à chaque fois :
retenu (`niveau_residu` = True par défaut).
| K1 | I4 + cibles de la branche choisies sur l'entraînement (comme J1) | 0.0821 | 0.0901 | 0.0861 | 658 s |
| K2 | K1 + regroupement des commandes nouvelles | 0.0816 | 0.0902 | 0.0859 | 657 s |

J, K : choisir les cibles de la branche sur l'entraînement ajoute NPWD2531 en directe
(annee1 : corrélation 0.74 avec les commandes) et ne change rien en inverse (annee2 : 2881
seul, résultat identique au chiffre près). Gain de 0.0004, sous le seuil de confirmation :
non retenu. Avec le regroupement, l'inverse remonte de 0.0001 : non retenu non plus,
bien que la moyenne baisse de 0.0006.
| L1 | I4 + 2e terme de correction (demi-vie initialisée à 180 jours), sur le résidu | 0.0827 | 0.0892 | 0.0859 | 657 s |
| L2 | I4, taille 192 au lieu de 128 | 0.0829 | 0.0911 | 0.0870 | 1508 s |
| **L1 bis** | **Confirmation, graines 5 à 9 : I4 + 2e terme lent (référence I4 ter : 0.0833 / 0.0897 / 0.0865) (retenu)** | **0.0831** | **0.0894** | **0.0862** | 657 s |

L1 : sans le résidu, un second terme lent n'aidait pas (I1 : +0.0004). Sur le résidu, il
améliore les deux directions sur les graines 0 à 4 (−0.0005) comme sur les graines 5 à 9
(−0.0003) : gain petit mais retrouvé dans les quatre mesures. Retenu (`niveau_lent` = True).
| M1 | I4 + l'encodeur lit aussi l'écart heure par heure entre power_init et la prédiction (référence I4) | 0.0831 | 0.0898 | 0.0865 | 684 s |
| M2 | I4, BPTT par morceaux de 60 jours au lieu de 30 (référence I4) | 0.0827 | 0.0900 | 0.0863 | 1196 s |
| N1 | L1 + cibles de la branche choisies sur l'entraînement (comme J1, K1) | 0.0820 | 0.0892 | 0.0856 | 651 s |
| N3 | L1 + bruit 0.2 : sans effet, le membre v1 fixe son bruit à 0.3 dans `mixte.py` (résultat identique à L1) | 0.0827 | 0.0892 | 0.0859 | 657 s |

N1 : troisième mesure du choix automatique des cibles (J1 −0.0003, K1 −0.0004, N1 −0.0003).
L'inverse est identique par construction (annee2 ne désigne que 2881) : le gain moyen
reste sous 0.0005, non retenu malgré sa constance en directe.
| N4 | L1, dropout 0.2 au lieu de 0.1 | 0.0825 | 0.0893 | 0.0859 | 659 s |
| N5 | L1, EMA 0.995 au lieu de 0.99 | 0.0821 | 0.0897 | 0.0859 | 661 s |

Arrêt : après L1, six essais de suite sans gain retenu (M1, M2 sur la base I4 ; N1, N4, N5 ;
N3 nul). Configuration finale : ensemble mixte (v1 + bruit 0.3, v2r ; 2 × 5 graines),
600 pas, EMA 0.99, commandes AOOO + ATTT dans une branche vers NPWD2881, correction de
niveau sur le résidu avec un terme rapide et un terme lent. Validation (graines 0 à 4) :
directe 0.0827, inverse 0.0892, moyenne **0.0859**, contre 0.0851 / 0.0941 / 0.0896 au
départ de la session (−0.0037 ; graines 5 à 9 : 0.0862).

Modèle final (`poids/`) : configuration finale entraînée sur annee1 + annee2 fusionnées
(`python train.py`, lancé sur GPU T4 Kaggle) : 602 s d'entraînement ; prédiction 20 s sur
CPU 4 cœurs ; poids 6,9 Mo. La branche vers NPWD2881 lit 71 variables de commandes.
Démo régénérée avec les poids de L1 directe (entraînés sur annee1 seul, RMSE 0.0827).
Ablations relancées sur la configuration finale (`ABLATIONS.md`, v1 seul, 2 graines) :
mémoire +0.0079, dmop +0.0035 (dont branche AOOO/ATTT +0.0019), saaf +0.0028, éclipses −0.0002.
