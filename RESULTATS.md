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
