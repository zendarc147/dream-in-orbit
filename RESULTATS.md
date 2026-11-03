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
