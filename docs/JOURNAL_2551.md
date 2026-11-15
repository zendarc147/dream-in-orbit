# Journal NPWD2551

Programme ciblé sur le seul circuit NPWD2551, à partir du modèle installé au commit `ff402a4`
(moyenne des 4 RMSE de validation : 0.076814). Les 32 autres colonnes ne doivent pas changer.

Règle d'installation, fixée avant toute mesure (celle de `outils/specialistes.py melanger`) : poids
choisis sur une direction et appliqués à l'autre ; la RMSE de 2551 doit baisser dans les 4 cas
(2 directions × graines 0–4 / 5–9), et aucune moitié de période ne doit monter de plus de 0.002.

## Point de départ

Base : `runs/fin_val_g04_sp`, `runs/fin_val_g59_sp` (modèle installé de bout en bout), liens
`runs/s51_base_g04|g59` pour l'outil de mélange.

| | RMSE 2551 | Part de la MSE totale | RMSE total si 2551 parfait | Biais (prédit − vrai) |
|---|---:|---:|---:|---:|
| Directe | 0.259 | 36 % | 0.060 | +0.030 |
| Inverse | 0.206 | 21 % | 0.070 | +0.010 |

- La part de 57 % citée au départ correspond à 2551 + 2451 (62 % en directe, 45 % en inverse),
  pas à 2551 seul.
- Bruit horaire 0.231 / 0.232. Autocorrélation du résidu : 0.59 à 1 h, 0.36 / 0.47 à 7 h (période
  orbitale), 0.31 / 0.35 à 24 h. La puissance du résidu est étalée sur toutes les bandes
  (> 30 j : 20 % / 29 %).
- Erreur très inégale dans l'année : RMSE par tranche de 60 jours de 0.04 à 0.44.
- Pente de calibration y ≈ 0.96 p en directe, 1.06 p en inverse : pas de correction affine
  transférable d'une direction à l'autre.

## Étape 1 : recombiner les composants actuels

`specialistes.py melanger`, base + GRU sans lettres + boosting R + GRU recette (`runs/s51_mel0`) :

| Cas | Base | Poids transférés |
|---|---:|---:|
| directe 0–4 | 0.2592 | 0.2592 |
| directe 5–9 | 0.2581 | 0.2575 |
| inverse 0–4 | 0.2063 | 0.2046 |
| inverse 5–9 | 0.2054 | 0.2038 |

Refusé : directe 0–4 ne baisse pas, et une moitié de l'inverse monte. Les poids globaux
(0.35 / 0.15 / 0.5) sont déjà presque les meilleurs pour 2551.

## Étape 2 : réglages de prédiction du boosting

Demi-vie du recalage sur power_init (2551 du boosting R seul, directe / inverse) : 1440 h (installé)
0.2701 / 0.2237 ; 360 h 0.2697 / 0.2242 ; 5760 h 0.2696 / 0.2237. Effet nul sur 2551.

## Étape 3 : nouveaux candidats

### Empilement local (`specialistes.py empiler`)

Second boosting pour 2551, appris sur les prédictions hors sac des 33 circuits du boosting R
(décalages ±1/±2 h, moyennes 6/24/168 h). Résultat 0.302 / 0.239, nettement pire que le boosting R
seul (0.270 / 0.224) : les prédictions hors sac sont trop bruitées. Dans le mélange (`runs/s51_mel1`,
avec le GRU physique `sp_phys`), poids nul, et le mélange est refusé.

### Boosting propre à 2551 (Kaggle CPU, `specialistes.py entrainer --circuits NPWD2551`)

Le boosting R s'arrêtait à la limite de 600 arbres sur 2551. Nouveaux réglages dans
`baselines/gbm_sacs.py` : D3 (D jusqu'à 3000 arbres), D3L (taux 0.01, jusqu'à 6000 arbres,
20 itérations sans gain), D31 (D3 avec 31 feuilles, 50 exemples par feuille).

Test : la part boosting de 2551 dans le modèle installé est remplacée par le candidat,
`base + 0.5 × (candidat − R)`. Aucun poids n'est choisi, donc rien n'est appris sur la validation,
hormis le choix entre les variantes.

| Candidat | Seul dir / inv | dir 0–4 | dir 5–9 | inv 0–4 | inv 5–9 | Règle |
|---|---|---:|---:|---:|---:|---|
| Base (installé) | 0.2701 / 0.2237 (R) | 0.2592 | 0.2581 | 0.2063 | 0.2054 | |
| R, contrôle | 0.2701 / 0.2237 | = | = | = | = | |
| D3 | 0.2685 / 0.2241 | 0.2585 | 0.2574 | 0.2061 | 0.2052 | oui (moitié inv +0.0001) |
| D31 | 0.2731 / 0.2262 | 0.2605 | 0.2593 | 0.2067 | 0.2058 | non |
| D3 + orbite | 0.2680 / 0.2235 | 0.2584 | 0.2573 | 0.2058 | 0.2050 | oui |
| D3 + orbite, 10 sacs | 0.2710 / 0.2206 | 0.2595 | 0.2583 | 0.2046 | 0.2038 | non (directe monte) |
| **D3L + orbite** | **0.2678 / 0.2229** | **0.2583** | **0.2572** | **0.2054** | **0.2046** | **oui, toutes moitiés en baisse** |
| D3 + orbite + regimes + historique | 0.2794 / 0.2199 | 0.2610 | 0.2602 | 0.2047 | 0.2040 | non (moitié +0.007) |

Le contrôle redonne exactement R. Les variantes qui gagnent beaucoup en inverse (10 sacs, régimes)
perdent en directe : pas de gain qui tienne dans les deux sens au-delà de D3L + orbite.

### GRU de plus grande capacité (Kaggle T4)

GRU mixte installé avec taille 192 et 1200 pas, graines 0–4 et 5–9 (environ 53 min d'entraînement
par session). GRU seul : 0.0799 / 0.0850 (moyenne 0.0824), contre 0.0812 pour le GRU installé.
Sur 2551, avec la part GRU mixte (0.35) remplacée : directe 0.2592 → 0.2557 et 0.2581 → 0.2566,
mais inverse 0.2063 → 0.2078 et 0.2054 → 0.2075. Refusé.

## Décision

D3L + orbite devient le boosting de 2551 : `SPECIALISTES["NPWD2551"]` dans `dio/ensemble.py`, avec
le boosting propre dans `specialiste/gbm_NPWD2551/`. Les 30 autres colonnes ne changent pas.
Il y a un léger optimisme de sélection : 6 variantes ont été comparées sur la validation.

## Validation de bout en bout et installation

Le boosting propre a été réentraîné sur Kaggle par `train.py --modele gbm_sacs` (réglages de
`SPECIALISTES`), une fois par direction (`runs/s51_spec`) puis sur annee1 + annee2
(`runs/final_gbm_2551`). Les runs `runs/s51_val_g04|g59` sont des copies de `fin_val_*_sp`
complétées par `specialiste/gbm_NPWD2551/`, puis repassées par `valider.py --sans-entrainement`.

| | Avant (`fin_val_*_sp`) | Après (`s51_val_*`) | 2551 avant → après |
|---|---:|---:|---|
| directe 0–4 | 0.075138 | 0.075046 | 0.2592 → 0.2583 |
| inverse 0–4 | 0.078541 | 0.078474 | 0.2063 → 0.2054 |
| directe 5–9 | 0.075170 | 0.075078 | 0.2581 → 0.2572 |
| inverse 5–9 | 0.078408 | 0.078343 | 0.2054 → 0.2046 |
| Moyenne des 4 | 0.076814 | **0.076735** | |

Dans les 4 cas, seule la colonne NPWD2551 diffère ; les 32 autres sont identiques au bit près.
Le gain reproduit exactement le calcul de la section précédente. Installé :
`poids/specialiste/gbm_NPWD2551/gbm_sacs.pkl.xz` et `poids/specialiste/specialistes.json`.

## Bilan et limites

- NPWD2551 baisse d'environ 0.0009 dans chaque cas (−0.35 % en directe, −0.43 % en inverse).
  La moyenne globale baisse de 0.00008.
- Le reste de l'erreur est en grande partie hors d'atteinte. Le bruit horaire de thermostat vaut
  0.23, et même un niveau hebdomadaire parfait laisserait 0.230 / 0.173. Le résidu actuel
  (0.258 / 0.205) n'est qu'à 0.03 de ce plancher, et ce plancher suppose un oracle.
- Pistes écartées dans ce programme : recombinaison des composants actuels, demi-vie du recalage,
  empilement sur les autres circuits, boosting à 10 sacs, arbres plus grands, ajouts regimes et
  historique, GRU de taille 192. Chacune gagne dans une direction et perd dans l'autre, ou perd
  dans les deux.

# Programme 2 (13 novembre 2026) : sous-estimation prolongée de NPWD2551

Demande : corriger la sous-estimation prolongée de NPWD2551 vue au diagnostic, avec un spécialiste si
nécessaire, sans changer les 32 autres sorties. Référence et repli : le livré (0.076735,
`runs/rc0_livre_g04|g59`). Tous les entraînements sur Kaggle (compte `zendarc148`).

## Protocole (fixé avant les mesures)

- Trois validations internes : directe (annee1 → annee2), inverse (annee2 → annee1), chronologique (annee1 +
  première moitié d'annee2 → seconde moitié). Directe et inverse sont notées pour les graines 0–4 et 5–9 de la
  base : 5 cas. Base chrono = même structure que le livré, 0.5 × (0.7 GRU mixte + 0.3 G2) + 0.5 × boosting
  spécialiste D3L + orbite réentraîné en chrono (`runs/n2_base_chrono`).
- Mélanges (`outils/bilan_2551.py`) : base + 1 à 3 candidats, poids de somme 1 au pas de 0.1 ; les poids d'une
  validation sont choisis sur les deux autres ; poids communs (trois validations) pour l'installation.
- Installation si : RMSE de 2551 en baisse dans les 5 cas avec les poids transférés, aucune moitié de période en
  hausse de plus de 0.002, gain moyen ≥ 0.001, et 32 autres colonnes identiques au bit près.

## Diagnostic de la base (`python outils/bilan_2551.py --nom n2_diag --validations directe inverse`)

| | RMSE | MAE | Biais | Moitiés | Niveau hebdo / journalier (part de la MSE) | Épisodes de sous-estimation (≥ 3 j < −0.1) |
|---|---:|---:|---:|---|---|---|
| directe 0–4 | 0.2583 | 0.1783 | +0.030 | 0.221 / 0.291 | 23 % / 33 % | 8 épisodes, 66 j, 30 % de la MSE |
| directe 5–9 | 0.2572 | 0.1778 | +0.028 | 0.220 / 0.290 | 22 % / 33 % | 9 épisodes, 69 j, 31 % |
| inverse 0–4 | 0.2054 | 0.1291 | +0.016 | 0.172 / 0.234 | 30 % / 39 % | 4 épisodes, 63 j, 36 % |
| inverse 5–9 | 0.2046 | 0.1294 | +0.017 | 0.171 / 0.234 | 29 % / 38 % | 4 épisodes, 62 j, 36 % |

- Par régime (directe 0–4) : la saison d'éclipses porte 74 % de la MSE (RMSE 0.318 contre 0.183 hors saison) ;
  semaines vraies basses : biais +0.068 (surestimation) ; semaines hautes : biais ≈ 0 mais grands écarts.
- **Corrélation observée (lecture des blocs de 4 jours vrai / prédit / ombre ; requalifiée au programme 3, la cause
  n'est pas établie)** : le régime haut du réchauffeur
  (≈ 0.45–0.8) ne suit pas exactement la saison d'éclipses, alors que le modèle n'y monte qu'avec les minutes
  d'ombre. Les grands épisodes sont tous à des bascules :
  - allumage avant la saison : directe jour 14871 (4–8 j avant les premières ombres, −0.32 à −0.41),
    15459 (8 j, −0.36) ; inverse 14536 (20 j avant, −0.33 à −0.42) ;
  - maintien après la saison : directe 15027 (12 j, −0.37), inverse 14720 (12 j, −0.37 à −0.42) ;
  - extinction avant la fin de la saison (surestimation symétrique) : directe 14991 (+0.25 pendant 32 j),
    inverse 14368 (+0.34 pendant 3 semaines), à une durée d'éclipse ltdata d'environ 27 min les deux fois ;
  - fin de saison 15291–15315 (directe) : réchauffeur maintenu à 0.45 pendant que le modèle descend avec l'ombre.
- Hypothèse (à tester, non démontrée) : il manque au modèle des fenêtres **futures** longues sur l'ombre et les
  éclipses (préchauffage avant une saison connue d'avance) ; les variables actuelles sont passées (histoires,
  ema, orbite) ou limitées à ±168 h sur le planning. Bloc `echelles` de `baselines/gbm_sacs.py`.
- Recherche d'une commande marquant les 14 bascules (5 en annee1, 9 en annee2) : aucune famille commune évidente
  dans ±36 h ; non poursuivie (trop peu de bascules pour une liste fiable).

## Commandes ATTT305C/O/P et ATTT306C/P

| | annee1 | annee2 avant la coupure chrono | annee2 après |
|---|---:|---:|---:|
| ATTT305C | 0 | 691 | 361 |
| ATTT305O | 0 | 0 | 452 |
| ATTT306C | 0 | 40 | 0 |
| ATTT305P, ATTT306P | 0 | 0 | 0 |

Dans annee2, 2551 vaut 0.02 en moyenne dans l'heure d'un ATTT306C (0.27 ailleurs), 0.10 après un ATTT305O,
0.17 après un ATTT305C (puis plus haut au-delà de 6 h). Comme variable (bloc `attt`), le signal n'est
apprenable qu'en inverse et en chrono (ATTT305C), et 305P/306P n'apparaissent dans aucune donnée.

Règle « 2551 = 0 pendant k h après la commande », évaluée à part sur la base (`--zero`). Seule la directe a des
occurrences, et la règle a été trouvée en regardant annee2 (mesure contaminée, favorable à la règle) :

| Règle | k = 1 h | 3 h | 6 h |
|---|---|---|---|
| ATTT306C (directe 0–4) | 0.2583 → 0.2583 | → 0.2584 | → 0.2585 |
| ATTT305O (directe 0–4) | 0.2583 → 0.2588 | → 0.2598 | → 0.2642 |

**Rejetée** : aucun gain même dans le cas le plus favorable (la base prédit déjà presque 0 après ATTT306C ;
après ATTT305O, 2551 ne s'arrête pas toujours). Inverse et chrono : aucune occurrence, donc aucune généralisation
démontrée.

## Incident Kaggle

Le compte `zendarc148` n'a pas d'accès internet : `pip install` échouait et les 5 premières sessions sont
mortes au démarrage. `outils/kaggle_distant.py` garde désormais les paquets préinstallés de Kaggle si pip échoue
(versions affichées dans le journal de session). Les prédictions rapatriées sont des CSV, indépendantes de ces
versions.

Une fois `zendarc148` vérifié, les sessions ont reçu un T4 (`torch 2.15.0+cu132`, Tesla T4, internet,
scikit-learn 1.9.1 comme en local). Le premier XGB20 (`n2_xgb20`) avait tourné sur CPU avec scikit-learn 1.6.1,
avant l'interdiction des sessions CPU ; ses prédictions sont gardées (XGBoost ne dépend pas de scikit-learn).
Garde-fou ajouté : une session demandée sur GPU et servie sans GPU s'arrête sans calculer
(`outils/kaggle_distant.py`, champ `gpu` de la commande).

## Candidats (Kaggle T4, `outils/specialistes.py entrainer --circuits NPWD2551 --directions directe inverse chrono`)

Toutes les variantes partent des variables du spécialiste installé (`GBM_DEFAUT` + `orbite`). Base chrono :
`n2_base_chrono` (D3L + orbite en chrono, 1135 s). Temps par direction sur T4 : XGB20 107–134 s ; HGB D3L avec
`echelles` 850–1616 s ; niveau lent 343–551 s ; composante rapide 923–1665 s.

| Run | Modèle | Variables | Justification |
|---|---|---|---|
| `n2_xgb20` | 20 XGBoost diversifiés (4 par sac × 5 sacs), 70 % des colonnes, profondeur {3,5,7,9}, arbres {200,400,600}, L2 {1,2}, taux 0.03 | actuelles | piste demandée |
| `n2_xgb20_ea` | idem | + `echelles` + `attt` | piste demandée + diagnostic |
| `n2_hgb_ea` | HGB D3L (= spécialiste installé) | + `echelles` + `attt` | sépare l'effet des variables de celui du modèle |
| `n2_hgb_e` | HGB D3L | + `echelles` | sépare `echelles` de `attt` |
| `n2_lent` + `n2_rapide` | niveau : HGB D3L sur la cible moyennée 168 h, recalé ; rapide : HGB D3L sur y − moyenne 168 h, sans recalage | + `echelles` + `attt` | l'erreur restante est surtout de niveau |

### Candidats seuls (colonne 2551 remplacée ; RMSE / biais)

| | directe 0–4 | directe 5–9 | inverse 0–4 | inverse 5–9 | chrono | MAE dir / inv / chr |
|---|---:|---:|---:|---:|---:|---|
| Base (livré) | 0.2583 / +0.030 | 0.2572 / +0.028 | 0.2054 / +0.016 | 0.2046 / +0.017 | 0.2557 / +0.000 | 0.178 / 0.129 / 0.186 |
| Spécialiste installé seul (D3L + orbite) | 0.2678 / +0.025 | = | 0.2229 / +0.011 | = | 0.2641 / +0.011 | 0.188 / 0.132 / 0.196 |
| XGB20 | 0.2852 / +0.066 | = | 0.2182 / −0.006 | = | 0.2814 / +0.072 | 0.210 / 0.134 / 0.220 |
| XGB20 + echelles + attt | 0.2777 / +0.087 | = | 0.2331 / +0.040 | = | 0.2871 / +0.076 | 0.206 / 0.161 / 0.221 |
| HGB + echelles + attt | 0.2609 / +0.041 | = | 0.2281 / +0.045 | = | 0.2742 / +0.032 | 0.191 / 0.155 / 0.206 |
| HGB + echelles | 0.2609 / +0.041 | = | 0.2258 / +0.047 | = | 0.2719 / +0.024 | |
| Lent + rapide | 0.2663 / +0.043 | = | 0.2287 / +0.049 | = | 0.2705 / −0.009 | 0.202 / 0.168 / 0.199 |

(Les candidats ne dépendent pas des graines du GRU : mêmes valeurs pour 0–4 et 5–9.)

### Mélanges, poids transférés (`python outils/bilan_2551.py --nom n2_B --candidats ...`)

Pour chaque candidat et chaque combinaison de 2 ou 3 candidats, le meilleur poids choisi sur deux validations
donne 0 en directe et 0.1–0.2 en inverse ou en chrono, où la RMSE **monte** : par exemple HGB + echelles + attt,
inverse 0.2054 → 0.2074, chrono 0.2557 → 0.2563 ; lent + rapide, inverse 0.2054 → 0.2069. Gain moyen sur les
5 cas : 0 (XGB20, XGB20 + variables), −0.0006 (lent + rapide), −0.0009 (HGB + variables). Aucun n'est accepté.

### Remplacement de la part boosting de 2551 (forme de l'installation de D3L), sans poids appris

`base + 0.5 × (candidat − spécialiste installé)` :

| Candidat | directe 0–4 | directe 5–9 | inverse 0–4 | inverse 5–9 | chrono | Gain moyen |
|---|---:|---:|---:|---:|---:|---:|
| HGB + echelles | 0.2583 → **0.2530** | 0.2572 → **0.2521** | 0.2054 → 0.2065 | 0.2046 → 0.2059 | 0.2557 → 0.2597 | +0.0008 |
| HGB + echelles + attt | → 0.2530 | → 0.2521 | → 0.2077 | → 0.2071 | → 0.2602 | +0.0003 |
| Lent + rapide | → 0.2569 | → 0.2559 | → 0.2089 | → 0.2086 | → 0.2602 | −0.0019 |
| XGB20 | → 0.2659 | → 0.2645 | → 0.2041 | → 0.2033 | → 0.2599 | −0.0033 |
| XGB20 + echelles + attt | → 0.2608 | → 0.2595 | → 0.2093 | → 0.2086 | → 0.2625 | −0.0039 |

Aucun ne baisse dans les 5 cas : refusés.

### Épisodes de sous-estimation prolongée (part de la MSE, candidat seul)

| | directe | inverse | chrono |
|---|---:|---:|---:|
| Spécialiste installé | 32 % | 33 % | 23 % |
| HGB + echelles | 28 % | 23 % | 16 % |
| HGB + echelles + attt | 28 % | 22 % | 16 % |

## Montée de l'ensemble XGBoost

Non faite : la règle fixée (passer à 50, 100 puis 200 modèles seulement si XGB20 améliore 2551 en mélange dans
au moins 4 cas sur 5) n'est pas remplie. XGB20 seul est moins bon que le boosting HGB D3L dans les trois
validations sauf l'inverse, et il surestime fortement en directe et en chrono (biais +0.07) ; plus de modèles
réduirait la variance, pas ce biais.

## Incident corrigé : composante rapide bornée à 0

`gbm_sacs.predire` bornait toute prédiction à 0, y compris la composante rapide (y − moyenne), qui doit pouvoir
être négative. Premier résultat faussé : lent + rapide 0.2864 / 0.2511 / 0.2914, biais +0.08. Correction : pas de
borne quand le paquet est `rapide` (la somme est bornée ensuite). Les poids de `n2_rapide` ont été rapatriés et
les prédictions refaites en local, sans réentraînement ; les chiffres ci-dessus sont les corrigés.

## Décision

**Rien n'est installé ; le modèle livré reste (0.076735), `poids/` est inchangé.** Aucun candidat ni mélange ne
remplit le critère fixé avant les mesures. Les 32 autres colonnes n'ont donc pas eu à être revérifiées.

## Faits établis et hypothèses (requalifiés au programme 3 : les liens avec les éclipses sont des corrélations)

Démontré sur les validations internes :
- La sous-estimation prolongée existe aussi en directe, inverse et chrono : épisodes de 1 à 3 semaines à
  −0.3 / −0.4, 21 à 36 % de la MSE de 2551.
- Ces épisodes sont aux bascules du réchauffeur (corrélation, pas cause), qui ne coïncident pas avec la saison d'éclipses : allumage
  jusqu'à 20 jours avant les premières ombres, maintien jusqu'à 12 jours après les dernières, extinction parfois
  avant la fin (surestimation symétrique). Le modèle suit les minutes d'ombre.
- Les fenêtres futures longues de l'ombre et des éclipses (`echelles`) réduisent ces épisodes (inverse 33 % →
  23 % de la MSE, chrono 23 % → 16 %), mais créent autant de surestimation ailleurs (biais +0.02 à +0.05) : la
  RMSE baisse en directe et monte en inverse et en chrono.
- Les commandes ATTT, comme variables, dégradent l'inverse et la chrono (+0.0023 chacune contre `echelles`
  seul) ; comme règle « zéro », elles n'apportent rien, même sur la période où la règle a été trouvée.
- L'ensemble XGBoost demandé et le modèle lent + rapide sont moins bons que le boosting HGB installé.

Hypothèses (non démontrées) :
- La date des bascules est une décision d'exploitation : deux années donnent 14 bascules, trop peu pour
  apprendre un décalage qui se généralise ; ce qu'apprend une année (préchauffage long en annee2) ne vaut pas
  pour l'autre.
- Une commande de bascule propre au réchauffeur existe peut-être dans le planning, mais aucune famille commune
  n'apparaît dans ±36 h des 14 bascules.

## Reproduction

```bash
uv run python outils/bilan_2551.py --nom n2_diag --validations directe inverse
uv run python outils/bilan_2551.py --nom n2_zero_di --zero --validations directe inverse
uv run python outils/lancer_kaggle.py --fente 1 -- outils/specialistes.py entrainer --nom n2_hgb_e \
  --circuits NPWD2551 --directions directe inverse chrono --options '{... GBM_DEFAUT, "config": "D3L", "ajouts": [... "orbite", "echelles"]}'
uv run python outils/bilan_2551.py --nom n2_B --candidats spec=n2_spec hgbea=n2_hgb_ea lr=n2_lent+n2_rapide --taille 2
```

Runs : `runs/n2_*` (prédictions, `bilan.json`). `runs/n2_spec` = prédictions du spécialiste installé
(`s51_gbm_D3Lo` en directe et inverse, `n2_base_chrono` en chrono). Code gardé : config `XGBn`, ajouts `echelles`
et `attt`, option `rapide` (`baselines/gbm_sacs.py`), directions dans `outils/specialistes.py`,
`outils/bilan_2551.py`. Tests : `tests.test_gbm_sacs` (3 tests ajoutés), `test_ensemble`, `test_caracteristiques`,
`test_recette` : 36 OK.

# Programme 3 (13–14 novembre 2026) : recherche guidée par les erreurs restantes

Demande : améliorer NPWD2551 de façon robuste, sans changer les 32 autres sorties, en remettant d'abord en question
le diagnostic et les essais précédents. Référence et repli : le livré (0.076735, `runs/rc0_livre_g04|g59`).
Entraînements sur Kaggle (`zendarc148`, T4) ; en local, seulement des calculs de moins de 2 min.

## Protocole, budget et critères d'arrêt (fixés avant les entraînements)

- **Périodes.** A1 = annee1 (test de l'inverse) ; A2a / A2b = moitiés d'annee2. La directe teste A2a + A2b, la
  chrono teste A2b : directe et chrono ne sont pas deux périodes indépendantes. Les groupes de graines g04 / g59
  de la base sont des répétitions, pas des périodes.
- **Sélection à deux niveaux.** (1) Variables, hyperparamètres, lissage et nombre d'arbres : validation croisée
  temporelle **interne à l'année d'entraînement** de chaque direction (5 blocs contigus, 168 h d'embargo de chaque
  côté, prédictions hors bloc), dans `outils/recherche_2551.py`. Les validations externes ne servent pas à ces
  choix. (2) Poids de mélange avec le livré, choisis sur des périodes disjointes : inverse ← directe, directe ←
  inverse, chrono ← inverse + A2a de la directe (`SOURCES` de `outils/bilan_2551.py`). Le programme 2 choisissait
  les poids de la directe sur l'inverse **et la chrono**, dont le test (A2b) est inclus dans celui de la directe.
- **Au plus 4 candidats** atteignent les validations externes (un par hypothèse, fixé en interne) ; aucune
  variante n'est ajoutée après les avoir vues.
- **Formes comparées** : livré, spécialiste S seul, candidat seul, remplacement de la part boosting
  (`base + 0.5 × (c − S)`, sans poids appris), mélange livré / candidat aux poids transférés.
- **Règle d'installation** (plus stricte, les validations ayant déjà servi à une quinzaine de choix) : RMSE en
  baisse dans les 5 cas ; gain ≥ 0.0005 sur chacune de A1, A2, A2b ; gain moyen ≥ 0.0015 ; aucune moitié en hausse
  > 0.002 ; MAE en baisse ou égale par période ; |biais| pas plus de 0.01 au-dessus ; RMSE dans les fenêtres de
  ±14 jours autour des bascules vraies pas plus de 0.005 au-dessus ; 32 autres colonnes identiques au bit près.
- **Budget** : 6 sessions Kaggle d'expérience au plus (2 en parallèle), 1 d'installation, une relance par session
  après un bogue.
- **Arrêt** : un candidat qui ne bat pas S d'au moins 1 % en validation interne dans les trois directions ne va pas
  aux validations externes ; un échec de transfert clôt sa famille ; membres XGBoost 1 → 5 → 10 seulement si l'OOB
  gagne au moins 0.5 % puis 0.2 %.
- Comparateur interne : prédictions hors sac de S lui-même (`s51_gbm_D3Lo`, `n2_base_chrono`), sur les mêmes 5 blocs
  mais **sans embargo** : la comparaison est défavorable aux candidats (prudente).

## Étape 0 : remise en question (calculs locaux)

### D1 — Décomposition de l'erreur (`bilan_2551.py --evaluer`, `runs/p3_d1`)

Jour « haut » si sa moyenne vraie dépasse 0.15 (creux de l'histogramme bimodal : 157 + 122 jours sous 0.1, 13 entre
0.1 et 0.15 en annee2). MSE = part journalière, découpée en *niveau* (régime du jour bien prédit), *date de
bascule* (régime faux à ±14 jours d'une bascule durable vraie) et *régime faux isolé*, + part *horaire* (heure −
moyenne du jour). Les parts somment exactement à la MSE (test `tests/test_regime_2551.py`).

| Part de la MSE (%) | directe | inverse | chrono |
|---|---|---|---|
| Livré : niveau / bascule / régime faux / horaire | 14 / 15 / 4 / 67 | 17 / 15 / 7 / 61 | 19 / 3 / 3 / 76 |
| S seul | 13 / 18 / 4 / 65 | 16 / 15 / 8 / 61 | 19 / 5 / 3 / 73 |
| RMSE livré dans les fenêtres de bascule / ailleurs | 0.275 / 0.248 | 0.307 / 0.178 | 0.206 / 0.284 |

- La **variation horaire** porte 61 à 76 % de la MSE du livré ; la date des bascules 3 à 15 % ; le régime faux isolé
  3 à 7 % ; l'erreur de niveau dans le bon régime 14 à 19 %. Le programme 2 visait surtout les épisodes de
  sous-estimation, soit moins d'un tiers de l'erreur.
- Hors bloc dans l'année d'entraînement, S garde 39 à 50 % d'erreur journalière et 50 à 61 % d'horaire : la part
  horaire n'est pas un défaut de transfert entre années.
- Le résidu horaire de S est corrélé à 7 h et 14 h (0.25 / 0.24 en annee1, 0.17 / 0.18 en annee2) : une structure
  liée à l'orbite reste. Mais même un oracle qui connaîtrait la moyenne du résidu par heure depuis le péricentre,
  sur 14 jours autour (sans l'orbite notée), n'en explique que 17 % (annee1) et 14 % (annee2). L'essentiel de la
  part horaire est le cycle du thermostat, non synchronisé à l'orbite : **limite démontrée** pour un modèle sans
  consommation récente.

### D2 — Les entrées distinguent-elles les régimes ? (classement journalier croisé entre années)

Régression logistique et arbre de profondeur 3 sur des agrégats journaliers, appris sur une année, testés sur
l'autre. Commandes vues comme **événements** (comptes journaliers par famille, ≥ 50) et comme **états persistants**
(`gbm_sacs.variables_etats` : dernier code de chaque famille à plusieurs variantes, catégorie « inconnu » pour un
code jamais vu, NaN avant la première commande et pendant une lacune du journal dmop de plus de 48 h ; annee2 a
des lacunes jusqu'à 839 h).

| Exactitude équilibrée (logistique) | a1 → a2 : tout / ±14 j des bascules / ailleurs | a2 → a1 |
|---|---|---|
| Contexte (ombre, pénombre, éclipse ltdata, distance, saaf, ftl, ombre passée et future 7 / 30 j) | 0.84 / 0.69 / 0.91 | 0.89 / 0.64 / 0.95 |
| Commandes, événements | 0.57 / 0.53 / 0.59 | 0.65 / 0.41 / 0.71 |
| Commandes, états persistants | 0.51 / 0.44 / 0.54 | 0.69 / 0.54 / 0.72 |
| Contexte + événements + états | 0.73 / 0.54 / 0.81 | 0.85 / 0.67 / 0.90 |

- Loin des bascules, le contexte suffit (91–95 %). Près des bascules, il est à 64–69 % ; sur ±7 jours, 12 des 14
  bascules durables n'ont que 50 à 57 % de jours bien classés (date déplacée d'une semaine ou plus) ; seules
  14284 (86 %) et 14368 (71 %), en annee1, sont bien placées.
- Les commandes, en événements comme en états, ne séparent pas les régimes d'une année à l'autre, et les ajouter
  au contexte **dégrade** le transfert (0.84 → 0.73). Le test d'états du programme 2 (±36 h) et celui-ci (états
  persistants, lacunes explicites) concluent pareil.
- Plusieurs régimes hauts n'ont aucune ombre (annee2, jours 15370–15415, Soleil le plus loin ; annee1, allumage 23
  jours avant la saison) : la relation avec les éclipses est une corrélation partielle, pas une cause établie.

### D3 — Pourquoi `echelles` déplaçait l'erreur

Jours de pré-saison (aucune ombre, ombre dans les 30 jours suivants), biais journalier S → `n2_hgb_e` :

| | directe | inverse | chrono |
|---|---|---|---|
| Pré-saison, vrai haut (jours) | 13 : −0.348 → −0.092 | 23 : −0.366 → −0.267 | 10 : −0.198 → −0.065 |
| Pré-saison, vrai bas (jours) | 77 : +0.040 → +0.143 | 37 : +0.003 → +0.044 | 24 : +0.114 → +0.195 |
| 60 à 30 derniers jours du test | +0.086 → +0.181 | +0.018 → +0.063 | +0.110 → +0.158 |

Les fenêtres futures font monter la prédiction **avant chaque saison d'ombre**, alors que le préchauffage n'a eu
lieu que sur 13 des 90 jours de pré-saison d'annee2 (23 sur 60 en annee1) : le modèle apprend une propension
moyenne, ce qui corrige les rares préchauffages et surestime les autres. S'y ajoute un effet de bord : à la fin du
test, les moyennes futures sur 336–720 h sont tronquées et le biais monte. Aucune entrée vue en D2 ne dit quelles
saisons seront préchauffées ; d'où la règle de ne pas ajouter d'autres fenêtres de ce genre, et l'essai en K3 de
`echelles` passé seul et de `bords_nan` (fenêtres incomplètes mises à NaN).

### D4 — Transformations, recalage et bornes

- Composante rapide : la borne à 0 corrigée au programme 2 est la seule anomalie trouvée. Somme lent + rapide
  bornée une fois ; `lissage_cible` garde NaN là où la cible manque ; cibles manquantes retirées, jamais imputées.
- Recalage sur `power_init` : décalage constant estimé sur 7 jours, demi-vie 60 jours. Les 30 premiers jours de S
  ont une RMSE de 0.033 (directe), 0.076 (inverse) ; le recalage ne pèse que sur le début. Demi-vies 360–5760 h
  déjà sans effet (programme 1). Le modèle de régime n'utilise pas ce recalage : `power_init` y fixe l'état initial.
- Sacs de `GBM_DEFAUT` sans embargo : leurs prédictions hors sac sont optimistes. La validation interne de ce
  programme met 168 h d'embargo.
- XGB20 : biais +0.066 en directe et +0.072 en chrono, mais −0.006 en inverse ; la décomposition montre surtout
  plus d'erreur de niveau (19 % contre 14 %) et de date (17 % contre 15 %). Réglage figé sans arrêt anticipé
  (200–600 arbres, profondeur jusqu'à 9) : K1 vérifie si un réglage choisi en interne corrige ce biais.

### D5 — Révision du programme 2

« Cause démontrée » et « Causes démontrées » y sont requalifiées en corrélations observées. Les poids transférés
du programme 2 utilisaient la chrono pour la directe (périodes recouvrantes) ; les mélanges sont refaits ici avec
des périodes disjointes. Sous la nouvelle règle, `n2_hgb_e` (gain moyen en remplacement +0.0008) et XGB20 restent
refusés (MAE en hausse sur les trois périodes, biais en hausse en inverse).

## K1 — XGBoost réglé en validation interne (`recherche_2551.py modele`, `runs/p3_xgb`, Kaggle T4, 23 min)

- **Hypothèse** : l'échec de XGB20 (programme 2) vient de son réglage figé (taux 0.03, profondeur jusqu'à 9,
  200–600 arbres sans arrêt anticipé), pas de XGBoost.
- **Changement** (variables de S figées, 807 à 934 colonnes) : 14 réglages, taux {0.01, 0.03, 0.1} × profondeur
  {4, 6, 8} × `min_child_weight` {20, 100, 500}, sous-échantillonnage 0.8 des lignes et des colonnes, arbres par
  arrêt anticipé (200 itérations sans gain) sur le bloc écarté. Pour le réglage retenu : 5 graines, et 4 sacs
  temporels contre 1 modèle.

| | directe | inverse | chrono |
|---|---|---|---|
| Réglage retenu (taux, profondeur, mcw) ; arbres | 0.03, 6, 500 ; 1223 | 0.03, 6, 100 ; 853 | 0.1, 4, 100 ; 274 |
| Meilleurs arbres par pli | 177–3971 | 179–2701 | 8–649 |
| OOB XGB / OOB S (sans embargo) | 0.2384 / 0.2405 (−0.9 %) | 0.2338 / 0.2422 (−3.5 %) | 0.2552 / 0.2653 (−3.8 %) |
| 5 graines (arbres fixés) | 0.2395 | 0.2354 | 0.2637 |
| 4 sacs temporels | 0.2484 | 0.2407 | 0.2678 |

- **Décision** : critère d'arrêt 1 non rempli (directe −0.9 % < 1 %) ; pas de validation externe pour la décision.
  Les graines n'apportent rien (la règle 1 → 5 n'est pas remplie, donc pas de montée en membres) ; les sacs
  temporels dégradent l'OOB dans les trois directions (chaque sac perd un quart des données).
- Le réglage retenu change d'une direction à l'autre et le meilleur nombre d'arbres varie d'un facteur 10 à 600
  entre plis : le choix est instable. L'OOB à l'arrêt anticipé est optimiste (le pli écarté choisit son propre
  arrêt) : avec des arbres fixés et 5 graines, les gains tombent à 0.4 %, 2.8 % et 0.6 %.
- **Information, hors décision** (candidat déjà écarté, `runs/p3_k1_info`) : en externe, XGB seul fait 0.2899 /
  0.2246 / 0.2739 contre 0.2678 / 0.2229 / 0.2641 pour S, avec un biais de +0.059 en directe. Les gains internes
  ne se transfèrent pas d'une année à l'autre.
- **Ce que cela invalide** : l'idée qu'un meilleur réglage de XGBoost suffirait. Le biais de la directe n'est pas
  un artefact du réglage figé : il reste après sélection interne, et la décomposition l'attribue au niveau et
  aux dates de bascule (17 % et 21 % de la MSE, contre 13 % et 18 % pour S). XGBoost ajuste plus fort que HGB D3L
  ce qui est propre à l'année d'entraînement.
- **Ce qui reste ouvert** : la validation croisée interne, nécessaire pour ne pas choisir sur les validations, ne
  prédit pas le transfert entre années pour 2551. Cela vaut aussi pour les sessions suivantes : un gain interne y
  est nécessaire, pas suffisant.

## K2 — Deux régimes avec mémoire du régime (`recherche_2551.py regime`, `baselines/regime.py`, `runs/p3_regime`, 19 min)

- **Hypothèse** : séparer la probabilité du régime haut et la consommation dans chaque régime réduit l'erreur de
  niveau ; la persistance du régime (chaîne de Markov collante sur les jours, état initial donné par
  `power_init`) supprime les régimes faux isolés.
- **Changement** (variables de S figées) : classifieur HGB D3 de P(jour haut), régressions HGB D3 dans chaque
  régime, prédiction π m_haut + (1 − π) m_bas, sans recalage. Renouvellement du régime choisi en interne parmi
  {1 (aucune persistance), 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01} ; dans chaque bloc écarté, ses 168 premières
  heures jouent le rôle de `power_init` et ne sont pas notées.

| OOB, mêmes lignes | directe | inverse | chrono |
|---|---|---|---|
| Renouvellement retenu | 1 (aucune persistance) | 1 | 0.01 (courbe plate : 0.2688–0.2701) |
| Deux régimes / S | 0.2250 / 0.2450 (−8.2 %) | 0.2425 / 0.2451 (−1.1 %) | 0.2688 / 0.2647 (+1.5 %) |
| Avec le régime vrai (oracle) | 0.1721 | 0.2273 | 0.1892 |
| Exactitude horaire du classifieur | 0.87 | 0.76 | 0.84 |

- **Décision** : critère d'arrêt 1 non rempli (chrono en hausse) : pas de validation externe pour la décision.
- **Ce que cela invalide** : la mémoire du régime, telle que testée. La persistance n'améliore jamais nettement
  l'OOB et le dégrade en directe et en inverse : les erreurs du classifieur durent elles-mêmes des semaines autour
  des bascules (D1 : régime faux isolé 3 à 7 % seulement), et le lissage les étale au lieu de les corriger.
- L'oracle montre que connaître le régime ferait gagner 7 à 30 % de RMSE dans l'année : la part journalière
  serait exploitable si l'information existait dans les entrées. D2 montre qu'elle n'y est pas aux bascules.
- **Information, hors décision** (`runs/p3_k2_info`) : en externe, deux régimes seuls fait 0.2882 / 0.2224 /
  0.2918 (S : 0.2678 / 0.2229 / 0.2641), biais −0.046 en directe. Le meilleur cas interne (directe, −8 %) est le
  pire en externe : le classifieur apprend la part de régime haut d'annee1 (0.41), alors qu'annee2 en a 0.56.

## K4 — Correction de S apprise sur ses résidus hors entraînement (dans la session K2, `runs/p3_regime_residu`)

- **Hypothèse** : les résidus hors sac de S sur l'année d'entraînement contiennent une erreur systématique
  apprenable (par saison, par régime) qui se transfère.
- **Changement** : cible y − S_oob (prédictions hors sac de S lui-même), variables de S + S_oob + π du modèle de
  régime (NaN sur les heures d'initialisation), HGB très régularisé (7 feuilles, 500 exemples par feuille, L2 10),
  sans réglage ; prédiction S_test + correction.

| OOB | directe | inverse | chrono |
|---|---|---|---|
| S / S corrigé | 0.2405 / 0.2657 | 0.2422 / 0.2741 | 0.2653 / 0.3303 |
| Corrélation correction / vrai résidu (hors bloc) | +0.01 | −0.16 | −0.41 |

- **Décision** : rejeté au critère interne ; la correction dégrade les trois directions.
- **Ce que cela invalide** : l'erreur de S n'est pas stationnaire d'un bloc de 2,5 mois à l'autre. La correction
  apprise sur les autres blocs a le signe opposé au vrai résidu du bloc écarté (−0.41 en chrono) : ce qui était
  sous-estimé dans un bloc (une saison préchauffée) est surestimé dans un autre (une saison sans préchauffage).
  C'est la même cause que l'effet d'`echelles` (D3).
- **Information, hors décision** : en externe, remplacement 0.2623 / 0.2010 / 0.2657 contre le livré 0.2583 /
  0.2054 / 0.2557 : l'inverse gagne 0.0044, la directe perd 0.0040 et la chrono 0.0100. Pas de gain robuste.

## K3 — Effet des variables à modèle figé (HGB D3, `recherche_2551.py variables`, `runs/p3_var_*`)

- **Hypothèse** : sous un même modèle, les états de commandes (`etats`), `echelles` sans fenêtres futures, ou
  `echelles` avec fenêtres incomplètes mises à NaN (`bords_nan`, aussi sur les EMA lentes d'`orbite`) améliorent
  2551. K1 n'ayant pas de gagnant, le modèle figé est HGB D3 (plan). `sans_horloge` a été retiré de la liste
  pour tenir le budget : aucun diagnostic ne le désignait.
- **Changement** : un jeu à la fois en plus des variables de S ; `base` = variables de S sous HGB D3 (contrôle).

| OOB (directe) | base | + etats | + echelles passé seul | + echelles, bords_nan |
|---|---|---|---|---|
| RMSE | 0.2453 | 0.2086 | 0.2617 | 0.2660 |

- `echelles` passé seul et avec `bords_nan` dégradent la directe de 7 à 8 % : rejetés au critère 1. L'effet de bord
  vu en D3 existe, mais le retirer ne compense pas la propension moyenne de préchauffage que ces fenêtres font
  apprendre : ces deux pistes sont closes.
- La session s'est arrêtée en inverse sur `etats` : une colonne d'états entièrement inconnue (NaN) sur les lignes
  d'entraînement d'un pli fait échouer le découpage en classes de HGB. Correction : colonnes à moins de deux valeurs
  finies écartées pli par pli (`utiles`). `etats` en inverse et en chrono (+ `base` en chrono) sont relancés en K5.

## Découverte : les états d'AXXX301 suivent le régime, mais leur lettre change d'une année à l'autre

L'écart entre le gros gain interne d'`etats` (directe −15 %) et l'échec du test d'états croisé entre années en D2
s'explique en regardant chaque année seule. Corrélation entre l'état d'une famille et le régime journalier de
2551 :

- annee1 : `AXXX301A` et `AXXX305A` +0.99 (régime haut), `AXXX301B` et `AXXX305B` −0.99, dans chaque bloc ;
- annee2 : `AXXX301C` +0.97 à +1.00 dans les blocs 0 à 2, `AXXX301E` −0.98, `AXXX301B` −0.99 ; `AXXX305` reste
  `B` toute l'année.

Les **changements de code** d'AXXX301 tombent à ±1 jour des bascules : annee1, 14116 → B (bascule 14117), 14284 → A
(14284), 14367 → B (14368), 14732 → B (14732) ; annee2, 14871 → C (14872), 15040 → E (15041), 15152 → C (15153).
Les bascules sans changement visible sont dans des lacunes du journal dmop (annee1 : 14536, lacune de 358 h dès
14528 ; annee2 : 14991 et 15026, lacune de 825 h dès 14990) ou en fin d'annee2 (15321, 15370, 15433, 15460), où
AXXX301 reste en `C` : le commutateur y paraît autoriser le chauffage, que le thermostat fait ensuite varier avec
le contexte.

D2 avait manqué ce signal : avec le vocabulaire de l'année d'entraînement, `C` et `E` sont des codes inconnus en
annee2, et `A` en annee1. C'est exactement le cas « changement de variante » de la demande.

**Ajout `commutateur`** (`baselines/gbm_sacs.py`, test `test_commutateur_bascule_sur_code_inconnu`) :
- appris sur l'entraînement seul : familles dont le code courant sépare le régime (η² ≥ 0.5 dans une période),
  et part de régime haut sous chaque code vu au moins 168 h ;
- état reconstruit commande par commande : code connu net (≥ 0.7 ou ≤ 0.3) → son état ; code inconnu ou ambigu
  différent du précédent → l'état bascule ; même code → inchangé ; avant la première commande et pour un premier
  code inconnu → régime du dernier jour de `power_init` (au test) ; NaN pendant une lacune dmop de plus de 48 h ;
- familles réunies en un commutateur « actif » (celle dont le code a changé le plus récemment), parce qu'AXXX305
  ne change plus en annee2 (constat sur les seules commandes, pas sur la consommation) ;
- sorties : `comm_haut`, `comm_inconnu`, heures depuis / jusqu'au changement de code, depuis la dernière commande.

Contrôle sans regarder la consommation de test : la part d'heures « haut » reconstruite sur le test avec le
vocabulaire de l'autre année (0.68 pour annee2, 0.41 pour annee1) est celle qu'on obtient avec le vocabulaire de
l'année elle-même.

**Honnêteté sur la sélection** : cette hypothèse vient de la lecture des deux années (comme les diagnostics des
programmes précédents) et s'ajoute aux 4 candidats prévus. Elle compte comme un cinquième candidat ; la règle
d'installation reste celle fixée plus haut, et la chrono n'est pas une période indépendante de la directe.

## K5 — `etats` en inverse et en chrono (relance après correction, `runs/p3_var_etats`, `runs/p3_var_base/chrono`)

| OOB | directe | inverse | chrono |
|---|---|---|---|
| S (hors sac, sans embargo) | 0.2405 | 0.2422 | 0.2653 |
| base (variables de S, HGB D3) | 0.2453 | 0.2370 | 0.2928 |
| + etats | 0.2086 (−13 % contre S) | 0.2349 (−3.0 %) | 0.2562 (−3.4 %) |

`etats` passe le critère interne (≥ 1 % dans les trois directions) : premier candidat envoyé en validation externe
(`bilan_2551.py --evaluer`, `runs/p3_eval_etats`).

| RMSE 2551 | directe g04 / g59 | inverse g04 / g59 | chrono |
|---|---|---|---|
| Livré | 0.2583 / 0.2572 | 0.2054 / 0.2046 | 0.2557 |
| S seul | 0.2678 | 0.2229 | 0.2641 |
| etats seul (biais) | 0.3080 (−0.072) | 0.2197 (−0.011) | 0.3196 (+0.110) |
| etats en remplacement | 0.2656 / 0.2653 | 0.2041 / 0.2034 | 0.2669 |
| Mélange aux poids transférés | poids 0 partout : = livré | = livré | = livré |

- **Décision** : refusé (gain moyen −0.0048 en remplacement ; directe et chrono en forte hausse, MAE en hausse, fenêtres
  de transition +0.010 à +0.026). Échec de transfert : la famille « états par code » est close.
- **Ce que cela invalide** : un état de commande défini par sa lettre. Le modèle apprend « A = régime haut » en
  annee1 ; en annee2, `C` et `E` sont inconnus, le modèle les lit comme « pas A » et sous-estime (biais −0.072).
  La validation interne ne pouvait pas le voir : dans une même année, les lettres ne changent pas.
- K6 (ci-dessous) avait été lancé avant ce résultat externe ; ce n'est pas une variante choisie après l'échec
  d'`etats`, mais l'hypothèse « bascule sur changement de code, indépendante de la lettre », formée par la lecture
  par année de la section précédente.

## K6 — Commutateur reconstruit (`recherche_2551.py variables` jeu `comm` ; candidat installable `runs/p3_comm`, 39 min)

- **Hypothèse** : l'état d'AXXX301, reconstruit sans dépendre de la lettre (bascule sur un code inconnu, état
  initial tiré de `power_init`, NaN dans les lacunes), donne la date des bascules que le contexte ne donne pas.
- **Changement** : ajout `commutateur` aux variables de S. Validation interne avec HGB D3 ; candidat installable
  = spécialiste installé (D3L + orbite, 5 sacs, `outils/specialistes.py entrainer`) + `commutateur`.

| OOB | directe | inverse | chrono |
|---|---|---|---|
| S | 0.2405 | 0.2422 | 0.2653 |
| + commutateur (HGB D3) | 0.2014 (−16 %) | 0.2348 (−3.1 %) | 0.1863 (−30 %) |

Critère interne rempli ; validation externe (`runs/p3_eval_comm`). Note : `specialistes.py entrainer` affiche de
lui-même les RMSE externes de la directe et de l'inverse à la fin ; cela n'a rien changé, aucun choix ne restait.

| RMSE 2551 (biais) | directe g04 / g59 | inverse g04 / g59 | chrono |
|---|---|---|---|
| Livré | 0.2583 / 0.2572 | 0.2054 / 0.2046 | 0.2557 |
| S seul | 0.2678 (+0.025) | 0.2229 (+0.011) | 0.2641 (+0.011) |
| Commutateur D3L seul | 0.3214 (+0.134) | **0.2081 (+0.012)** | 0.4069 (+0.177) |
| Commutateur D3L en remplacement | 0.2651 / 0.2637 | **0.1981 / 0.1973** | 0.2931 |
| Mélange aux poids transférés | 0.2622 / 0.2615 | 0.2048 / 0.2040 | 0.3381 |
| Commutateur HGB D3 en remplacement | 0.2742 / 0.2728 | 0.1994 / 0.1986 | 0.2968 |

- **Décision** : refusé (gain moyen −0.0072 en remplacement). En inverse, c'est le meilleur résultat jamais obtenu
  sur 2551 (−0.0073, fenêtres de transition 0.307 → 0.292), mais la directe et la chrono se dégradent fortement.
  La famille « commutateur comme variable » est close (critère 2).
- **Cause, mois par mois sur la directe** : quand le commutateur reconstruit dit « coupé », la vérité est basse dans
  99 % des heures ; quand il dit « actif », elle n'est haute que dans 82 % des heures. Les cinq derniers mois
  d'annee2 (jours ≈ 15310–15460) gardent AXXX301 en `C` (actif) alors que 2551 reste autour de 0.07–0.27 : le
  candidat prédit 0.56–0.67 (RMSE mensuelle 0.41 à 0.55 contre 0.13 à 0.33 pour S). C'est aussi la période de test
  de la chrono, d'où son biais de +0.18.
- **Ce que cela démontre** : le commutateur est une condition **nécessaire** du régime haut, pas suffisante, et la
  part « actif mais froid / chaud » change d'une année à l'autre. On le voit sur les seules années
  d'entraînement : en annee1, le code actif `A` donne 99 % d'heures hautes ; en annee2, le code actif `C` en donne
  82 % ; les codes coupés en donnent 0 à 1 % les deux années. Un modèle appris sur annee1 ne peut pas apprendre
  l'exception d'annee2.
- **Ce qui reste ouvert** : une règle asymétrique (« commutateur coupé ⇒ régime bas », le livré gardé ailleurs)
  correspond à ce que montrent les deux années d'entraînement. Mais je l'ai formulée après avoir vu les erreurs
  de la directe ; l'évaluer sur ces mêmes validations serait contaminé. Elle n'est donc pas essayée ici (voir la
  conclusion).

## Conclusion du programme 3

**Rien n'est installé ; le livré reste (0.076735), `poids/` est inchangé.** Aucun candidat ne remplit la règle
d'installation, et la règle d'arrêt 3 s'applique. 5 sessions d'expérience sur 6 utilisées, aucune session
d'installation.

Tableau de transfert (validations externes ; RMSE de 2551, groupe de graines g04 pour la base) :

| | Interne : gain contre S (dir / inv / chr) | Seul dir / inv / chr | Remplacement dir / inv / chr | Décision |
|---|---|---|---|---|
| Livré | | 0.2583 / 0.2054 / 0.2557 | | référence |
| S (spécialiste installé) | | 0.2678 / 0.2229 / 0.2641 | | |
| K1 XGB réglé | −0.9 / −3.5 / −3.8 % | 0.2899 / 0.2246 / 0.2739 | 0.2651 / 0.2055 / 0.2567 | critère interne |
| K2 deux régimes | −8.2 / −1.1 / +1.5 % | 0.2882 / 0.2224 / 0.2918 | 0.2624 / 0.2044 / 0.2632 | critère interne |
| K4 correction de S | +10 / +13 / +24 % | 0.2767 / 0.2117 / 0.3109 | 0.2623 / 0.2010 / 0.2657 | critère interne |
| K3/K5 + etats | −13 / −3.0 / −3.4 % | 0.3080 / 0.2197 / 0.3196 | 0.2656 / 0.2041 / 0.2669 | échec de transfert |
| K6 + commutateur | −16 / −3.1 / −30 % | 0.3214 / 0.2081 / 0.4069 | 0.2651 / 0.1981 / 0.2931 | échec de transfert |

### Limites démontrées

1. **61 à 76 % de la MSE du livré est horaire** (D1). Elle n'est verrouillée sur l'orbite qu'à 14–17 %, même pour un
   oracle local : c'est le cycle du thermostat, hors de portée sans consommation récente.
2. **Le régime aux bascules n'est pas lisible dans le contexte** (D2 : 64–69 % de jours bien classés à ±14 jours,
   date déplacée d'au moins une semaine pour 12 bascules sur 14). Il l'est dans les commandes AXXX301 (bascules à
   ±1 jour), mais (a) la lettre change d'une année à l'autre, (b) des lacunes dmop de 15 à 35 jours masquent des
   bascules, (c) le commutateur actif n'implique pas le chauffage en fin d'annee2.
3. **La validation interne ne prédit pas le transfert entre années** pour 2551 : les 5 candidats qui gagnent en
   interne perdent en directe ou en chrono. Les comportements diffèrent d'une année à l'autre : part de régime haut
   0.41 / 0.56, préchauffage avant 23 jours de pré-saison sur 60 / 13 sur 90, commutateur actif = chauffage 99 % /
   82 %. Avec deux années, aucune ne contient l'exception de l'autre.
4. Les fenêtres futures longues (`echelles`) apprennent une propension moyenne de préchauffage (D3) ; leurs
   variantes sans futur ou avec bords NaN dégradent l'OOB.

### Prochaine expérience la plus informative

**Porte asymétrique du commutateur** : 2551 = niveau bas (médiane des jours bas de l'entraînement, ou
min(livré, ce niveau)) quand le commutateur reconstruit dit « coupé » ; livré inchangé quand il est actif ou
inconnu. Elle ne demande aucun entraînement, n'utilise que les commandes et `power_init`, et ne modifie que 2551.
Ce qui la justifie se voit sur les deux années d'entraînement (codes coupés : 0 à 1 % d'heures hautes ; actifs : 82
à 99 %). Ce qui la limite : elle a été formulée après lecture des erreurs de la directe ; ses scores sur les trois
validations actuelles seraient donc optimistes.

### Vérifications finales

- Refactor de `baselines/gbm_sacs.py` (`preparer`, `construire_test`) : les prédictions du spécialiste installé
  (`rc0_livre_g04/directe/specialiste/gbm_NPWD2551`) sur `validation/test` sont identiques au bit près avec le code
  de `HEAD` et avec le nouveau code.
- Réserve sur la forme « remplacement » (programmes 2 et 3) : S y est lu dans `runs/n2_spec` (= `s51_gbm_D3Lo`), qui
  n'est pas exactement l'entraînement installé dans le livré (`s51_spec`, réentraîné ensuite) : jusqu'à 0.0021
  d'écart horaire. L'effet sur les RMSE de remplacement est de l'ordre de 10⁻⁴ et ne change aucune décision
  (rejets à plus de 0.004 de gain moyen manquant).
- Tests : `uv run python -m unittest tests.test_gbm_sacs tests.test_regime_2551 tests.test_ensemble
  tests.test_caracteristiques tests.test_recette` : 42 tests OK (1 ignoré : xgboost absent en local).
- `poids/` inchangé ; aucune sortie des 32 autres circuits modifiée (rien n'est installé).

### Reproduction

```bash
uv run python outils/bilan_2551.py --nom p3_d1 --evaluer --candidats hgbe=n2_hgb_e xgb20=n2_xgb20
uv run python outils/lancer_kaggle.py --fente 1 --legers -- outils/recherche_2551.py modele --nom p3_xgb
uv run python outils/lancer_kaggle.py --fente 2 --legers -- outils/recherche_2551.py regime --nom p3_regime
uv run python outils/lancer_kaggle.py --fente 1 --legers -- outils/recherche_2551.py variables --nom p3_var \
  --modele '{"hgb": "D3"}' --jeux '{"base": {}, "etats": {"ajouts": ["etats"]}, "comm": {"ajouts": ["commutateur"]}}'
uv run python outils/bilan_2551.py --nom p3_eval_comm --evaluer --candidats comm=p3_comm comm_d3=p3_var_comm
```

`regime` (correction K4) lit les prédictions de S dans `entrees_2551/DIRECTION.npz` (heures en écarts int16,
valeurs × 1e4 : le code envoyé à Kaggle doit rester sous environ 1 Mo). Ces fichiers sont rangés dans
`runs/p3_entrees_2551/` ; les remettre à la racine avant de relancer. Sorties : `runs/p3_*` (`choix.json`,
`oob.csv`, `pred.csv`, `bilan.json`), journaux de session `runs/_kaggle/202611*`.

# Programme 3 bis (14 novembre 2026) : porte asymétrique du commutateur, à la demande

L'utilisateur demande de tester la règle proposée en conclusion du programme 3. **Réserve écrite avant la mesure** :
la règle a été formulée après avoir vu les erreurs de la directe du candidat commutateur. Ses scores sur les trois
validations sont donc optimistes ; un succès ici rend l'hypothèse plausible, il ne la démontre pas.

## Règle, fixée avant toute mesure

- Commutateur appris sur l'année (ou les périodes) d'entraînement de chaque direction seulement
  (`gbm_sacs.apprendre_commutateur`), état reconstruit sur le test (`variables_commutateur`, état initial tiré de
  `power_init`, NaN pendant les lacunes dmop).
- Niveau bas = médiane horaire de 2551 sur les jours bas de l'entraînement.
- **Forme principale** : 2551 = min(livré, niveau bas) aux heures où le commutateur reconstruit vaut 0 (coupé) ;
  livré inchangé ailleurs (actif ou inconnu). Seule cette forme décide.
- Forme secondaire, à titre d'information : 2551 = niveau bas aux heures coupées (remplacement au lieu du min).
- Aucun paramètre réglé ; même règle d'installation que le programme 3 (5 cas, gain par période, gain moyen ≥ 0.0015,
  moitiés, MAE, biais, fenêtres de transition). Les 32 autres colonnes ne sont pas touchées par construction.

## Résultats (`python outils/bilan_2551.py --nom p3b_porte --porte`, calcul local, aucun entraînement)

| | directe g04 / g59 | inverse g04 / g59 | chrono |
|---|---|---|---|
| Heures « coupé » (sur les heures notées) | 4066 / 16073 | 8758 / 16287 | 46 / 7844 |
| Part de ces heures dans des jours vrais bas | 99.2 % | 99.5 % | 100 % |
| Livré | 0.2583 / 0.2572 | 0.2054 / 0.2046 | 0.2557 |
| **Porte, forme principale** (min(livré, 0.0005)) | 0.2638 / 0.2627 | 0.2044 / 0.2038 | 0.2557 |
| Moitiés directe g04 / inverse g04, livré → porte | 0.2214 → 0.2344 ; 0.2905 → 0.2902 | 0.1716 → 0.1520 ; 0.2344 → 0.2458 | = |
| Biais livré → porte | +0.030 → +0.014 | +0.016 → −0.045 | = |

- **Décision** : refusée (gain moyen −0.0018 ; directe en hausse de 0.0055, moitiés en hausse de plus de 0.002 en
  directe et en inverse, |biais| en hausse en inverse). Forme secondaire (remplacement) : mêmes chiffres à 10⁻⁴ près.
- **La première moitié de l'hypothèse est fondée** : quand le commutateur reconstruit (appris sur l'autre année, avec
  sa lettre de variante différente) dit « coupé », le jour est vraiment bas dans 99.2 à 99.5 % des heures.
- **La seconde moitié ne l'est pas** : forcer un niveau bas constant fait perdre. Le régime bas n'est pas plat : aux
  heures coupées, 2551 a une médiane de 0.0005 mais une moyenne de 0.05, un 90ᵉ centile de 0.21 et des pointes
  jusqu'à 1.0. Le livré suit déjà ces pointes : corrélation de 0.86 avec la vérité en directe (0.66 en inverse), RMSE
  0.058 contre 0.111 pour la meilleure constante. La porte efface cette information.
- Contrôle a posteriori, hors décision : même avec la constante optimale au sens de la MSE (moyenne des heures basses
  de l'entraînement, 0.0495), la directe perd (0.2583 → 0.2621) et l'inverse gagne (0.2054 → 0.2000).
- **Où la porte gagne** : un seul épisode, en inverse, mois 8–9 d'annee1 (≈ jours 14350–14410). Le livré y reste haut
  alors que le réchauffeur est éteint depuis la bascule 14368 (la surestimation de +0.34 vue au programme 2). Sur ces
  deux mois, la somme des carrés baisse de 110 ; partout ailleurs, elle monte.
- **Pourquoi aucune porte « coupé » ne peut passer la règle** : sur A2b (test de la chrono), le commutateur n'est
  presque jamais coupé (46 heures, AXXX301 reste en `C`). Toute règle déclenchée par « coupé » donne donc un gain nul
  sur A2b, alors que la règle d'installation exige au moins 0.0005 sur chaque période.

## Conclusion

L'information « commutateur coupé ⇒ régime bas » est réelle et se transfère d'une année à l'autre malgré le
changement de lettre. Mais elle est presque redondante avec le livré, qui prédit déjà bas sur ces heures, sauf lors
de rares épisodes où il reste haut à tort. Une règle limitée à ces épisodes (porte seulement quand le livré prédit un
jour haut alors que le commutateur est coupé) serait la suite logique. Elle serait formulée après lecture de ces
résultats, ne toucherait qu'un épisode d'annee1, et ne peut rien gagner sur A2b : elle n'est pas essayée. Rien n'est
installé ; le livré reste (0.076735).

# Programme 4 (14–15 novembre 2026) : reproduction de la recette B

Demande : implémenter et évaluer la recette indépendante `controles/recette_b/RECETTE_AUTONOME.md` (copie des
fichiers joints, empreintes vérifiées contre `SHA256.json`), d'abord comme **reproduction fidèle**, puis isoler ce
que changent les entrées, les réglages d'entraînement et le masque. Pas de recherche de paramètres. Années 1 et 2
seulement. Entraînements sur Kaggle `zendarc148` (T4) ; en local, seulement préparation
des entrées, scores et re-prédiction avec des poids existants. Référence et repli : le livré (0.076735).

## Étape 1 — Livré gelé (`python outils/recette_b.py geler`, `runs/b0_gel/`)

`empreintes.json` : sha256 de tous les fichiers de `poids/`, des prédictions de référence (`rc0_livre_g04|g59`,
composants chrono `rc7_chrono_mixte`, `rc7_chrono_g2`, `n2_base_chrono`, `n2_spec`), état git. Copies en lecture
seule dans `copies/`. **Spécialiste exact** : le boosting de 2551 installé dans `rc0_livre_*/DIR/specialiste/`
(même fichier pour g04 et g59) re-prédit sur `validation/` et `validation_inverse/` ; chrono = `n2_base_chrono`
(composant exact de la base chrono). Il remplace `n2_spec` dans les formes « S seul » et « remplacement », ce qui
lève la réserve du programme 3.

## Étape 2 — Module indépendant

`baselines/recette_b.py` : numpy, pandas, scikit-learn seulement ; aucune fonction de `dio/` ni de `gbm_sacs` (il
doit tourner dans l'environnement épinglé, sans torch) ; aucun défaut du livré changé. Fonctions existantes
examinées et **non réutilisées** faute d'équivalence numérique : `gbm_sacs.periodes_actives` et
`caracteristiques.periode_paire` (min des débuts / max des fins par numéro, paires > 7 jours écartées ; la recette
veut tous les couples de la jointure interne, sans plafond), `contexte.intervalles` (paires > 6 h écartées),
`caracteristiques.log_altitude` (moyenne des log aux timestamps dupliqués ; la recette ne déduplique pas),
`contexte.geometrie` (période médiane ajoutée aux bords), `caracteristiques.ltdata` / `_interp` (`np.interp` au lieu
de `interpolate(method="index")` sur l'union des instants). Réutilisé : `baselines/boosting.recaler` (fonction pure)
pour l'axe recalage.

Interprétations de la recette (là où le texte laisse un choix) : tris par `ut_ms` stables, sauf l'altitude
(`np.argsort` par défaut, comme écrit) ; changements de variante datés à leur heure même hors grille ; récences en
float32 avant `log1p` (§8) ; distances aux changements de variante en float64 (voir étape 3).

## Étape 3 — Contrôles des entrées (avant tout entraînement)

`python outils/recette_b.py controle` dans deux environnements : principal (Python 3.13.9, numpy 2.3.5, pandas
2.3.3, scikit-learn 1.9.1) et **épinglé** séparé (`uv venv` Python 3.11.16, numpy 1.26.4, pandas 2.3.3,
scikit-learn 1.6.1, scipy 1.13.1, joblib 1.6.0, threadpoolctl 3.8.0 ; l'environnement du projet n'est pas touché).
Sorties `runs/b1_controle/controle_{principal,epingle}.json`.

| Contrôle (X année 1, vocabulaire de contrôle années 1+2) | principal | épinglé |
|---|---|---|
| 846 colonnes, noms et ordre de `colonnes_reference.json` | oui | oui |
| Grille 16 488 h, 1219363200000 → 1278716400000, sha256 de ut_ms | oui | oui |
| float32 | oui | oui |
| 11 lignes de `controle_entrees_annee1.csv` (atol = rtol = 1e-6) | 0 cellule hors tolérance (écart max 4.7e-7) | idem |
| **sha256 binaire de X** (`7ebb2195…`) | **identique** | **identique** |
| Vocabulaire réappris sur années 1+2 (préfixes 31, stems 149, couples de switch 77, FTL 16, commandes 198) | identique | identique |

Première version : colonnes et 11 lignes justes, mais empreinte différente (`55a653fe…`) dans les deux
environnements, donc pas un effet de version. Recherche de la première divergence par variantes : passer les
distances aux changements de variante (`dm_sw_*_chg`, `dm_sw_*_next`, 58 colonnes) de float32 à float64 avant
`log1p` redonne l'empreinte exacte ; passer aussi les récences en float64 l'éloigne (144 colonnes changent). Les
écarts étaient d'un ulp float32 sur `log1p`, invisibles à 1e-6. Corrigé dans le module (la recette ne fixait le
float32 que pour les récences).

Vocabulaires propres à chaque validation (`runs/b1_controle/vocabulaires.json`, jamais le vocabulaire de contrôle) :

| Validation | Lignes d'entraînement | Colonnes | Préfixes | Stems | Couples de switch | Types FTL | Lignes test / notées |
|---|---:|---:|---:|---:|---:|---:|---|
| directe (annee1) | 16 488 | 684 | 26 | 117 | 49 | 14 | 16 488 / 16 319 |
| inverse (annee2) | 16 488 | 809 | 31 | 142 | 64 | 15 | 16 488 / 16 319 |
| chrono (annee1 + A2a) | 24 732 | 710 | 28 | 122 | 51 | 16 | 8 244 / 8 075 |

- **Masque collectif = masque propre** dans les trois validations : aucune heure n'a NPWD2551 connue avec une autre
  cible manquante (16 456, 16 242 et 24 685 lignes gardées dans les deux cas). L'effet du masque est nul par
  construction sur ces données ; une cellule à masque propre est quand même entraînée comme preuve.
- Chrono : coupure 1308398400000 (`outils/faire_chrono.py`), power_init sur 7 jours, première heure notée
  1309006800000, dernière 1338073200000 (8 075 heures). Sur Kaggle, `power.csv` est reconstruit depuis `y.csv`
  (`outils/reconstruire_power.py`, mesures placées à hh:30) : mêmes heures, mêmes moyennes horaires.
- **Bords** (`runs/b1_controle/bords.json`) : directe et inverse, X du dossier de test = X de l'année entière
  (0 cellule différente). Chrono, dossiers découpés contre annee2 entière puis tranchée (référence de la recette) :
  même vocabulaire ; entraînement A2a : 1 687 cellules sur 519 lignes (les 3 derniers jours avant la coupure :
  changements de variante, comptes futurs, ltdata du dernier jour, décalages) ; test A2b : 11 913 cellules sur
  6 321 lignes, dont `dm_act_*` sur 6 318 lignes (périodes PENS…/PENE… ouvertes avant la coupure, sans plafond de
  durée, invisibles depuis le dossier de test), récences de préfixes et de stems au début du test. La forme
  principale garde les dossiers découpés (mêmes informations que les références locales) ; la variante « année
  entière puis tranchée » est entraînée à part, notée hors décision.

## Protocole et critères (fixés avant les entraînements)

- **Plan factoriel** sur NPWD2551 seul, mêmes 5 sacs (`linspace(0, n, 6)`, sans embargo, avant masque), graine 0,
  même code d'ajustement : entrées {actuelles = matrice du spécialiste livré par `gbm_sacs.preparer` /
  `construire_test`, float64 avec NaN ; recette B, float32} × réglages {D3L du livré ; recette : `PARAMS_B`} ×
  recalage power_init {livré : demi-vie 1440 h ; aucun}, + une cellule recette à masque propre. Le recalage est un
  post-traitement (pas de réentraînement). Les cibles suivent leur préparation (y.csv float64 pour les entrées
  actuelles, moyenne horaire de power.csv en float32 pour la recette).
- **Recette B stricte** = entrées recette, réglages recette, masque collectif, sans recalage, entraînée dans
  l'environnement épinglé (reproduction) **et** dans l'environnement principal (version installable par
  `predict.py`, effet de l'environnement mesuré).
- **Statut des chiffres.** Reproduction : contrôles d'entrées, recette stricte. Comparaison exploratoire : tout le
  reste ; les validations directe / inverse / chrono ont servi à une vingtaine de choix ; directe et chrono ne sont
  pas indépendantes (A2b ⊂ A2) ; les groupes de graines g04 / g59 sont des répétitions. **Aucune preuve de
  généralisation** n'est possible ici : aucune période des années 1–2 n'est vierge.
- **Seul candidat installable** : la recette B stricte (version environnement principal), en forme remplacement
  (part boosting de 2551 dans le livré, `clip(L + 0.5 (B − S))`, part GRU conservée), sans coefficient appris. Les
  autres cellules sont des ablations explicatives, jamais des candidats. Le mélange à poids transférés (périodes
  disjointes, `bilan_2551.SOURCES`) est donné pour information.
- **Règle d'installation** = règle du programme 3 (`bilan_2551.regle`) : RMSE en baisse dans les 5 cas ; gain ≥ 0.0005
  sur A1, A2 et A2b ; gain moyen ≥ 0.0015 ; aucune moitié en hausse > 0.002 ; MAE non en hausse par période ;
  |biais| pas plus de 0.01 au-dessus ; fenêtres de transition pas plus de 0.005 au-dessus ; **plus** aucun mois
  (≥ 300 h notées) en hausse de plus de 0.01 ; la version épinglée doit donner la même décision. Une égalité ne
  serait tolérée que sur une période où la correction ne s'active pas : sans objet, le remplacement agit partout.
- Sinon rien n'est installé. Si installation : entraînement années 1+2 sur Kaggle, `poids/specialiste/gbm_NPWD2551_b/`
  à côté de l'ancien (repli), puis preuve que les 32 autres colonnes sont identiques au bit près.

## Étapes 4–5 — Entraînements Kaggle (`zendarc148`, sessions T4)

| Session | Commandes | Durée des cellules |
|---|---|---|
| K1 (`dream-in-orbit-1`) | `recette_b.py matrices --session k1` ; `entrainer --cellules rec_recette_coll --prefixe b3k` (env principal) ; `entrainer --env-recette --cellules rec_recette_coll rec_recette_propre cur_recette_coll rec_d3l_coll --bord` | réglage recette 63–100 s par validation ; D3L 456–707 s |
| K2 (`dream-in-orbit-2`) | `matrices --session k2` ; `entrainer --env-recette --cellules cur_d3l_coll` | D3L 521 / 601 / 840 s |

- Environnements sur Kaggle : principal Python 3.13.16, numpy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1 (Tesla T4,
  torch 2.15.0+cu132) ; **épinglé créé dans la session** par `uv` : Python 3.11.17, numpy 1.26.4, pandas 2.3.3,
  scikit-learn 1.6.1. Pas de repli.
- Entrées recette reconstruites dans l'env épinglé : identiques au bit près à celles de l'env principal, dans les
  trois validations. Matrices K1 = K2 = local au bit près (x et y, actuelles et recette), sauf `y_init` / `n_init`
  (power_init reconstruit à hh:30 sur Kaggle ; n'affecte que la variante recalée).
- Empreintes sha256 (12 premiers caractères ; complètes dans `runs/b2_matrices/resume_k1.json` et `info.json`) :

| | X recette train / test | X actuelles train / test | Colonnes recette / actuelles |
|---|---|---|---:|
| directe | `c2a1c5a1fba4` / `d915fdf62890` | `34765e5de604` / `b2d0d1200d0d` | 684 / 807 |
| inverse | `e284b3c1e7c7` / `08ebe7e54a5f` | `c0e099dd8a56` / `d23060d68428` | 809 / 934 |
| chrono | `5bf3a8fb0ef6` / `db8579912ed4` | `b0dab75031ec` / `eb6a5f16fc68` | 710 / 827 |
| chrono, année entière tranchée | `4814078d9fb9` / `076784a5bce2` | – | 710 |

- Réglage recette : `early_stopping='auto'` est **actif** (plus de 10 000 lignes par sac : 13 159–13 174 en
  directe), mais n'arrête jamais avant 600 : les 5 sacs ont 600 arbres dans les trois validations et les deux
  environnements. D3L : 2 969 à 6 000 arbres (plafond atteint dans tous les sacs de la chrono).
- Masque : `masque_collectif_egal_propre = true` dans toutes les cellules ; les cellules à masque collectif et à
  masque propre donnent des prédictions identiques au bit près.

## Étapes 6–7 — Résultats (`python outils/recette_b.py evaluer`, `runs/b4_bilan/bilan.json`, calcul local)

### Contrôles

- **Forme remplacement exacte** : la part GRU du livré est déjà bornée à 0 (`mixte` et `recette` bornent leurs
  prédictions), donc L − 0.5 S ≥ 0 partout (minimum −6e-17) : `clip(L + 0.5 (B − S))` est exactement le livré
  dont seule la part boosting de 2551 est remplacée.
- S exact contre `n2_spec` : même RMSE à 1e-5, écart horaire max 0.0021 (directe), 0.0020 (inverse), 0 (chrono).
- Spécialiste livré réentraîné par ce code dans l'env épinglé (cellule cur_d3l, recalée) contre S exact : écart
  horaire moyen 0.006, max 0.05 ; RMSE 0.2680 / 0.2231 / 0.2644 contre 0.2678 / 0.2229 / 0.2641. Pas d'identité
  au bit : versions de scikit-learn différentes et power_init reconstruit. **Effet d'environnement** sur la recette
  stricte (épinglé contre principal) : RMSE 0.2786 / 0.2270 / 0.2828 contre 0.2772 / 0.2260 / 0.2828, écart horaire
  max 0.11. Il est de l'ordre de 0.001 : tout écart de RMSE inférieur n'est pas interprétable.

### Les trois objets, séparés (RMSE / MAE / biais de 2551)

| | directe g04 | directe g59 | inverse g04 | inverse g59 | chrono |
|---|---|---|---|---|---|
| Livré complet | 0.2583 / 0.178 / +0.030 | 0.2572 / 0.178 / +0.028 | 0.2054 / 0.129 / +0.016 | 0.2046 / 0.129 / +0.017 | 0.2557 / 0.186 / +0.000 |
| Spécialiste livré seul (S exact) | 0.2678 / 0.188 / +0.025 | = | 0.2229 / 0.132 / +0.011 | = | 0.2641 / 0.196 / +0.011 |
| **Recette B seule** (stricte, épinglée) | 0.2786 / 0.199 / +0.036 | = | 0.2270 / 0.141 / −0.009 | = | 0.2828 / 0.223 / +0.074 |
| Recette B seule (env principal, installable) | 0.2772 / 0.197 / +0.032 | = | 0.2260 / 0.140 / −0.009 | = | 0.2828 / 0.224 / +0.075 |
| **Livré, part boosting de 2551 remplacée par B** (épinglée) | 0.2612 / 0.182 / +0.036 | 0.2601 / 0.181 / +0.033 | 0.2080 / 0.135 / +0.006 | 0.2072 / 0.135 / +0.007 | 0.2603 / 0.197 / +0.032 |
| idem, B env principal | 0.2607 | 0.2596 | 0.2074 | 0.2067 | 0.2604 |
| Mélange à poids transférés (information) | poids de B = 0 partout : = livré | = | = | = | = |

Par période (remplacement, B env principal) : A2 (directe) +0.0024, A1 (inverse) +0.0020, A2b (chrono) +0.0046.
Moitiés : directe 0.2214 → 0.2211 puis 0.2905 → 0.2950 ; inverse 0.1716 → 0.1739 puis 0.2344 → 0.2363 ; chrono
0.2942 → 0.2960 puis 0.2104 → 0.2190. Fenêtres de transition chrono 0.206 → 0.214. 20 mois en hausse de plus de
0.01 (directe 2011-10, 2012-01 à 03 ; inverse 2008-09 à 11, 2009-12, 2010-01 ; chrono 2011-10, 2012-01, 2012-04).

Par régime (B seule contre S, RMSE) : hors saison d'éclipses, directe 0.200 → 0.221, chrono 0.185 → 0.227 (biais
+0.059 → +0.136) ; semaines vraies basses de la chrono 0.168 → 0.237 (biais +0.131 → +0.211). En inverse, B est
meilleure hors saison (0.158 → 0.156) mais pire en saison (0.295 → 0.304). Par mois, B bat S dans 7 mois sur 23
(directe), 8 sur 24 (inverse), 4 sur 12 (chrono). Premier mois de l'inverse (2008-08) : 0.089 → 0.238, sans
recalage sur power_init.

### Décomposition (plan factoriel, RMSE du boosting seul)

| RMSE seule (directe / inverse / chrono) | sans recalage | avec recalage livré |
|---|---|---|
| entrées actuelles, réglage D3L (= livré réentraîné) | 0.2677 / 0.2247 / 0.2646 | 0.2680 / 0.2231 / 0.2644 |
| entrées actuelles, réglage recette | 0.2705 / 0.2237 / 0.2701 | 0.2710 / 0.2228 / 0.2693 |
| entrées recette, réglage D3L | 0.2776 / 0.2243 / 0.2773 | 0.2784 / 0.2228 / 0.2763 |
| entrées recette, réglage recette (B) | 0.2786 / 0.2270 / 0.2828 | 0.2796 / 0.2261 / 0.2811 |
| entrées recette, réglage recette, masque propre | identique au bit près | identique |

| Effet moyen sur la RMSE (+ = pire) | directe | inverse | chrono |
|---|---:|---:|---:|
| Entrées (recette − actuelles) | **+0.0092** | +0.0015 | **+0.0123** |
| Réglages (recette − D3L) | +0.0020 | +0.0011 | +0.0052 |
| Interaction entrées × réglages (sans recalage) | −0.0018 | +0.0037 | 0.0000 |
| Recalage power_init (avec − sans) | +0.0006 | −0.0012 | −0.0009 |
| Masque (propre − collectif) | 0 | 0 | 0 |
| Variante de bord (année entière tranchée − dossiers) | | | −0.0010 |

Chemin S → B (directe) : 0.2680 → sans recalage 0.2677 → réglage recette 0.2705 → entrées recette 0.2786.

Validation interne (hors bloc, année d'entraînement, sans embargo) : entrées actuelles 0.240 / 0.238 / 0.265
(D3L) contre recette 0.265 / 0.236 / 0.297 (recette). Les entrées recette perdent déjà **dans l'année**
d'entraînement en directe et en chrono : ce n'est pas un défaut de transfert entre années, il leur manque de
l'information présente dans la matrice actuelle (blocs V1/V2 de `prepare.py`, planning par famille avec lettres
fusionnées, géométrie et ombres de `contexte.py`, orbite lente, flux de `variables.physiques`).

## Décision

**Refusé : rien n'est installé, `poids/` est inchangé (empreintes de `runs/b0_gel/empreintes.json`), le livré reste
(0.076735).** La recette B stricte en remplacement est pire dans les 5 cas (gain moyen −0.0027 en env principal,
−0.0031 épinglée) et échoue à tous les critères fixés avant mesure (gain par période, MAE, moitiés, mois, biais et
transitions en chrono). Les deux environnements donnent la même décision. Le mélange à poids choisis hors période
notée donne un poids nul à B dans chaque cas. Les 32 autres colonnes n'ont pas eu à être revérifiées.

## Conclusion : d'où vient l'écart

- **La reproduction est fidèle** : entrées identiques au bit près à l'empreinte de contrôle, vocabulaire années 1+2
  identique, entraînement aux paramètres exacts dans l'environnement de référence (600 arbres partout).
- **L'écart entre la recette B et le spécialiste livré vient surtout des entrées** (directe +0.009, chrono +0.012 ;
  visible aussi en validation interne), **ensuite des réglages** (+0.002 / +0.001 / +0.005 : 600 arbres à taux 0.02
  contre jusqu'à 6 000 à 0.01), **pas du masque** (masques identiques sur ces données, prédictions identiques).
  Le recalage power_init joue peu et change de signe (+0.0006 en directe, −0.001 ailleurs ; il corrige surtout le
  premier mois et le biais de la chrono).
- **En inverse, c'est une combinaison** : chaque facteur seul est petit (entrées +0.0015, réglages +0.0011, au
  niveau de l'effet d'environnement) et l'interaction domine (+0.0037) ; entrées recette avec D3L font jeu égal
  avec le livré réentraîné (0.2243 contre 0.2247).
- Statut : reproduction (contrôles, recette stricte) ; tout le reste est exploratoire sur des validations déjà très
  utilisées ; rien ici ne prouve une généralisation, et aucun gain n'est à transférer.

## Pistes proposées (hors de cette reproduction, non lancées)

- Ajouter aux entrées actuelles les seuls blocs de la recette absents du livré (par exemple `ev_gs_sum`,
  `ev_n_events`, `dm_act_*` sans plafond de durée), un bloc à la fois, réglage D3L figé : test d'apport
  d'information, pas de remplacement.
- Mesurer l'effet d'environnement sur le livré lui-même (D3L, env principal contre épinglé) avant d'interpréter des
  écarts de 0.001.

## Reproduction

```bash
uv run python outils/recette_b.py geler
uv run python outils/recette_b.py controle            # et avec l'env épinglé : ENV/bin/python -I outils/recette_b.py controle
uv run python outils/recette_b.py bords
uv run python outils/lancer_kaggle.py --fente 1 --legers -- outils/recette_b.py matrices --sortie /tmp/b_mat --session k1 ";" \
  outils/recette_b.py entrainer --matrices /tmp/b_mat --cellules rec_recette_coll --prefixe b3k ";" \
  outils/recette_b.py entrainer --env-recette --matrices /tmp/b_mat --cellules rec_recette_coll rec_recette_propre cur_recette_coll rec_d3l_coll --bord
uv run python outils/lancer_kaggle.py --fente 2 --legers -- outils/recette_b.py matrices --sortie /tmp/b_mat --session k2 ";" \
  outils/recette_b.py entrainer --env-recette --matrices /tmp/b_mat --cellules cur_d3l_coll
uv run python outils/recette_b.py evaluer
```

Code : `baselines/recette_b.py`, `outils/recette_b.py`, `controles/recette_b/` ; `outils/bilan_2551.py` (mois,
`SPECIALISTE`). Tests : `tests.test_recette_b` (7) avec `test_gbm_sacs`, `test_regime_2551`, `test_ensemble`,
`test_caracteristiques`, `test_recette` : 49 OK (1 ignoré : xgboost absent en local). Journaux Kaggle :
`runs/_kaggle/20261114_195611-2`, `runs/_kaggle/20261114_200847-1`.

# Programme 5 (14–15 novembre 2026) : l'erreur infra-horaire

Angle : tout ce qui précède agrège à l'heure avant de modéliser, alors que 61 à 76 % de l'erreur de 2551 est
horaire. Le programme lit le courant brut (32 s), en tire des cibles plus fines, et comble les trous de variables
relevés dans le boosting. Règle d'installation inchangée (`bilan_2551.regle`, porte chronologique +0.0002), poids de
mélange fixés avant mesure (remplacement de la part S, soit 0.5 ; part 0.175 pour un membre du GRU). Tous les
entraînements sur Kaggle `zendarc148`, T4, 6 sessions.

Référence (RMSE 2551) : directe 0.2583 / 0.2572, inverse 0.2054 / 0.2046, chrono 0.2557 ; global 0.076735.

## Phase 1 — Diagnostic brut (`python outils/diag_fin_2551.py`, `runs/p5_diag/diag.json`, local)

- Le courant brut de 2551 est une somme de lignes de chauffage tout-ou-rien : A ≈ 0.205 (deux exemplaires),
  B ≈ 0.63, C ≈ 1.045. Décodage à 0.009 près (RMSE du reste).
- Parts de la variance horaire : C 59–60 %, B 32–34 %, A 8 %.
- Créneaux : 5.3 min allumé, 6.4 min éteint (médianes), environ 5 cycles par heure. Le bruit horaire vient de
  rafales d'activité, pas du découpage d'un cycle régulier par l'heure.
- Repli sur la phase orbitale exacte, jours hauts, part de variance intra-jour expliquée : 21.5 % (annee1) et
  17.0 % (annee2) par cases de 10 min, contre 17.8 % et 15.1 % par cases de 60 min. Écart : +3.7 et +1.9 points,
  sous le seuil de 5 fixé au plan. Conséquence : réseau séquentiel à 10 min (phase 4b) non lancé ; boosting à
  10 min lancé une fois.

## Phase 2 — Variables manquantes du boosting (`outils/programme5.py variables`, sessions 1 et 2)

Sélection en réglage D3, un seul modèle (comme le programme 3), validation interne 5 blocs. Boosting seul,
directe / inverse / chrono :

| Jeu | Externe | Lecture |
|---|---|---|
| base (S en D3) | 0.2832 / 0.2257 / 0.2820 | identique à `runs/p3_var_base` |
| lentes : EMA 336 h et 720 h rétablies | 0.2690 / 0.2288 / 0.2677 | finaliste |
| thermique : filtres du premier ordre, 6 h à 60 j | 0.2653 / 0.2253 / 0.2688 | finaliste |
| orbite_exacte : phase des péricentres exacts | 0.2874 / 0.2268 / 0.2861 | nul |
| evtf_plus : EVTF jamais lus | 0.3334 / 0.2272 / 0.3726 | nuisible : angle zénithal et temps lumière servent d'horloge |
| blocs_b : blocs de la recette B absents du livré | 0.2828 / 0.2254 / 0.2816 | neutre |
| saison d'éclipses | 0.2858 / 0.2270 / 0.2709 | directe en hausse |

Défaut corrigé pour ces essais : `retirer: ["ema"]` supprimait les EMA 336 h et 720 h créées par `orbite` ; les
filtres lents portent un autre nom (`lent_*_tau*`, `th_*_tau*`) et survivent au retrait.

Finalistes réentraînés dans la forme exacte du spécialiste (D3L, 5 sacs, `outils/specialistes.py entrainer`,
session 4), en remplacement de S :

| Candidat | Seul | Mélange : directe g04 / g59, inverse g04 / g59, chrono | Verdict |
|---|---|---|---|
| S livré | 0.2678 / 0.2229 / 0.2641 | 0.2583 / 0.2572 / 0.2054 / 0.2046 / 0.2557 | |
| thermique | 0.2627 / 0.2202 / 0.2752 | 0.2550 / 0.2541 / 0.2042 / 0.2036 / 0.2590 | refusé : chrono +0.0033, MAE en hausse |
| lentes | 0.2624 / 0.2207 / 0.2715 | 0.2556 / 0.2548 / 0.2045 / 0.2037 / 0.2585 | refusé : chrono +0.0028, MAE en hausse |
| thermique, part 0.25 | — | 0.2563 / 0.2553 / 0.2046 / 0.2038 / 0.2569 | refusé : chrono +0.0012 |

## Phase 3 — Cibles fines (`baselines/fin_2551.py`, cibles dans `data/prepared/p5/`)

Mélange : livré + 0.5 × (candidat − base D3).

| Candidat | Seul | Mélange | Verdict |
|---|---|---|---|
| lignes : un boosting par ligne de chauffage, somme | 0.2887 / 0.2223 / 0.2748 | refusé | directe en hausse |
| lignes + thermique | 0.2695 / 0.2182 / 0.2689 | 0.2556 / 0.2549 / 0.2033 / 0.2027 / 0.2598 | refusé : chrono +0.0041 |
| fin : boosting sur grille de 10 min, moyenne par heure | 0.2732 / 0.2229 / 0.2753 | 0.2572 / 0.2563 / 0.2045 / 0.2037 / 0.2637 | refusé : chrono +0.0080 |

Erreur interne par ligne (directe) : A 0.034, B 0.088, C 0.160. Écart au plan : la variante « taux d'allumage ×
amplitude » est remplacée par « un boosting par ligne », l'amplitude de chaque ligne étant constante.

## Phase 4 — Modèles sur GPU

- 4a, GRU avec entrées 2551 (`ajouts_2551` dans `dio/monde.py`, 5 graines, session 5), part 0.175 à la place du
  membre v2r : `echelles + saison + thermique` 0.2647 / 0.2635 / 0.2112 / 0.2116 / 0.2591 ; avec `etats` et
  `commutateur` 0.2623 / 0.2614 / 0.2084 / 0.2087 / 0.2594. Pire dans les 5 cas. Écart au plan : 5 graines au lieu
  de 10.
- 4b : non lancé (seuil de la phase 1).
- 4c, thermostat différentiable (`baselines/thermostat.py`, 54 paramètres, session 3) : seul 0.3436 / 0.2985 /
  0.3521 ; comme variable de S (prédictions hors bloc) 0.3000 / 0.2247 / 0.3019 contre 0.2832 / 0.2257 / 0.2820
  pour la base. Refusé. Écart au plan : 54 paramètres au lieu de moins de 30.

## Phase 5 — LightGBM et CatBoost (`outils/programme5.py arbres`, session 6)

Un modèle par bibliothèque, arbres par arrêt anticipé sur le bloc écarté (écart au plan : pas de 20 sacs).

| Candidat | Seul | Mélange | Verdict |
|---|---|---|---|
| LightGBM | 0.2822 / 0.2220 / 0.2763 | 0.2591 / 0.2580 / 0.2061 / 0.2052 / 0.2592 | refusé |
| CatBoost | 0.2959 / 0.2225 / 0.2738 | 0.2644 / 0.2631 / 0.2069 / 0.2060 / 0.2616 | refusé |
| LightGBM + thermique | 0.2720 / 0.2146 / 0.2717 | 0.2578 / 0.2570 / 0.2027 / 0.2019 / 0.2616 | refusé : chrono +0.0059 |
| CatBoost + thermique | 0.2693 / 0.2176 / 0.2706 | 0.2580 / 0.2572 / 0.2046 / 0.2038 / 0.2633 | refusé : chrono +0.0076 |

## Décision

Aucun candidat ne passe la règle : rien d'installé, modèle livré inchangé (0.076735). Phase 7 (entraînement final,
essai sur NPWD2451) sans objet. Mélanges de deux candidats non calculés : chaque candidat dégrade la chrono de
+0.0012 à +0.0080 dans le mélange, la porte (+0.0002) ne peut pas être passée par leur moyenne.

## Conclusion

- L'information brute existe (lignes décodées, grille de 10 min) et améliore le boosting seul à réglage égal, mais
  le gain ne se transmet pas au mélange en chronologique. Tous les candidats ont le même profil : directe −0.001 à
  −0.003, inverse −0.001 à −0.003, chrono en hausse.
- L'état thermique est le meilleur apport en directe et en inverse (4 cas sur 5 en baisse), mais il se dégrade
  en chrono : les filtres longs (jusqu'à 60 jours) portent une position dans l'année que le découpage
  chronologique ne recouvre pas. Lecture non testée séparément.
- Statut : exploratoire, sur des validations déjà très utilisées. Aucun gain à transférer.

## Modèles gardés et limites

- `poids_candidats/npwd2551/INDEX.md` : liste des candidats, scores, verdict. Poids complets gardés pour
  `thermique` et `lentes` (D3L 5 sacs, 3 directions).
- Les autres essais ne sauvegardent pas le modèle : prédictions, prédictions hors bloc et recette restent dans
  `runs/p5*` (rejouables). Les poids des deux GRU n'ont pas été rapatriés (`--legers`).
- Kaggle : une session est restée 4 h en file (`runs/p5_logs/s3_file_bloquee.log`), relance refusée (« Maximum
  batch GPU session count of 2 reached »), puis elle a tourné. 6 sessions consommées sur 8.

## Reproduction

```bash
uv run python outils/diag_fin_2551.py
uv run python outils/programme5.py preparer            # data/prepared/p5/, puis lancer_kaggle --donnees une fois
uv run python outils/lancer_kaggle.py --fente 1 --legers -- outils/programme5.py variables --nom p5v --jeux '{"base": {}, "thermique": {"ajouts": ["thermique"]}}'
uv run python outils/lancer_kaggle.py --fente 1 --legers -- outils/programme5.py thermo --nom p5_thermo ";" outils/programme5.py fin --nom p5_fin
uv run python outils/lancer_kaggle.py --fente 2 --legers -- outils/programme5.py arbres --nom p5a --modeles lgb cat
uv run python outils/programme5.py bilan --nom p5_bilan_s4 --candidats thermique=p5s_thermique lentes=p5s_lentes
```

Code : `outils/diag_fin_2551.py`, `baselines/fin_2551.py`, `baselines/thermostat.py`, `outils/programme5.py`,
ajouts dans `baselines/gbm_sacs.py` et `dio/monde.py`. Tests : `tests.test_fin_2551` (6). Bilans :
`runs/p5_bilan_s{1,2,3,4,5,6}/bilan.json`.

## Complément (15 novembre 2026) : trois spécialistes entraînés sur annee1 + annee2

Demande : réentraîner sur années 1 + 2 la référence (S livré), `thermique` et `lentes`, même réglage D3L, 5 sacs,
graine 0 ; seuls les ajouts de variables changent. Poids gardés à part, rien d'installé, aucun autre circuit touché.

- Réglage : options de `runs/final_gbm_2551/gbm_NPWD2551/modele.json` (entraînement qui a produit le livré),
  `compresser: true` ; `ajouts` + `thermique` ou + `lentes`. La configuration de la référence est identique à
  celle du livré (comparaison JSON).
- Kaggle `zendarc148`, T4 (`cuda Tesla T4` dans les deux journaux), deux sessions en parallèle : fente 1
  référence puis lentes (`runs/_kaggle/20261115_161501-1`), fente 2 thermique (`runs/_kaggle/20261115_154617-2`).
- Poids, configurations et journaux : `poids_candidats/npwd2551/annees12/{reference,thermique,lentes}/`.
  Tableau (variables, arbres par sac, durées, hors sac interne) : `poids_candidats/npwd2551/INDEX.md`.

| | Variables | Durée | Hors sac interne RMSE / MAE |
|---|---:|---:|---|
| livré | 973 | 1948 s (CPU) | 0.2220 / 0.1458 |
| référence | 973 | 1646 s | 0.2233 / 0.1469 |
| thermique | 1029 | 1661 s | 0.2171 / 0.1459 |
| lentes | 989 | 1730 s | 0.2207 / 0.1451 |

Contrôles locaux (prédiction seule) : les trois paquets se chargent et prédisent `validation/test` (16 319 heures,
valeurs finies, ≥ 0). Aucun score : ces heures font partie de l'entraînement. La référence ne reproduit pas le
livré au bit près : mêmes 973 colonnes et mêmes variables de test, mais l'arrêt anticipé diffère dans 2 sacs sur 5
(5612 et 4730 arbres contre 5227 et 6000). Écart de prédiction moyen 0.0025, max 0.025. Cause probable : matériel
(livré entraîné en session CPU, Python 3.13.18 ; ici session T4, Python 3.13.16 ; scikit-learn 1.9.1 dans les
deux cas). Non vérifié plus avant, ce qui demanderait un nouvel entraînement. Le hors sac est indicatif (blocs
sans embargo sur les années d'entraînement) et ne remplace pas la porte chronologique qui a refusé les deux
variantes.

```bash
OPTS='"config": "D3L", "jeu": "V2", "compresser": true, "familles_variante": true, "plan": {"changements": true, "recence": "debut", "prefixes": true, "paires": "liste"}, "contexte": {"soleil": "interpole", "ltdata": "lineaire"}, "embargo_h": 0, "filtre": false, "retirer": ["lettres", "distance_terre", "ema"], "circuits": ["NPWD2551"]'
AJ='"planning", "ema", "geometrie", "altitude", "ombres", "contacts", "decalages", "histoires", "face_sa_moins", "orbite"'
uv run python outils/lancer_kaggle.py --fente 1 --sans-attendre -- \
  train.py --modele gbm_sacs --annees annee1 annee2 --out runs/f12_2551_reference/gbm_NPWD2551 --options "{$OPTS, \"ajouts\": [$AJ]}" ";" \
  train.py --modele gbm_sacs --annees annee1 annee2 --out runs/f12_2551_lentes/gbm_NPWD2551 --options "{$OPTS, \"ajouts\": [$AJ, \"lentes\"]}"
uv run python outils/lancer_kaggle.py --fente 2 --sans-attendre -- \
  train.py --modele gbm_sacs --annees annee1 annee2 --out runs/f12_2551_thermique/gbm_NPWD2551 --options "{$OPTS, \"ajouts\": [$AJ, \"thermique\"]}"
uv run python outils/lancer_kaggle.py --fente 1 --rapatrier
uv run python outils/lancer_kaggle.py --fente 2 --rapatrier
```
