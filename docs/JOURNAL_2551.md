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
