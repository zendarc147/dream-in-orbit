# Journal — recette complète « S » (à partir du 12 novembre 2026, soir)

Programme en 10 points fourni par l'utilisateur : référence figée, recette de features exacte et distincte de la
préparation actuelle (`dio/caracteristiques.py`), GBM et GRU à entrées séparées, recette complète puis ablations,
complémentarité, choix. Tout est calculé **sur la machine locale** (demande de l'utilisateur pour cette session :
pas de Kaggle). Branche locale `recette-complete` (depuis `recette`), rien n'est poussé.

## Reprendre ici

- Branche : `recette-complete`.
- Jobs : voir `runs/rc_lot*.sh` / `.log` / `.pid` (attente par `kill -0 PID`).
- Statut : voir « État » en bas.

## Matériel et temps

AMD Ryzen AI 7 350 (8 cœurs, 16 threads), 30 Go, pas de GPU utilisable. Python 3.13, torch 2.15.0 (CPU),
scikit-learn 1.9.1, pandas 2.3.3, numpy 2.3.5 (liste complète : `runs/rc0_reference/manifeste.json`).
Les temps de cette session sont tous mesurés sur cette machine ; ceux des journaux précédents viennent de Kaggle.

## Critères de décision (fixés avant les mesures)

- Ablations et composants, sans spécialistes : règle du dépôt (`docs/JOURNAL_RECETTE.md`), référence = recette complète S.
  (a) moyenne des RMSE du mélange en baisse ≥ 0.0003 et aucune direction en hausse > 0.0002 ;
  (b) composant changé seul en baisse dans les deux directions et mélange ≤ +0.0001.
  Mélange des ablations : 0.5 × GBM + 0.5 × GRU, poids fixe. Outil : `outils/comparer_complete.py`.
- Installation (modèle complet avec spécialistes, `predict.py` → `evaluate.py`, 4 RMSE directe/inverse × graines
  0–4/5–9) contre le livré 0.076735 : règle (a) ou les 4 RMSE en baisse ; et validation chronologique pas pire que la
  référence de plus de 0.0002. Sinon, le modèle livré reste.
- Poids et parts choisis sur les graines 0–4, confirmés sur 5–9.

## Point 1 — référence figée

- `runs/rc0_reference/manifeste.json` : commit `070ae7f`, sha256 des 60 sources suivies, des 13 fichiers de `poids/`,
  de 18 prédictions de référence, bibliothèques (`uv pip freeze`), machine, configurations (ensemble, modèle,
  spécialistes).
- **Inférence du livré reproduite en local** (`runs/rc0_livre_g04|g59`, copies de `s51_val_*`,
  `valider.py --sans-entrainement`) :

| | Directe | Inverse | Prédiction (s, local) |
|---|---:|---:|---|
| graines 0–4 | 0.075046 | 0.078474 | 46 / 51 |
| graines 5–9 | 0.075078 | 0.078343 | 44 / 54 |
| **Moyenne des 4** | | **0.076735** | identique au journal |

- Référence sans spécialistes, mélange 0.5 × GBM R + 0.5 × GRU mixte sans lettres (sans GRU recette) : 0.077971 ;
  installé sans spécialistes (avec GRU recette 0.3) : 0.07768 (`docs/JOURNAL_RECETTE.md`).

## Écarts formule par formule entre la spec et le boosting R (avant ce programme)

| Spec | R (`gbm_sacs` + `contexte`) |
|---|---|
| SAAF : moyennes arithmétiques, std pandas ddof=1, NaN→0 si heure observée | moyennes pondérées par la durée (maintien 60 s), écart-type de population pondéré |
| LTDATA au début de l'heure | milieu de l'heure |
| récences : occurrences quantifiées à l'heure | instants exacts (fractions d'heure) |
| changements : ≥ 2 variantes à ≥ 50 par période, union ; quantifiés ; 0 dans l'heure | seuil 20 sur l'ensemble ; instants exacts ; pas de 0 |
| paires : union des intervalles | somme plafonnée à 1 |
| altitude : moyenne des 4 log | moyenne des 4 altitudes |
| historiques : umbra seule, flux_incident = solar × shade | umbra + 0.5 pénombre, somme des flux des faces |
| décalages : 10 séries (dont log-altitude) | 9 séries |
| vocabulaires de chaque validation appris sur son entraînement | schéma `prepare.py` (types FTL, 10 commandes, médianes) appris sur annee1, aussi en inverse |
| pas de batteries glissantes ni EMA | fenêtres v2r 3–24 h passées/futures, moyennes `physiques` 6/24/72 h |

Conformes et réutilisés : `commandes.famille_variante`, réglage HGB « D » (= spec), 5 sacs contigus sans embargo ni
filtre, géométrie aux péricentres exacts (`contexte.geometrie`), garde-fous d'appariement (`contexte.intervalles`,
contacts au seuil `_10` seulement), architecture et boucle d'entraînement de `dio/recette.py`.

## Point 2 à 6 — `dio/caracteristiques.py`

Conventions : docstring du module (grille, SAAF, LTDATA, familles, récences, changements, paires, événements, FTL,
géométrie, altitude, thermique, décalages). Tests : `tests/test_caracteristiques.py` (11 tests, exemples de la spec).
Vocabulaire appris sur annee1 + annee2 (pour information) : 149 familles, 23 préfixes, 29 familles à changements,
16 types FTL. Construction d'une année : environ 2 s.
