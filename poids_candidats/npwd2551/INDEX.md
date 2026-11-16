# Candidats NPWD2551 — programme 5

Aucun modèle supprimé. Référence livrée (RMSE 2551) : directe 0.2583 / 0.2572, inverse 0.2054 / 0.2046,
chrono 0.2557. Règle : `outils/bilan_2551.regle` + porte chronologique (pas plus de +0.0002).

Colonnes « seul » : boosting seul, directe / inverse / chrono. Colonnes « mélange » : livré avec la part S
remplacée (0.5), directe g04 / g59, inverse g04 / g59, chrono.

## Poids gardés ici

| Nom | Dossier | Forme | Seul | Mélange | Verdict |
|---|---|---|---|---|---|
| thermique | `thermique/` | D3L 5 sacs, S + état thermique (dont EMA lentes) | 0.2627 / 0.2202 / 0.2752 | 0.2550 / 0.2541 / 0.2042 / 0.2036 / 0.2590 | refusé : chrono +0.0033, MAE en hausse |
| lentes | `lentes/` | D3L 5 sacs, S + EMA 336 h et 720 h rétablies | 0.2624 / 0.2207 / 0.2715 | 0.2556 / 0.2548 / 0.2045 / 0.2037 / 0.2585 | refusé : chrono +0.0028, MAE en hausse |

Journal Kaggle de la session : `thermique/session4.log`.

## Entraînés sur annee1 + annee2 (`annees12/`, 15 novembre 2026)

Forme finale, comme `runs/final_gbm_2551` qui a produit le livré : `train.py --modele gbm_sacs --annees annee1
annee2`, D3L, 5 sacs, graine 0, mêmes options ; seul `ajouts` change (+ `thermique` ou + `lentes`). Kaggle
`zendarc148`, T4. Chaque dossier : `gbm_sacs.pkl.xz`, `modele.json` (configuration complète), `colonnes.json`,
`iterations.json`, `hors_sac.csv`, `session.log` (journal Kaggle). Rien d'installé : `poids/` inchangé.

| Nom | Dossier | Variables | Arbres par sac | Durée | Hors sac interne (RMSE / MAE) |
|---|---|---:|---|---:|---|
| référence (livré réentraîné) | `annees12/reference/` | 973 | 6000, 5612, 6000, 4730, 6000 | 1646 s | 0.2233 / 0.1469 |
| thermique | `annees12/thermique/` | 1029 | 6000, 6000, 6000, 5650, 6000 | 1661 s | 0.2171 / 0.1459 |
| lentes | `annees12/lentes/` | 989 | 6000, 5364, 6000, 6000, 6000 | 1730 s | 0.2207 / 0.1451 |
| livré (pour mémoire) | `poids/specialiste/gbm_NPWD2551/` | 973 | 6000, 5227, 6000, 6000, 6000 | 1948 s | 0.2220 / 0.1458 |

Hors sac : chaque bloc temporel prédit par le sac qui ne l'a pas vu, sans embargo, sur les années
d'entraînement ; indicatif seulement (pas une validation, les filtres longs traversent les bords de blocs).
La référence n'est pas identique au bit près au livré : même configuration, mêmes 973 colonnes, mêmes variables
de test, mais arrêt anticipé différent dans 2 sacs. Le livré a été entraîné dans une session Kaggle CPU (Python
3.13.18), celle-ci dans une session T4 (Python 3.13.16), même scikit-learn 1.9.1. Écart des prédictions sur
`validation/test` : moyen 0.0025, max 0.025.

## Essais de sélection sans poids sauvegardés

Réglage D3, un seul modèle, validation interne (`recherche_2551.cv_fige`) : seules les prédictions, les
prédictions hors bloc et la recette (`choix.json`) sont gardées, dans `runs/`. Rejouables à l'identique
(`outils/programme5.py variables` ou `lignes`).

| Nom | Dossier | Seul | Verdict |
|---|---|---|---|
| base (S en D3) | `runs/p5v_base` | 0.2832 / 0.2257 / 0.2820 | référence de sélection |
| lentes | `runs/p5v_lentes` | 0.2690 / 0.2288 / 0.2677 | finaliste |
| thermique | `runs/p5v_thermique` | 0.2653 / 0.2253 / 0.2688 | finaliste |
| orbite_exacte | `runs/p5v_orbite_exacte` | 0.2874 / 0.2268 / 0.2861 | refusé : nul |
| evtf_plus | `runs/p5v_evtf_plus` | 0.3334 / 0.2272 / 0.3726 | refusé : nuisible |
| blocs_b | `runs/p5v_blocs_b` | 0.2828 / 0.2254 / 0.2816 | refusé : neutre |
| saison | `runs/p5v_saison` | 0.2858 / 0.2270 / 0.2709 | refusé : directe en hausse |
| lignes (un boosting par ligne de chauffage) | `runs/p5_lignes` | 0.2887 / 0.2223 / 0.2748 | refusé : directe en hausse |
| lignes + thermique | `runs/p5_lignes_th` | 0.2695 / 0.2182 / 0.2689 | refusé : chrono +0.0041 en mélange |
| fin (boosting sur grille de 10 min) | `runs/p5_fin` | 0.2732 / 0.2229 / 0.2753 | refusé : chrono +0.0080 en mélange |
| thermo (thermostat différentiable, 54 paramètres) | `runs/p5_thermo` | 0.3436 / 0.2985 / 0.3521 | refusé : pire partout |
| thermo_s (thermostat comme variable de S) | `runs/p5_thermo_s` | 0.3000 / 0.2247 / 0.3019 | refusé : pire partout |
| lgb (LightGBM) | `runs/p5a_lgb` | 0.2822 / 0.2220 / 0.2763 | refusé |
| cat (CatBoost) | `runs/p5a_cat` | 0.2959 / 0.2225 / 0.2738 | refusé |
| lgb + thermique | `runs/p5a_th_lgb` | 0.2720 / 0.2146 / 0.2717 | refusé : chrono +0.0059 en mélange |
| cat + thermique | `runs/p5a_th_cat` | 0.2693 / 0.2176 / 0.2706 | refusé : chrono +0.0076 en mélange |

## GRU avec entrées 2551 (poids non rapatriés, prédictions et journaux gardés)

Part 0.175 à la place du membre v2r du GRU livré. Mélange : directe g04 / g59, inverse g04 / g59, chrono.

| Nom | Dossier | Mélange | Verdict |
|---|---|---|---|
| gru_th (echelles, saison, thermique) | `runs/p5_gru_th` | 0.2647 / 0.2635 / 0.2112 / 0.2116 / 0.2591 | refusé : pire dans les 5 cas |
| gru_etats (+ etats, commutateur) | `runs/p5_gru_etats` | 0.2623 / 0.2614 / 0.2084 / 0.2087 / 0.2594 | refusé : pire dans les 5 cas |

Bilans complets : `runs/p5_bilan_s{1,2,3,4,5,6}/bilan.json`. Détail : `docs/journaux/JOURNAL_2551.md`, programme 5.
