# Poids finaux (annee1 + annee2) du programme « GRU à boucle » (`docs/journaux/JOURNAL_GRU_BOUCLE.md`)

Base de chaque dossier : copie indépendante (aucun lien dur) de `poids_candidats/structure/T3_g10/` (part boosting S,
recette B pour NPWD2551, spécialistes NPWD2532 et NPWD2531, G0 dans `recette/`), plus `recette2/recette.pt` = GB
final (`runs/gb_final_GB_g04` + `_g59` fusionnés, Kaggle `zendarc147` session K3, 680 et 676 s sur T4).
Prédiction : `python predict.py --test_dir … --out … --poids poids_candidats/gru_boucle/<nom>`.

| Dossier | Part réseau | Poids boosting | Moy. directe / inverse | Moy. coupures | Verdict |
|---|---|---:|---:|---:|---|
| (T3 = S0, `../structure/T3_g10/`) | G0 seul | 0.5 | 0.077525 | 0.074158 | ancien livré, gardé |
| S1 | GB seul (`recette/` absent) | 0.5 | 0.077845 | 0.073759 | non installé (hors liste d'installation) |
| **S2** | 0.5 G0 + 0.5 GB | 0.5 | 0.077534 | 0.073709 | **installé dans `poids/`** (règle : aucun échec technique) |
| S2_b06 | 0.5 G0 + 0.5 GB | 0.6 | 0.077566 | 0.073714 | variante de rapport, jamais installée sans accord de l'utilisateur |
| GBmm_final | paquet GRU seul (`recette.pt`) : GB avec normalisation min/max | — | — | — | diagnostic, gardé (689 et 683 s sur T4) |

GB-r n'a pas de dossier : mêmes poids que GB, prédiction avec `"recalage_recette2": 1.0` dans `ensemble.json`.
Paquets GRU de chaque validation : `runs/gb_<directe|inverse|c62|c75|c87>_<GB|GBmm>_<g04|g59|g10>/`.
