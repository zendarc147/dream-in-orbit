# Candidats de structure — 16 novembre 2026

Poids complets entraînés sur annee1 + annee2, un dossier par candidat, installé ou non (`python outils/structure.py
finaux`). Base = livré (GRU mixte, G2, R, D3L, physique, marqueurs de `poids_repli/`) ; GRU S borné, S et B : Kaggle
`zendarc148` S4 (`runs/st_final_*`). NPWD2532 et NPWD2531 : spécialistes du livré partout. Chaque dossier se lit par
`python predict.py --poids poids_candidats/structure/<nom>` (vérifié pour les 10 : 3953 × 33 valeurs finies ≥ 0).
Règle, mesures et limites : `docs/JOURNAL_STRUCTURE.md`. Écarts au livré : Δ moyenne des 4 RMSE temporelles,
Δ moyenne des coupures glissantes, Δ pire coupure (c87).

| Nom | Structure | Δ moy. 4 | Δ moy. coupures | Δ c87 | Verdict |
|---|---|---:|---:|---:|---|
| `L/` | livré : 0.5 × (0.7 GRU mixte + 0.3 G2) + 0.5 × R ; D3L pour NPWD2551 | 0 | 0 | 0 | livré précédent, gardé en repli (`poids_repli/`) |
| `C1_g5/` | 0.5 × (0.5 GRU mixte + 0.5 GRU S) + 0.5 × R ; D3L pour NPWD2551 (5 graines de GRU S) | -0.000447 | +0.000597 | +0.003125 | **installé dans `poids/`** |
| `T1_g10/` | 0.5 × GRU S + 0.5 × R ; D3L pour NPWD2551 (10 graines de GRU S) | +0.000037 | +0.001887 | +0.008996 | refusé : chaque coupure <= +0.0040 |
| `T1_g5/` | 0.5 × GRU S + 0.5 × R ; D3L pour NPWD2551 (5 graines de GRU S) | +0.000067 | +0.001918 | +0.009074 | refusé : chaque coupure <= +0.0040 |
| `T2_g10/` | 0.5 × GRU S + 0.5 × (R + S)/2 ; NPWD2551 : (D3L + B)/2 dans la part boosting (10 graines de GRU S) | +0.000254 | +0.001948 | +0.008568 | refusé : chaque coupure <= +0.0040 |
| `T2_g5/` | 0.5 × GRU S + 0.5 × (R + S)/2 ; NPWD2551 : (D3L + B)/2 dans la part boosting (5 graines de GRU S) | +0.000284 | +0.001979 | +0.008647 | refusé : chaque coupure <= +0.0040 |
| `T3_g10/` | 0.5 × GRU S + 0.5 × S ; NPWD2551 : B seule dans la part boosting (10 graines de GRU S) | +0.000790 | +0.002273 | +0.008365 | refusé : chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| `T3_g5/` | 0.5 × GRU S + 0.5 × S ; NPWD2551 : B seule dans la part boosting (5 graines de GRU S) | +0.000819 | +0.002303 | +0.008444 | refusé : chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| `T3sr_g10/` | comme T3, prédiction sans recalage de la part boosting (`options_prediction` de `modele.json`) (10 graines de GRU S) | +0.001136 | +0.002397 | +0.008152 | refusé : moyenne_4 <= +0.0010 ; chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| `T3sr_g5/` | comme T3, prédiction sans recalage de la part boosting (`options_prediction` de `modele.json`) (5 graines de GRU S) | +0.001165 | +0.002427 | +0.008232 | refusé : moyenne_4 <= +0.0010 ; chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
