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
