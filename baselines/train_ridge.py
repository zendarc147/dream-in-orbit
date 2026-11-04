"""Entraîner un Ridge par circuit avec les données préparées d'annee1."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def entrainer(racine, sortie=None):
    racine = Path(racine)
    prepare = racine / "data/prepared"
    schema = json.loads((prepare / "schema.json").read_text())
    variables_entree = schema["feature_columns"]
    circuits = schema["target_columns"]
    x = pd.read_csv(prepare / "annee1/X.csv")
    y = pd.read_csv(prepare / "annee1/y.csv")
    if (len(variables_entree) != 65 or len(circuits) != 33
            or x.columns.tolist() != ["ut_ms"] + variables_entree
            or y.columns.tolist() != ["ut_ms"] + circuits):
        raise ValueError("Colonnes X/y incompatibles avec le schéma préparé")
    if (x.ut_ms.isna().any() or not x.ut_ms.is_unique
            or not x.ut_ms.equals(y.ut_ms)):
        raise ValueError("Les heures X/y d'apprentissage doivent être alignées et uniques")

    # Couverture quasi constante sur annee1 : son échelle amplifie les absences de télémétrie.
    variables = [c for c in variables_entree if c != "saaf_coverage_fraction"]
    entrees = x[variables_entree].to_numpy(dtype=float)
    cibles = y[circuits].to_numpy(dtype=float)
    if not np.isfinite(entrees).all() or np.isinf(cibles).any():
        raise ValueError("X doit être fini et y ne doit pas contenir d'infini")

    entrees = x[variables].to_numpy(dtype=float)
    scaler = StandardScaler()
    entrees = scaler.fit_transform(entrees)
    coefficients = []
    constantes = []
    effectifs = []
    for i, circuit in enumerate(circuits):
        connus = np.isfinite(cibles[:, i])
        if not connus.any():
            raise ValueError(f"Aucune cible connue pour {circuit}")
        modele = Ridge(alpha=1.0)
        modele.fit(entrees[connus], cibles[connus, i])
        coefficients.append(modele.coef_.tolist())
        constantes.append(float(modele.intercept_))
        effectifs.append(int(connus.sum()))

    modele = {
        "alpha": 1.0,
        "input_feature_columns": variables_entree,
        "feature_columns": variables,
        "target_columns": circuits,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "intercept": constantes,
        "coef": coefficients,
        "n_observations": effectifs,
    }
    sortie = Path(sortie) if sortie else racine / "ridge_model.json"
    sortie.write_text(json.dumps(modele, indent=2, allow_nan=False) + "\n")
    print(f"{len(circuits)} Ridge entraînés sur annee1, {len(variables)} variables utilisées"
          f" sur {len(variables_entree)} ; modèle : {sortie.name}")
    return sortie


if __name__ == "__main__":
    ici = Path(__file__).resolve().parent
    entrainer(ici.parent, ici / "ridge_model.json")
