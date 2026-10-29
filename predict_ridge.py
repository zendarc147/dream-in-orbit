"""Prédire les courants depuis un X.csv déjà produit par prepare.py."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def predire(x_path, out, model_path=None):
    if model_path is None:
        model_path = Path(__file__).resolve().parent / "ridge_model.json"
    modele = json.loads(Path(model_path).read_text())
    variables = modele["feature_columns"]
    variables_entree = modele.get("input_feature_columns", variables)
    circuits = modele["target_columns"]
    x = pd.read_csv(x_path)
    if (len(variables_entree) != 65 or len(circuits) != 33 or not variables
            or variables != [c for c in variables_entree if c in variables]
            or x.columns.tolist() != ["ut_ms"] + variables_entree):
        raise ValueError("Colonnes X incompatibles avec le modèle Ridge")
    heures = x.ut_ms
    if (heures.isna().any() or not pd.api.types.is_integer_dtype(heures)
            or not heures.is_unique or (heures % 3_600_000 != 0).any()):
        raise ValueError("Les heures doivent être entières, uniques et alignées sur UTC")

    entrees = x[variables].to_numpy(dtype=float)
    moyenne = np.asarray(modele["mean"], dtype=float)
    echelle = np.asarray(modele["scale"], dtype=float)
    coefficients = np.asarray(modele["coef"], dtype=float)
    constantes = np.asarray(modele["intercept"], dtype=float)
    if (moyenne.shape != (len(variables),) or echelle.shape != (len(variables),)
            or coefficients.shape != (33, len(variables)) or constantes.shape != (33,)
            or not np.isfinite(x[variables_entree].to_numpy(dtype=float)).all()
            or not np.isfinite(moyenne).all()
            or not np.isfinite(echelle).all() or (echelle <= 0).any()
            or not np.isfinite(coefficients).all() or not np.isfinite(constantes).all()):
        raise ValueError("Entrées ou paramètres Ridge invalides")
    valeurs = ((entrees - moyenne) / echelle) @ coefficients.T + constantes
    if not np.isfinite(valeurs).all():
        raise ValueError("Prédictions Ridge non finies")
    predictions = pd.DataFrame(valeurs, columns=circuits)
    predictions.insert(0, "ut_ms", heures)
    predictions.to_csv(out, index=False)
    print(f"{len(predictions)} heures prédites dans {out}")
    return predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", required=True, type=Path, help="X.csv préparé pour les heures à prédire")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    predire(args.x, args.out)


if __name__ == "__main__":
    main()
