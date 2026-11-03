"""Prédire un courant constant par circuit, sans lire de consommation de test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def predire(test_dir, out):
    chemin = Path(__file__).resolve().parent / "baseline_constante.json"
    constantes = json.loads(chemin.read_text())
    if (len(constantes) != 33 or not all(c.startswith("NPWD") for c in constantes)
            or not np.isfinite(list(constantes.values())).all()):
        raise ValueError("33 constantes NPWD finies sont requises")
    heures = pd.read_csv(Path(test_dir) / "heures_a_predire.csv")
    if heures.columns.tolist() != ["ut_ms"]:
        raise ValueError("heures_a_predire.csv doit contenir seulement ut_ms")
    temps = heures.ut_ms
    if (temps.isna().any() or not pd.api.types.is_integer_dtype(temps)
            or temps.duplicated().any() or (temps % 3_600_000 != 0).any()):
        raise ValueError("Les heures doivent être entières, uniques et alignées sur une heure UTC")
    predictions = pd.DataFrame(constantes, index=heures.index)
    predictions.insert(0, "ut_ms", temps)
    predictions.to_csv(out, index=False)
    print(f"{len(predictions)} heures prédites dans {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    predire(args.test_dir, args.out)


if __name__ == "__main__":
    main()
