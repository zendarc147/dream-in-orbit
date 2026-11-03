"""Calculer les constantes depuis les seules cibles horaires d'annee1."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    racine = Path(__file__).resolve().parent.parent
    schema = json.loads((racine / "data/prepared/schema.json").read_text())
    circuits = schema["target_columns"]
    y = pd.read_csv(racine / "data/prepared/annee1/y.csv")
    if (len(circuits) != 33 or len(set(circuits)) != 33
            or not all(c.startswith("NPWD") for c in circuits)
            or y.columns.tolist() != ["ut_ms"] + circuits):
        raise ValueError("33 circuits NPWD dans l'ordre du schéma sont requis")
    if y.ut_ms.isna().any() or not y.ut_ms.is_unique:
        raise ValueError("Les heures d'apprentissage doivent être présentes et uniques")
    cibles = y[circuits]
    if np.isinf(cibles.to_numpy(dtype=float)).any():
        raise ValueError("Les cibles contiennent une valeur infinie")
    # Une voix par heure connue, sans remplacer les heures manquantes par zéro.
    constantes = cibles.mean(skipna=True)
    if not np.isfinite(constantes.to_numpy()).all():
        raise ValueError("Chaque circuit doit avoir au moins une cible finie")
    sortie = Path(__file__).resolve().parent / "baseline_constante.json"
    sortie.write_text(json.dumps(constantes.to_dict(), indent=2, allow_nan=False) + "\n")
    print(f"{len(constantes)} constantes enregistrées dans {sortie.name}")


if __name__ == "__main__":
    main()
