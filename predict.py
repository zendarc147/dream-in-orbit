"""Simuler la consommation horaire des 33 circuits sur un dossier de test.

    python predict.py --test_dir validation/test --out predictions.csv

Le dossier contient les 5 contextes, power_init.csv (7 jours) et heures_a_predire.csv.
Le modèle et ses options sont lus dans poids/modele.json.
"""
import argparse
import json
import time
from pathlib import Path

from dio import modeles

RACINE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--poids", type=Path, default=RACINE / "poids")
    ap.add_argument("--options", default="{}", help="options JSON de prédiction (surchargent modele.json)")
    args = ap.parse_args()
    debut = time.time()
    meta = json.loads((args.poids / "modele.json").read_text())
    options = {**meta.get("options_prediction", {}), **json.loads(args.options)}
    modeles.module(meta["modele"]).predire(args.test_dir, args.out, args.poids, **options)
    print(f"Prédiction terminée en {time.time() - debut:.0f} s")


if __name__ == "__main__":
    main()
