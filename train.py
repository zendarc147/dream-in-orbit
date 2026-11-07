"""Entraîner le modèle final à partir du kit, en une commande.

    python train.py                       # annee1 + annee2 -> poids/
    python train.py --annees annee1 --out runs/essai

Les données horaires sont lues dans data/prepared/ (créées par prepare.py si absentes).
"""
import argparse
import json
import time
from pathlib import Path

from dio import modeles

RACINE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annees", nargs="+", default=["annee1", "annee2"])
    ap.add_argument("--out", type=Path, default=RACINE / "poids")
    ap.add_argument("--modele", default=modeles.FINAL, choices=sorted(modeles.MODELES))
    ap.add_argument("--options", default="{}", help="options JSON passées à l'entraînement")
    args = ap.parse_args()
    debut = time.time()
    options = json.loads(args.options)
    modeles.module(args.modele).entrainer(args.annees, args.out, **options)
    meta = {"modele": args.modele, "annees": args.annees, "options": options}
    (args.out / "modele.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Entraînement terminé en {time.time() - debut:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
