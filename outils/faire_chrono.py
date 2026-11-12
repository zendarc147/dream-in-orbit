"""Validation chronologique : annee1 + première moitié d'annee2 -> seconde moitié d'annee2.

    python outils/faire_chrono.py

Écrit validation_chrono/entrainement/ (période d'entraînement au format du kit : power.csv et contextes avant la
coupure) et validation_chrono/test/ + verite_horaire.csv (make_validation.construire sur la seconde moitié :
7 jours de power_init puis heures à prédire). Les contextes du test commencent à la coupure, comme ceux d'un
dossier de test ; ltdata garde le jour qui contient la coupure (interpolation du début).
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from make_validation import CONTEXTES, H, construire  # noqa: E402

JOUR = 24 * H


def couper(source, cible, garder):
    cible.mkdir(parents=True, exist_ok=True)
    for nom in CONTEXTES + ["power"]:
        t = pd.read_csv(source / f"{nom}.csv")
        temps = t["utb_ms" if nom == "ftl" else "ut_ms"]
        t[garder(temps, nom)].to_csv(cible / f"{nom}.csv", index=False)


def main():
    source = RACINE / "data" / "annee2"
    power = pd.read_csv(source / "power.csv", usecols=["ut_ms"]).ut_ms
    debut = power.min() // H * H
    n = (power.max() // H * H - debut) // H + 1
    coupure = int(debut + (n // 2) * H)
    sortie = RACINE / "validation_chrono"
    couper(source, sortie / "entrainement",
           lambda t, nom: t < (coupure // JOUR + 1) * JOUR if nom == "ltdata" else t < coupure)
    with tempfile.TemporaryDirectory() as temp:
        couper(source, Path(temp), lambda t, nom: t >= (coupure // JOUR) * JOUR if nom == "ltdata" else t >= coupure)
        print(construire(Path(temp) / "power.csv", Path(temp), sortie), "coupure", coupure)


if __name__ == "__main__":
    main()
