"""Validation chronologique : annee1 + première moitié d'annee2 -> seconde moitié d'annee2.

    python outils/faire_chrono.py
    python outils/faire_chrono.py --glissant   # validation_glissant/c50, c62, c75, c87 (docs/journaux/JOURNAL_STRUCTURE.md)

Écrit validation_chrono/entrainement/ (période d'entraînement au format du kit : power.csv et contextes avant la
coupure) et validation_chrono/test/ + verite_horaire.csv (make_validation.construire sur la seconde moitié :
7 jours de power_init puis heures à prédire). Les contextes du test commencent à la coupure, comme ceux d'un
dossier de test ; ltdata garde le jour qui contient la coupure (interpolation du début).

--glissant : origine glissante. Coupures au début d'annee2 puis à 25, 50 et 75 % d'annee2 (50, 62,5, 75 et 87,5 % de la
durée totale des deux années contiguës) ; entraînement = annee1 + annee2 avant la coupure (validation_glissant/cNN/
entrainement, absent pour c50 : annee1 seule) ; test = tranche suivante de même longueur (12,5 % du total), contextes
coupés aux deux bords (ltdata : jours entiers qui touchent la tranche).
"""
import argparse
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


GLISSANT = {"c50": 0, "c62": 1, "c75": 2, "c87": 3}


def glissant():
    source = RACINE / "data" / "annee2"
    power = pd.read_csv(source / "power.csv", usecols=["ut_ms"]).ut_ms
    debut = power.min() // H * H
    n = (power.max() // H * H - debut) // H + 1
    for nom, k in GLISSANT.items():
        coupure, fin = int(debut + (k * n // 4) * H), int(debut + ((k + 1) * n // 4) * H)
        sortie = RACINE / "validation_glissant" / nom
        if k:
            couper(source, sortie / "entrainement",
                   lambda t, nom: t < (coupure // JOUR + 1) * JOUR if nom == "ltdata" else t < coupure)
        with tempfile.TemporaryDirectory() as temp:
            couper(source, Path(temp), lambda t, nom, a=coupure, b=fin: (
                (t >= a // JOUR * JOUR) & (t < ((b - 1) // JOUR + 1) * JOUR)) if nom == "ltdata" else (t >= a) & (t < b))
            print(nom, construire(Path(temp) / "power.csv", Path(temp), sortie), "coupure", coupure, "fin", fin)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glissant", action="store_true")
    if ap.parse_args().glissant:
        return glissant()
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
