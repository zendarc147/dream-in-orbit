"""Reconstruire un power.csv horaire à partir des cibles déjà préparées.

    python outils/reconstruire_power.py

Les power.csv bruts (environ 600 Mo chacun) ne sont pas versionnés. Quand ils
manquent, on écrit data/anneeX/power.csv à partir de data/prepared/anneeX/y.csv :
une ligne par heure connue, placée à hh:30 pour que make_validation.py découpe
les 7 jours initiaux et la première heure notée exactement comme avec les
mesures minute par minute. La moyenne horaire obtenue est identique à y.csv.
Un power.csv déjà présent n'est jamais écrasé.
"""
from pathlib import Path

import pandas as pd

H = 3_600_000
RACINE = Path(__file__).resolve().parent.parent


def reconstruire(annee):
    sortie = RACINE / "data" / annee / "power.csv"
    if sortie.exists():
        print(f"{sortie} existe déjà : conservé")
        return sortie
    y = pd.read_csv(RACINE / "data/prepared" / annee / "y.csv")
    # Une heure sans aucune mesure n'a pas de ligne dans le vrai power.csv.
    y = y[y.drop(columns="ut_ms").notna().any(axis=1)].copy()
    y["ut_ms"] = y["ut_ms"] + H // 2
    y.to_csv(sortie, index=False)
    print(f"{sortie} : {len(y)} heures reconstruites")
    return sortie


def main():
    for annee in ("annee1", "annee2"):
        reconstruire(annee)


if __name__ == "__main__":
    main()
