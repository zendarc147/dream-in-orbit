"""Plafond d'un travail ciblé sur quelques circuits : ce que vaudrait une prédiction parfaite,
et quelle part de leur erreur est du bruit de thermostat à l'échelle horaire.

    python outils/plafond_circuits.py --run runs/p3_val_g04 --circuits NPWD2551 NPWD2451

Pour chaque direction : RMSE totale actuelle, puis avec les circuits parfaits (erreur 0) ;
par circuit : RMSE du run, bruit d'une heure à l'autre (RMSE de la persistance lag 1 / √2,
borne du bruit blanc horaire), résidu d'un lissage oracle centré (3 à 24 h), autocorrélation
du résidu et part de sa puissance par bande de période.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.valider import DIRECTIONS  # noqa: E402

BANDES = [("> 30 j", 720, np.inf), ("1-30 j", 24, 720), ("12-24 h", 12, 24), ("6-12 h", 6, 12),
          ("3-6 h", 3, 6), ("2-3 h", 2, 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--circuits", nargs="+", default=["NPWD2551", "NPWD2451"])
    args = ap.parse_args()
    for direction, (_, validation) in DIRECTIONS.items():
        verite = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        pred = pd.read_csv(Path(args.run) / direction / "pred.csv", index_col="ut_ms").loc[verite.index, verite.columns]
        e2 = (pred - verite) ** 2
        n = np.isfinite(verite.to_numpy()).sum()
        print(f"{direction} : RMSE {np.sqrt(np.nansum(e2.to_numpy()) / n):.4f}, circuits parfaits "
              f"{np.sqrt(np.nansum(e2.drop(columns=args.circuits).to_numpy()) / n):.4f}")
        for c in args.circuits:
            x = verite[c]
            erreur = (x - pred[c]).interpolate().fillna(0).to_numpy()
            lisse = " ".join(f"{w} h {np.sqrt(np.nanmean((x - x.rolling(w, center=True, min_periods=1).mean()) ** 2)):.3f}"
                             for w in (3, 6, 12, 24))
            acf = " ".join(f"{k}:{np.corrcoef(erreur[:-k], erreur[k:])[0, 1]:.2f}" for k in (1, 3, 7, 24))
            f = np.fft.rfftfreq(len(erreur))
            p = np.abs(np.fft.rfft(erreur - erreur.mean())) ** 2
            with np.errstate(divide="ignore"):
                periode = 1 / f
            bandes = " ".join(f"{nom} {p[(periode >= a) & (periode < b)].sum() / p.sum():.2f}" for nom, a, b in BANDES)
            print(f"  {c} : RMSE {np.sqrt(np.nanmean(e2[c])):.3f}, part MSE {np.nansum(e2[c]) / np.nansum(e2.to_numpy()):.2f}, "
                  f"bruit horaire {np.sqrt(np.nanmean((x - x.shift(1)) ** 2) / 2):.3f}")
            print(f"    lissage oracle : {lisse}")
            print(f"    autocorrélation du résidu : {acf} ; puissance : {bandes}")


if __name__ == "__main__":
    main()
