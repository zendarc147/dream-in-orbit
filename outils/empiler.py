"""Empiler des modèles classiques avec des poids appris hors sac, sur l'année d'entraînement seule.

    python outils/empiler.py --runs runs/gbm_A_V1 runs/lin_fir --nom pile_gbm_fir

Pour chaque direction, les poids (non négatifs, un seul jeu pour tous les circuits) minimisent
l'erreur quadratique des prédictions hors sac (hors_sac.csv : chaque bloc prédit par un modèle
qui ne l'a pas vu) contre les courants de l'année d'entraînement. Ils sont ensuite appliqués aux
prédictions de validation (pred.csv) ; aucune vérité de validation n'intervient dans les poids.
Le résultat est noté par evaluate.py, et la moyenne simple sert de point de comparaison.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import donnees  # noqa: E402
from outils.valider import DIRECTIONS, lancer  # noqa: E402


def noter(pred, validation):
    texte = lancer(["evaluate.py", "--pred", pred, "--verite", validation / "verite_horaire.csv",
                    "--heures", validation / "test" / "heures_a_predire.csv"])
    return float(re.search(r"RMSE = ([0-9.]+)", texte).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nom", required=True)
    args = ap.parse_args()
    resultat = {}
    for direction, (annee, validation) in DIRECTIONS.items():
        _, _, y = donnees.charger_annee(annee)
        hors_sac = [pd.read_csv(Path(r) / direction / "hors_sac.csv", index_col="ut_ms").to_numpy() for r in args.runs]
        connus = np.isfinite(y) & np.logical_and.reduce([np.isfinite(h) for h in hors_sac])
        a = np.stack([h[connus] for h in hors_sac], axis=1)
        poids = nnls(a, y[connus])[0]
        preds = [pd.read_csv(Path(r) / direction / "pred.csv", index_col="ut_ms") for r in args.runs]
        dossier = RACINE / "runs" / args.nom / direction
        dossier.mkdir(parents=True, exist_ok=True)
        sum(p * w for p, w in zip(preds, poids)).clip(lower=0).to_csv(dossier / "pred.csv")
        (sum(preds) / len(preds)).to_csv(dossier / "pred_moyenne.csv")
        resultat[direction] = {"poids": dict(zip(args.runs, poids.tolist())),
                               "rmse_hors_sac": float(np.sqrt(np.mean((a @ poids - y[connus]) ** 2))),
                               "rmse_empilement": noter(dossier / "pred.csv", validation),
                               "rmse_moyenne_simple": noter(dossier / "pred_moyenne.csv", validation)}
    (RACINE / "runs" / args.nom / "empilement.json").write_text(json.dumps(resultat, indent=2) + "\n")
    print(json.dumps(resultat, indent=2))


if __name__ == "__main__":
    main()
