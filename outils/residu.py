"""Apprendre l'erreur du GRU avec un boosting, sur ses prédictions hors bloc (étape 11 du plan).

    python outils/residu.py --hors-sac runs/gru_hors_sac --gru runs/rob_cand_g04 runs/rob_cand_g59 --nom residu

Entraînement, sur l'année d'entraînement seule : pour chaque circuit, un boosting (réglage A,
variables V1 + les 33 prédictions hors bloc du GRU) apprend courant - prédiction hors bloc,
uniquement aux heures où cette prédiction existe. Prédiction : GRU de validation + résidu appris,
évaluée par evaluate.py. Les prédictions hors bloc viennent de GRU qui n'ont jamais vu ces heures.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from baselines import gbm_sacs  # noqa: E402
from dio import donnees, monde  # noqa: E402
from outils.valider import DIRECTIONS, lancer  # noqa: E402


def variables(x, dmop, gru):
    """Contexte V1 calculé sur toute la grille (fenêtres glissantes complètes), puis restreint
    aux heures de gru, auxquelles s'ajoutent les 33 prédictions du GRU."""
    table = gbm_sacs.construire(x, dmop, "V1").loc[gru.index]
    return pd.concat([table, gru.add_prefix("gru_")], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hors-sac", required=True)
    ap.add_argument("--gru", nargs="+", required=True)
    ap.add_argument("--nom", required=True)
    args = ap.parse_args()
    resultat = {}
    for direction, (annee, validation) in DIRECTIONS.items():
        _, x, y = donnees.charger_annee(annee)
        hors_sac = pd.read_csv(Path(args.hors_sac) / direction / "hors_sac.csv", index_col="ut_ms")
        lignes = np.isfinite(hors_sac.to_numpy()).all(axis=1)
        table = variables(x, donnees.lire_dmop(annee), hors_sac)
        f = table.to_numpy(dtype=np.float64)
        garder = monde.calculer_stats(f[lignes], y[lignes])["garder"]
        colonnes = [c for c, g in zip(table.columns, garder) if g]
        residu = y - hors_sac.to_numpy()
        modeles = Parallel(n_jobs=-1)(delayed(gbm_sacs._ajuster)(f[:, garder], residu[:, i], np.flatnonzero(lignes), "A", 0)
                                      for i in range(y.shape[1]))
        test = donnees.lire_dossier_test(validation / "test")
        for run in args.gru:
            gru = pd.read_csv(Path(run) / direction / "pred.csv", index_col="ut_ms")
            x_test = test["X"].set_axis(pd.Index(test["heures"], name="ut_ms"))
            f_test = variables(x_test, test["dmop"], gru)[colonnes].to_numpy(dtype=np.float64)
            correction = np.stack([m.predict(f_test) for m in modeles], axis=1)
            dossier = RACINE / "runs" / args.nom / Path(run).name / direction
            dossier.mkdir(parents=True, exist_ok=True)
            (gru + correction).clip(lower=0).to_csv(dossier / "pred.csv")
            texte = lancer(["evaluate.py", "--pred", dossier / "pred.csv", "--verite", validation / "verite_horaire.csv",
                            "--heures", validation / "test" / "heures_a_predire.csv"])
            resultat[f"{Path(run).name}/{direction}"] = float(re.search(r"RMSE = ([0-9.]+)", texte).group(1))
            print(Path(run).name, direction, resultat[f"{Path(run).name}/{direction}"], flush=True)
    (RACINE / "runs" / args.nom / "residu.json").write_text(json.dumps(resultat, indent=2) + "\n")


if __name__ == "__main__":
    main()
