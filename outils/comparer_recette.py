"""Comparer un boosting candidat à la référence, mélangé au poids fixe 0.3 (docs/JOURNAL_RECETTE.md).

    python outils/comparer_recette.py --reference ref_gbm --candidat e1_familles

Pour chaque run (directe/ et inverse/ avec pred.csv) : RMSE du boosting seul, RMSE du mélange
0.7 × GRU + 0.3 × boosting avec les deux groupes de graines du GRU de référence, contributions MSE
des circuits suivis, puis décision selon la règle fixée avant les mesures :
(a) mélange −0.0003 au moins sur la moyenne des 4 RMSE et aucune direction > +0.0002 ;
(b) boosting seul en baisse dans les deux directions et mélange ≤ +0.0001.
Résultats écrits dans runs/<candidat>/comparaison.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.melanger import lire, rmse  # noqa: E402
from outils.valider import DIRECTIONS  # noqa: E402

GRU = ("p1_gru_g04", "p1_gru_g59")
POIDS = 0.3
SUIVIS = ("NPWD2551", "NPWD2451", "NPWD2851")


def mesurer(run, gru=GRU, poids=POIDS):
    sortie = {"seul": {}, "melange": {}, "contributions_seul": {}, "contributions_melange": {}}
    for direction, (_, validation) in DIRECTIONS.items():
        verite = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        v = verite.to_numpy()
        autre = lire(RACINE / "runs" / run, direction, validation).reindex(verite.index)[verite.columns].to_numpy()
        n = np.isfinite(v).sum()
        sortie["seul"][direction] = rmse(autre - v)
        sortie["contributions_seul"][direction] = {
            c: float(np.nansum((autre[:, j] - v[:, j]) ** 2) / n) for j, c in enumerate(verite.columns)}
        for g in gru:
            p = lire(RACINE / "runs" / g, direction, validation).reindex(verite.index)[verite.columns].to_numpy()
            m = (1 - poids) * p + poids * autre
            sortie["melange"][f"{g}/{direction}"] = rmse(m - v)
            sortie["contributions_melange"][f"{g}/{direction}"] = {
                c: float(np.nansum((m[:, j] - v[:, j]) ** 2) / n) for j, c in enumerate(verite.columns)}
    sortie["moyenne_melange"] = float(np.mean(list(sortie["melange"].values())))
    return sortie


def par_direction(mesure, d):
    return float(np.mean([r for k, r in mesure["melange"].items() if k.endswith(d)]))


def decider(ref, cand):
    gain = cand["moyenne_melange"] - ref["moyenne_melange"]
    hausse_max = max(par_direction(cand, d) - par_direction(ref, d) for d in DIRECTIONS)
    a = gain <= -0.0003 and hausse_max <= 0.0002
    b = all(cand["seul"][d] < ref["seul"][d] for d in DIRECTIONS) and gain <= 0.0001
    return {"delta_moyenne_melange": gain, "hausse_max_direction": hausse_max, "regle_a": a, "regle_b": b,
            "garder": bool(a or b)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidat", required=True, nargs="+")
    args = ap.parse_args()
    ref = mesurer(args.reference)
    lignes = [(args.reference, ref, None)] + [(c, m, decider(ref, m)) for c in args.candidat for m in [mesurer(c)]]
    print(f"{'run':24s} {'seul dir':>9s} {'seul inv':>9s} {'mél dir':>9s} {'mél inv':>9s} {'moy 4':>9s} {'Δ':>9s}  "
          + "  ".join(f"{c[4:]} d/i (×1e-4)" for c in SUIVIS) + "  décision")
    for nom, m, dec in lignes:
        suivis = "  ".join(
            f"{np.mean([m['contributions_melange'][f'{g}/directe'][c] for g in GRU]) * 1e4:5.2f}/"
            f"{np.mean([m['contributions_melange'][f'{g}/inverse'][c] for g in GRU]) * 1e4:5.2f}      "
            for c in SUIVIS)
        delta = f"{dec['delta_moyenne_melange']:+.6f}" if dec else ""
        verdict = ("garder" + (" (a)" if dec["regle_a"] else "") + (" (b)" if dec["regle_b"] else "")) if dec and dec["garder"] \
            else ("rejeter" if dec else "référence")
        print(f"{nom:24s} {m['seul']['directe']:9.6f} {m['seul']['inverse']:9.6f} {par_direction(m, 'directe'):9.6f} "
              f"{par_direction(m, 'inverse'):9.6f} {m['moyenne_melange']:9.6f} {delta:>9s}  {suivis}{verdict}")
        if dec:
            (RACINE / "runs" / nom / "comparaison.json").write_text(
                json.dumps({"reference": args.reference, "mesure": m, "reference_mesure": ref, "decision": dec},
                           indent=2) + "\n")


if __name__ == "__main__":
    main()
