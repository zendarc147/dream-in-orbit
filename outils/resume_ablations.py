"""Résumer des ablations du boosting : score seul et mélange avec le GRU, contre une référence.

    python outils/resume_ablations.py --reference p2_base p2_planning p2_historique ...

Chaque nom désigne runs/NOM (scores.json de valider.py) et runs/NOM_mel/melange.json
(melanger.py). Pour chaque variante : RMSE seule par direction, puis, au meilleur poids de la
grille, moyenne des 4 RMSE du mélange et moyenne de chaque direction (2 groupes de graines),
en écart à la référence. Règle de décision fixée dans docs/JOURNAL_ENSEMBLE.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"


def lire(nom):
    scores = json.loads((RUNS / nom / "scores.json").read_text())
    melange = json.loads((RUNS / f"{nom}_mel" / "melange.json").read_text())
    courbes, i = melange["courbes"], int(np.argmin(melange["moyenne"]))
    direction = {d: float(np.mean([c[i] for k, c in courbes.items() if k.endswith(d)])) for d in ("directe", "inverse")}
    return {"seul": {d: scores[d]["rmse"] for d in ("directe", "inverse")}, "w": melange["poids"][i],
            "melange": float(melange["moyenne"][i]), "direction": direction}


def decider(v, r):
    seul = all(v["seul"][d] < r["seul"][d] for d in v["seul"])
    a = seul and v["melange"] - r["melange"] <= 0.0001
    b = r["melange"] - v["melange"] >= 0.0003 and all(v["direction"][d] - r["direction"][d] <= 0.0002
                                                     for d in v["direction"])
    return "garder" if a or b else "rejeter"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("variantes", nargs="*")
    args = ap.parse_args()
    r = lire(args.reference)
    print(f"{'variante':24s} {'seul dir':>9s} {'seul inv':>9s} {'w*':>4s} {'mél dir':>8s} {'mél inv':>8s}"
          f" {'mél moy':>8s} {'Δ moy':>8s}  décision")
    for nom in [args.reference, *args.variantes]:
        v = lire(nom)
        print(f"{nom:24s} {v['seul']['directe']:9.6f} {v['seul']['inverse']:9.6f} {v['w']:4.1f}"
              f" {v['direction']['directe']:8.5f} {v['direction']['inverse']:8.5f} {v['melange']:8.5f}"
              f" {v['melange'] - r['melange']:+8.5f}  {'—' if nom == args.reference else decider(v, r)}")


if __name__ == "__main__":
    main()
