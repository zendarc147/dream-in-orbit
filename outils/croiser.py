"""Croisement hors ligne des composants (point 10, docs/JOURNAL_RECETTE_COMPLETE.md), sans spécialistes.

    python outils/croiser.py --gbm e22b_recette_sans_ema rc1_gbm_S --nom rc5_croisement

Mélange : w × GBM + (1 − w) × [(1 − a − b) × GRU mixte + a × GRU recette G2 + b × GRU S], chaque groupe de graines
d'un GRU apparié au groupe de mêmes graines des autres. Grille : w ∈ {0.3, 0.4, 0.5, 0.6}, (a, b) parmi des parts
fixées d'avance. Choix sur les graines 0–4 ; les graines 5–9 confirment. Écrit runs/<nom>/croisement.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.comparer_complete import DIRECTIONS, lire, verite  # noqa: E402

R = '@pred_{"recalage": 1.0}.csv'
GROUPES = {"g04": {"mixte": "l1_gru_sans_lettres_g04", "g2": "g2_sans_reinjection_recal", "s": "rc3_gruA_norm" + R},
           "g59": {"mixte": "l1_gru_sans_lettres_g59", "g2": "g2_sans_reinjection_g59_recal", "s": "rc4_gruA_mm_g59" + R}}
PARTS = [(0, 0), (0.3, 0), (0, 0.3), (0, 0.5), (0.15, 0.3), (0, 0.7), (0, 1.0)]
POIDS = (0.3, 0.4, 0.5, 0.6)


def rmse(p, d):
    v = verite(d).to_numpy()
    return float(np.sqrt(np.nanmean((p - v) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gbm", nargs="+", required=True)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--s", nargs=2, help="runs GRU S des graines 0–4 et 5–9")
    a = ap.parse_args()
    if a.s:
        GROUPES["g04"]["s"], GROUPES["g59"]["s"] = a.s
    cache = {(g, k, d): lire(r, d) for g, runs in GROUPES.items() for k, r in runs.items() for d in DIRECTIONS}
    lignes = []
    for gbm, w, (pa, pb) in itertools.product(a.gbm, POIDS, PARTS):
        res = {}
        for g, d in itertools.product(GROUPES, DIRECTIONS):
            gru = (1 - pa - pb) * cache[(g, "mixte", d)] + pa * cache[(g, "g2", d)] + pb * cache[(g, "s", d)]
            res[f"{g}/{d}"] = rmse(w * lire(gbm, d) + (1 - w) * gru, d)
        lignes.append({"gbm": gbm, "w": w, "part_g2": pa, "part_s": pb, **res,
                       "moy_g04": np.mean([res[f"g04/{d}"] for d in DIRECTIONS]),
                       "moy_g59": np.mean([res[f"g59/{d}"] for d in DIRECTIONS]),
                       "moy4": np.mean(list(res.values()))})
    lignes.sort(key=lambda x: x["moy_g04"])
    print(f"{'gbm':24s} {'w':>4s} {'g2':>5s} {'S':>5s} {'g04 dir':>8s} {'g04 inv':>8s} {'g59 dir':>8s} {'g59 inv':>8s} {'moy g04':>8s} {'moy4':>8s}")
    for x in lignes[:15] + [x for x in lignes if x["w"] == 0.5 and (x["part_g2"], x["part_s"]) in ((0.3, 0), (0, 0))]:
        print(f"{x['gbm'][:24]:24s} {x['w']:4.1f} {x['part_g2']:5.2f} {x['part_s']:5.2f} {x['g04/directe']:.6f} {x['g04/inverse']:.6f} "
              f"{x['g59/directe']:.6f} {x['g59/inverse']:.6f} {x['moy_g04']:.6f} {x['moy4']:.6f}")
    dossier = RACINE / "runs" / a.nom
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "croisement.json").write_text(json.dumps({"groupes": GROUPES, "lignes": lignes}, indent=1) + "\n")


if __name__ == "__main__":
    main()
