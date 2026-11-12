"""Bilan d'un essai de la recette complète (docs/JOURNAL_RECETTE_COMPLETE.md) contre une référence.

    python outils/comparer_complete.py --nom rc2_saaf --gbm rc2_saaf_gbm --gru rc1_gru_A \
        --ref-gbm rc1_gbm_S --ref-gru rc1_gru_A --composant gbm

Un run s'écrit NOM ou NOM@fichier.csv (variante de prédiction, ex. rc1_gru_A@pred_{"recalage": 1.0}.csv).
Plusieurs --gru = groupes de graines (moyennés par direction). Mélange : (1 − w) × GRU + w × GBM, w fixe (0.5).
Règle fixée avant les mesures (celle de docs/JOURNAL_RECETTE.md) :
(a) moyenne des RMSE du mélange en baisse ≥ 0.0003 et aucune direction en hausse > 0.0002 ;
(b) composant changé seul en baisse dans les deux directions et mélange ≤ +0.0001.
Écrit runs/<nom>/bilan.json : RMSE seuls et mélange, contributions MSE par circuit, colonnes, temps, décision.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils import valider  # noqa: E402

DIRECTIONS = dict(valider.DIRECTIONS)

SUIVIS = ("NPWD2551", "NPWD2451", "NPWD2851")
_VERITES = {}


def verite(direction):
    if direction not in _VERITES:
        _VERITES[direction] = pd.read_csv(valider.TOUTES[direction][1] / "verite_horaire.csv", index_col="ut_ms")
    return _VERITES[direction]


def lire(run, direction):
    nom, _, fichier = run.partition("@")
    v = verite(direction)
    p = pd.read_csv(RACINE / "runs" / nom / direction / (fichier or "pred.csv"), index_col="ut_ms")
    return p.reindex(v.index)[v.columns].to_numpy()


def mesure(p, direction):
    v = verite(direction).to_numpy()
    e = p - v
    n = np.isfinite(v).sum()
    return {"rmse": float(np.sqrt(np.nansum(e ** 2) / n)),
            "contributions": {c: float(np.nansum(e[:, j] ** 2) / n) for j, c in enumerate(verite(direction).columns)}}


def infos(run):
    nom = run.partition("@")[0]
    sortie = {}
    for d in DIRECTIONS:
        dossier = RACINE / "runs" / nom / d
        i = {}
        if (dossier / "colonnes.json").is_file():
            i["n_colonnes"] = len(json.loads((dossier / "colonnes.json").read_text()))
        if (dossier / "recette.pt").is_file():
            import torch
            paquet = torch.load(dossier / "recette.pt", weights_only=True)
            i["n_colonnes"], i["graines"] = len(paquet["variables"]), paquet["config"]["graines"]
        if (dossier / "modele.json").is_file():
            i["options"] = json.loads((dossier / "modele.json").read_text()).get("options")
        if (dossier / "train.log").is_file():
            m = re.search(r"terminé en (\d+) s", (dossier / "train.log").read_text())
            i["train_s"] = int(m.group(1)) if m else None
        sortie[d] = i
    return sortie


def evaluer(gbm, grus, poids):
    s = {"gbm": {}, "gru": {}, "melange": {}}
    for d in DIRECTIONS:
        pg = lire(gbm, d)
        s["gbm"][d] = mesure(pg, d)
        for g in grus:
            pr = lire(g, d)
            s["gru"][f"{g}/{d}"] = mesure(pr, d)
            s["melange"][f"{g}/{d}"] = mesure((1 - poids) * pr + poids * pg, d)
    s["moyenne_melange"] = float(np.mean([m["rmse"] for m in s["melange"].values()]))
    return s


def par_direction(s, cle, d):
    return float(np.mean([m["rmse"] for k, m in s[cle].items() if k.endswith(d)]))


def contribution(s, cle, d, c):
    return float(np.mean([m["contributions"][c] for k, m in s[cle].items() if k.endswith(d)]))


def decider(ref, cand, composant):
    gain = cand["moyenne_melange"] - ref["moyenne_melange"]
    hausse = max(par_direction(cand, "melange", d) - par_direction(ref, "melange", d) for d in DIRECTIONS)
    a = gain <= -0.0003 and hausse <= 0.0002
    b = (composant in ("gbm", "gru")
         and all(par_direction(cand, composant, d) < par_direction(ref, composant, d) for d in DIRECTIONS)
         and gain <= 0.0001)
    return {"delta": gain, "hausse_max_direction": hausse, "regle_a": a, "regle_b": bool(b), "garder": bool(a or b)}


def ligne(nom, s, dec=None):
    if "chrono" in DIRECTIONS:
        return (f"{nom:22s} chrono gbm {par_direction(s, 'gbm', 'chrono'):.6f}  gru {par_direction(s, 'gru', 'chrono'):.6f}"
                f"  mél {s['moyenne_melange']:.6f}" + ("" if dec is None else f"  {dec['delta']:+.6f}"))
    suivis = " ".join(f"{contribution(s, 'melange', 'directe', c) * 1e4:5.2f}/{contribution(s, 'melange', 'inverse', c) * 1e4:5.2f}"
                      for c in SUIVIS)
    verdict = "" if dec is None else (f"{dec['delta']:+.6f} " + ("garder" + (" (a)" if dec["regle_a"] else "")
                                                               + (" (b)" if dec["regle_b"] else "") if dec["garder"] else "rejet"))
    return (f"{nom:22s} gbm {par_direction(s, 'gbm', 'directe'):.6f}/{par_direction(s, 'gbm', 'inverse'):.6f}  "
            f"gru {par_direction(s, 'gru', 'directe'):.6f}/{par_direction(s, 'gru', 'inverse'):.6f}  "
            f"mél {par_direction(s, 'melange', 'directe'):.6f}/{par_direction(s, 'melange', 'inverse'):.6f}  "
            f"moy {s['moyenne_melange']:.6f}  2551/2451/2851 ×1e-4 {suivis}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--gbm", required=True)
    ap.add_argument("--gru", required=True, nargs="+")
    ap.add_argument("--ref-gbm")
    ap.add_argument("--ref-gru", nargs="+")
    ap.add_argument("--composant", choices=["gbm", "gru", "aucun"], default="aucun")
    ap.add_argument("--poids", type=float, default=0.5)
    ap.add_argument("--note", default="")
    ap.add_argument("--chrono", action="store_true", help="validation chronologique seule")
    a = ap.parse_args()
    if a.chrono:
        DIRECTIONS.clear()
        DIRECTIONS["chrono"] = valider.TOUTES["chrono"]
    cand = evaluer(a.gbm, a.gru, a.poids)
    sortie = {"gbm": a.gbm, "gru": a.gru, "poids": a.poids, "note": a.note, "mesure": cand,
              "infos": {r: infos(r) for r in [a.gbm, *a.gru]}}
    if a.ref_gbm:
        ref = evaluer(a.ref_gbm, a.ref_gru or a.gru, a.poids)
        dec = decider(ref, cand, a.composant)
        sortie.update({"reference": {"gbm": a.ref_gbm, "gru": a.ref_gru or a.gru, "mesure": ref}, "decision": dec})
        print(ligne("référence", ref))
        print(ligne(a.nom, cand, dec))
    else:
        print(ligne(a.nom, cand))
    dossier = RACINE / "runs" / a.nom
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "bilan.json").write_text(json.dumps(sortie, indent=1) + "\n")


if __name__ == "__main__":
    main()
