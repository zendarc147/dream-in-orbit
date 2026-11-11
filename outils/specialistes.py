"""Spécialistes de quelques circuits (NPWD2551, NPWD2451) : entraînement, mesure et mélange par colonne.

    python outils/specialistes.py entrainer --nom sp_base --options '{"config": "B", "jeu": "V2"}'
    python outils/specialistes.py melanger --base runs/p3_val_g04 runs/p3_val_g59 \
        --composants runs/sp_base 'runs/sp_gru_{g}' --nom sp_mel

entrainer : boosting à sacs sur les seules colonnes visées, dans les deux directions (annee1 ->
validation/, annee2 -> validation_inverse/), prédictions dans runs/NOM/DIRECTION/pred.csv.
melanger : pour chaque colonne visée, poids positifs de somme 1 (grille de pas 0.1) entre la base
(modèle livré) et les composants. Un « {g} » dans un composant est remplacé par le groupe de graines
de la base (g04, g59). Les poids sont choisis sur une direction (moyenne des groupes de graines) et
appliqués à l'autre : la RMSE notée d'une direction ne sert jamais à choisir ses propres poids.
Critère fixé avant mesure : la colonne baisse dans les 4 cas (2 directions × 2 groupes) avec les
poids transférés, et aucune moitié de période ne monte de plus de 0.002.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.valider import DIRECTIONS  # noqa: E402

CIRCUITS = ["NPWD2551", "NPWD2451"]
PAS = 0.1


def entrainer(nom, options, circuits):
    from baselines import gbm_sacs
    for direction, (annee, validation) in DIRECTIONS.items():
        dossier = RACINE / "runs" / nom / direction
        debut = time.time()
        gbm_sacs.entrainer([annee], dossier, circuits=circuits, **options)
        gbm_sacs.predire(validation / "test", dossier / "pred.csv", dossier)
        print(f"{nom} {direction} : {time.time() - debut:.0f} s", flush=True)
    scores = mesurer([RACINE / "runs" / nom], circuits)
    (RACINE / "runs" / nom / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores))


def lire(run, direction, verite, colonnes):
    return pd.read_csv(Path(run) / direction / "pred.csv", index_col="ut_ms").reindex(verite.index)[colonnes]


def rmse(e):
    return float(np.sqrt(np.nanmean(e ** 2)))


def mesurer(runs, circuits):
    sortie = {}
    for direction, (_, validation) in DIRECTIONS.items():
        verite = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        for run in runs:
            p = lire(run, direction, verite, circuits)
            sortie[f"{Path(run).name}/{direction}"] = {c: rmse(p[c] - verite[c]) for c in circuits}
    return sortie


def grille(n):
    """Poids positifs de somme 1 au pas PAS, n composants."""
    k = round(1 / PAS)
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def melanger(bases, composants, circuits, nom):
    groupes = {Path(b).name.rsplit("_", 1)[-1]: b for b in bases}
    erreurs, moities = {}, {}  # (circuit, direction, groupe) -> matrice heures × (base + composants)
    for direction, (_, validation) in DIRECTIONS.items():
        verite = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        for g, base in groupes.items():
            runs = [base] + [c.replace("{g}", g) for c in composants]
            preds = [lire(r, direction, verite, circuits) for r in runs]
            for c in circuits:
                connus = verite[c].notna().to_numpy()
                erreurs[c, direction, g] = np.stack([p[c].to_numpy() - verite[c].to_numpy() for p in preds], 1)[connus]
    poids = grille(1 + len(composants))
    resultat = {"bases": bases, "composants": composants, "circuits": {}}
    for c in circuits:
        courbes = {d: np.mean([[rmse(erreurs[c, d, g] @ w) for w in poids] for g in groupes], axis=0)
                   for d in DIRECTIONS}
        choix = {d: poids[int(np.argmin(courbes[d]))] for d in DIRECTIONS}
        commun = poids[int(np.argmin(courbes["directe"] + courbes["inverse"]))]
        cas, ok = {}, True
        for d in DIRECTIONS:
            autre = next(a for a in DIRECTIONS if a != d)
            w = choix[autre]  # poids appris sur l'autre direction
            for g in groupes:
                e = erreurs[c, d, g]
                n = len(e) // 2
                avant, apres = rmse(e[:, 0]), rmse(e @ w)
                demis = [rmse(e[s] @ w) - rmse(e[s, 0]) for s in (slice(0, n), slice(n, None))]
                ok &= apres < avant and max(demis) <= 0.002
                cas[f"{d}/{g}"] = {"base": avant, "transfere": apres, "commun": rmse(e @ commun),
                                   "moities": demis}
        resultat["circuits"][c] = {"poids_directe": choix["directe"].tolist(), "poids_inverse": choix["inverse"].tolist(),
                                   "poids_commun": commun.tolist(), "cas": cas, "accepte": bool(ok)}
        print(f"{c} : poids directe {choix['directe']}, inverse {choix['inverse']}, commun {commun}, accepté {ok}")
        for k, v in cas.items():
            print(f"  {k:14s} base {v['base']:.4f} -> transféré {v['transfere']:.4f} (commun {v['commun']:.4f})"
                  f" moitiés {v['moities'][0]:+.4f} {v['moities'][1]:+.4f}")
    sortie = RACINE / "runs" / nom
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "melange.json").write_text(json.dumps(resultat, indent=2) + "\n")
    return resultat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="action", required=True)
    e = sous.add_parser("entrainer")
    e.add_argument("--nom", required=True)
    e.add_argument("--options", default="{}")
    e.add_argument("--circuits", nargs="+", default=CIRCUITS)
    m = sous.add_parser("melanger")
    m.add_argument("--base", nargs="+", required=True)
    m.add_argument("--composants", nargs="+", required=True)
    m.add_argument("--circuits", nargs="+", default=CIRCUITS)
    m.add_argument("--nom", required=True)
    args = ap.parse_args()
    if args.action == "entrainer":
        entrainer(args.nom, json.loads(args.options), args.circuits)
    else:
        melanger(args.base, args.composants, args.circuits, args.nom)


if __name__ == "__main__":
    main()
