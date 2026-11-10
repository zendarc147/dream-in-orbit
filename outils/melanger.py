"""Mélanger les prédictions d'un world model GRU et d'un autre modèle, poids global unique.

    python outils/melanger.py --gru runs/rob_cand_g04 runs/rob_cand_g59 --autre runs/gbm_A_V1 --nom hybride

Pour chaque run GRU et chaque direction : corrélation des erreurs (globale et par circuit),
RMSE du mélange (1 - w) GRU + w autre pour chaque poids, puis, au poids w* qui minimise la
moyenne de toutes les RMSE, RMSE, biais et contribution MSE par circuit, RMSE par tranche
d'horizon et par moitié de période. Le mélange au poids w* est écrit et noté par evaluate.py.
Aucun poids par circuit : un seul scalaire pour tout, choisi sur directe + inverse.
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
from outils.valider import DIRECTIONS, lancer  # noqa: E402

TRANCHES = [(0, 30), (30, 90), (90, 180), (180, 365), (365, 10_000)]


def lire(run, direction, validation):
    """Prédiction alignée sur les heures notées ; refusée si les heures diffèrent de heures_a_predire."""
    p = pd.read_csv(Path(run) / direction / "pred.csv", index_col="ut_ms")
    attendues = pd.read_csv(validation / "test" / "heures_a_predire.csv")["ut_ms"]
    if not p.index.equals(pd.Index(attendues, name="ut_ms")):
        raise ValueError(f"{run}/{direction} : heures différentes de heures_a_predire.csv")
    return p


def rmse(e):
    return float(np.sqrt(np.nanmean(e ** 2)))


def analyser(gru, autre, verite, w):
    """Mesures au poids w ; erreurs NaN là où la vérité manque (exclues, jamais mises à 0)."""
    noms, verite = verite.columns, verite.to_numpy()
    eg, ea = gru - verite, autre - verite
    em = (1 - w) * eg + w * ea
    connus = np.isfinite(verite)
    jours = (np.arange(len(verite)) / 24.0)
    moitie = len(verite) // 2
    circuits = {}
    for j, c in enumerate(noms):
        k = connus[:, j]
        circuits[c] = {
            "rmse_gru": rmse(eg[k, j]), "rmse_autre": rmse(ea[k, j]), "rmse_melange": rmse(em[k, j]),
            "biais_gru": float(eg[k, j].mean()), "biais_melange": float(em[k, j].mean()),
            "contribution_mse_gru": float(np.nansum(eg[:, j] ** 2) / connus.sum()),
            "contribution_mse_melange": float(np.nansum(em[:, j] ** 2) / connus.sum()),
            "correlation_erreurs": float(np.corrcoef(eg[k, j], ea[k, j])[0, 1]) if eg[k, j].std() > 0 else None,
        }
    tranches = {}
    for a, b in TRANCHES:
        ok = (jours >= a) & (jours < b)
        if ok.any():
            tranches[f"{a}-{min(b, int(jours.max()) + 1)}"] = {
                "gru": rmse(eg[ok]), "autre": rmse(ea[ok]), "melange": rmse(em[ok])}
    return {
        "correlation_erreurs": float(np.corrcoef(eg[connus], ea[connus])[0, 1]),
        "moities": [{"gru": rmse(eg[s]), "melange": rmse(em[s])} for s in (slice(0, moitie), slice(moitie, None))],
        "tranches": tranches, "circuits": circuits,
    }


def decider(poids, courbes, moyenne, i, par_direction, mesures):
    """Critères fixés avant mesure : toutes les RMSE baissent, gain moyen ≥ 0.001, plateau autour
    de w*, poids d'une direction utile à l'autre, gain sur au moins 3/4 des demi-périodes."""
    gain = float(moyenne[0] - moyenne[i])
    voisins = [moyenne[0] - moyenne[k] for k in (i - 1, i + 1) if 0 <= k < len(poids)]
    transfert = {}
    for d, w_d in par_direction.items():
        autre = [c for k, c in courbes.items() if not k.endswith(d)]
        k = int(np.flatnonzero(poids == w_d)[0])
        transfert[d] = bool(all(c[k] < c[0] for c in autre))
    moities = [m["melange"] < m["gru"] for mesure in mesures.values() for m in mesure["moities"]]
    criteres = {
        "toutes_baissent": bool(all(c[i] < c[0] for c in courbes.values())),
        "gain_moyen": gain,
        "gain_suffisant": gain >= 0.001,
        "plateau": bool(all(v >= gain / 2 for v in voisins)),
        "transfert": transfert,
        "moities_gagnantes": f"{sum(moities)}/{len(moities)}",
    }
    criteres["accepte"] = bool(criteres["toutes_baissent"] and criteres["gain_suffisant"] and criteres["plateau"]
                               and all(transfert.values()) and sum(moities) >= 0.75 * len(moities))
    criteres["incertain"] = bool(not criteres["accepte"] and criteres["toutes_baissent"] and gain >= 0.0005)
    return criteres


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gru", nargs="+", required=True, help="runs GRU (directe/ et inverse/ avec pred.csv)")
    ap.add_argument("--autre", nargs="+", required=True, help="run(s) des autres modèles, moyennés à poids égal")
    ap.add_argument("--poids", nargs="+", type=float, default=[0, .1, .2, .3, .4, .5, .6])
    ap.add_argument("--nom", required=True)
    args = ap.parse_args()
    poids = np.array(args.poids)
    donnees, courbes = {}, {}
    for direction, (_, validation) in DIRECTIONS.items():
        verite = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        brut = sum(lire(run, direction, validation) for run in args.autre) / len(args.autre)
        autre = brut.reindex(verite.index)[verite.columns]
        for run in args.gru:
            gru = lire(run, direction, validation).reindex(verite.index)[verite.columns]
            cle = f"{Path(run).name}/{direction}"
            donnees[cle] = (gru, autre, brut, verite, run, direction, validation)
            e = gru.to_numpy() - verite.to_numpy(), autre.to_numpy() - verite.to_numpy()
            courbes[cle] = [rmse((1 - w) * e[0] + w * e[1]) for w in poids]
    moyenne = np.mean(list(courbes.values()), axis=0)
    i = int(np.argmin(moyenne))
    w = float(poids[i])
    par_direction = {d: poids[int(np.argmin(np.mean([c for k, c in courbes.items() if k.endswith(d)], axis=0)))]
                     for d in DIRECTIONS}
    sortie = RACINE / "runs" / args.nom
    resultat = {"gru": args.gru, "autre": args.autre, "poids": poids.tolist(), "w_etoile": w,
                "w_etoile_par_direction": {d: float(v) for d, v in par_direction.items()},
                "courbes": courbes, "moyenne": moyenne.tolist(), "mesures": {}, "evaluate": {}}
    for cle, (gru, autre, brut, verite, run, direction, validation) in donnees.items():
        resultat["mesures"][cle] = analyser(gru.to_numpy(), autre.to_numpy(), verite, w)
        dossier = sortie / Path(run).name / direction
        dossier.mkdir(parents=True, exist_ok=True)
        pred = dossier / "pred.csv"
        lire(run, direction, validation).mul(1 - w).add(brut.mul(w)).to_csv(pred)
        texte = lancer(["evaluate.py", "--pred", pred, "--verite", validation / "verite_horaire.csv",
                        "--heures", validation / "test" / "heures_a_predire.csv"])
        resultat["evaluate"][cle] = float(re.search(r"RMSE = ([0-9.]+)", texte).group(1))
    resultat["decision"] = decider(poids, courbes, moyenne, i, par_direction, resultat["mesures"])
    (sortie / "melange.json").write_text(json.dumps(resultat, indent=2) + "\n")
    print("poids    " + " ".join(f"{p:8.2f}" for p in poids))
    for cle, c in courbes.items():
        print(f"{cle:28s}" [:28] + " ".join(f"{r:8.5f}" for r in c), f"corr {resultat['mesures'][cle]['correlation_erreurs']:.3f}")
    print("moyenne  " + " ".join(f"{r:8.5f}" for r in moyenne))
    print(f"w* = {w} (directe seule : {par_direction['directe']}, inverse seule : {par_direction['inverse']})")
    print("evaluate.py au poids w* :", json.dumps(resultat["evaluate"]))
    print("décision :", json.dumps(resultat["decision"]))


if __name__ == "__main__":
    main()
