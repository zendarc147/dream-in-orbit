"""Régimes de NPWD2551, NPWD2451 et NPWD2851 sur les validations autorisées (annee1, annee2).

    python outils/diagnostic_regimes.py --run runs/rob_cand_g04 [--run runs/autre ...]

Les seuils de régime sont appris, pour chaque direction, sur la seule année d'entraînement
(2-moyennes 1D sur log10, comme la perte auxiliaire de monde.py). La vérité évaluée sert
uniquement à mesurer les erreurs. Sorties JSON dans runs/regimes/ (statistiques des données)
et dans RUN/regimes.json (erreurs du modèle).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import donnees  # noqa: E402
from dio import monde  # noqa: E402
from dio import variables  # noqa: E402

CIBLES = monde.CIBLES_REGIME
TRANCHES = ((0, 30), (30, 90), (90, 180), (180, 365), (365, 10_000))
DIRECTIONS = {"directe": ("annee1", RACINE / "validation"), "inverse": ("annee2", RACINE / "validation_inverse")}


def y_annee(annee):
    return pd.read_csv(RACINE / "data/prepared" / annee / "y.csv", index_col="ut_ms")


def regimes(y, seuils):
    """(heures, circuits) entiers : 0 bas, 1 haut ; -1 si inconnu."""
    r = (y >= seuils).astype(int)
    return np.where(np.isfinite(y), r, -1)


def durees(r):
    """Longueurs des séquences contiguës de chaque régime (heures inconnues coupent la séquence)."""
    sortie = {0: [], 1: []}
    debut = 0
    for t in range(1, len(r) + 1):
        if t == len(r) or r[t] != r[debut]:
            if r[debut] >= 0:
                sortie[int(r[debut])].append(t - debut)
            debut = t
    return {k: {"n": len(v), "mediane_h": float(np.median(v)) if v else None,
                "p90_h": float(np.quantile(v, .9)) if v else None, "max_h": int(max(v)) if v else None}
            for k, v in sortie.items()}


def statistiques_donnees(out):
    resultat = {}
    for direction, (annee, validation) in DIRECTIONS.items():
        train = y_annee(annee)[CIBLES].to_numpy()
        seuils2 = monde.seuils_regime(train)
        seuils3 = np.array([monde.kmeans_log(train[:, j], 3)[1] for j in range(len(CIBLES))])
        test = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")[CIBLES].to_numpy()
        bloc = {"seuils_k2": dict(zip(CIBLES, seuils2.tolist())),
                "seuils_k3": dict(zip(CIBLES, seuils3.tolist()))}
        for nom, y in (("entrainement", train), ("evaluation", test)):
            r = regimes(y, seuils2)
            connus = (r >= 0).all(axis=1)
            par_circuit = {}
            for j, c in enumerate(CIBLES):
                rj = r[:, j]
                k3 = np.digitize(y[:, j], seuils3[j])
                par_circuit[c] = {
                    "part_haut": float((rj[rj >= 0] == 1).mean()),
                    "parts_k3": np.bincount(k3[np.isfinite(y[:, j])], minlength=3).tolist(),
                    "moyenne_par_regime": [float(np.nanmean(y[rj == k, j])) for k in (0, 1)],
                    "ecart_par_regime": [float(np.nanstd(y[rj == k, j])) for k in (0, 1)],
                    "transitions_par_jour": float((np.diff(rj)[(rj[1:] >= 0) & (rj[:-1] >= 0)] != 0).sum()
                                                  / (len(rj) / 24)),
                    "durees": durees(rj),
                    "quantiles": dict(zip(["5", "25", "50", "75", "95"],
                                          np.nanquantile(y[:, j], [.05, .25, .5, .75, .95]).round(4).tolist())),
                }
            # Coactivations : fréquence de chaque combinaison (2551, 2451, 2851) contre l'indépendance.
            combos = {}
            rc = r[connus]
            marges = rc.mean(axis=0)
            for code in range(8):
                bits = np.array([(code >> (2 - j)) & 1 for j in range(3)])
                observe = float((rc == bits).all(axis=1).mean())
                independant = float(np.prod(np.where(bits == 1, marges, 1 - marges)))
                combos["".join(map(str, bits))] = {"observe": observe, "independance": independant}
            # Régime latent commun : corrélations des indicateurs, horaires et en moyenne journalière.
            jours = len(rc) // 24
            journalier = rc[:jours * 24].reshape(jours, 24, 3).mean(axis=1)
            bloc[nom] = {"circuits": par_circuit, "coactivations": combos,
                         "correlation_regimes_horaire": np.corrcoef(rc.T).round(3).tolist(),
                         "correlation_regimes_journaliere": np.corrcoef(journalier.T).round(3).tolist(),
                         "premiere_composante_part_variance": float(
                             np.linalg.eigvalsh(np.corrcoef(journalier.T))[-1] / 3)}
        resultat[direction] = bloc
    out.mkdir(parents=True, exist_ok=True)
    (out / "donnees.json").write_text(json.dumps(resultat, indent=2) + "\n")
    return resultat


def rmse(e):
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else None


def erreurs_modele(run):
    resultat = {}
    for direction, (annee, validation) in DIRECTIONS.items():
        v = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        p = pd.read_csv(Path(run) / direction / "pred.csv", index_col="ut_ms").reindex(v.index)[v.columns]
        seuils = monde.seuils_regime(y_annee(annee)[CIBLES].to_numpy())
        jours = (v.index.to_numpy() - v.index[0]) / 86_400_000
        erreur = p.to_numpy() - v.to_numpy()
        n_total = np.isfinite(v.to_numpy()).sum()
        bloc = {"rmse_globale": rmse(erreur),
                "rmse_par_circuit": {c: rmse(erreur[:, j]) for j, c in enumerate(v.columns)},
                "contribution_mse": {c: float(np.nansum(erreur[:, j] ** 2) / n_total)
                                     for j, c in enumerate(v.columns)},
                "tranches": {}}
        for a, b in TRANCHES:
            ok = (jours >= a) & (jours < b)
            if ok.any():
                bloc["tranches"][f"{a}-{min(b, int(jours.max()) + 1)}"] = {
                    "globale": rmse(erreur[ok]), "trois_cibles": rmse(erreur[ok][:, [v.columns.get_loc(c) for c in CIBLES]]),
                    **{c: rmse(erreur[ok, v.columns.get_loc(c)]) for c in CIBLES}}
        for j, c in enumerate(CIBLES):
            y, q = v[c].to_numpy(), p[c].to_numpy()
            ok = np.isfinite(y)
            vrai = y[ok] >= seuils[j]
            # Moyenne conditionnelle : par décile de prédiction, moyenne observée contre prédite.
            deciles = np.quantile(q[ok], np.linspace(0, 1, 11))
            classe = np.clip(np.digitize(q[ok], deciles[1:-1]), 0, 9)
            calibration = [{"predit": float(q[ok][classe == k].mean()), "observe": float(y[ok][classe == k].mean())}
                           for k in range(10) if (classe == k).any()]
            bloc[c] = {
                "seuil": float(seuils[j]),
                "part_haut_observee": float(vrai.mean()),
                "part_haut_predite": float((q[ok] >= seuils[j]).mean()),
                "ecart_type_observe": float(y[ok].std()), "ecart_type_predit": float(q[ok].std()),
                "prediction_moyenne_si_bas": float(q[ok][~vrai].mean()) if (~vrai).any() else None,
                "prediction_moyenne_si_haut": float(q[ok][vrai].mean()) if vrai.any() else None,
                "observe_moyen_si_bas": float(y[ok][~vrai].mean()) if (~vrai).any() else None,
                "observe_moyen_si_haut": float(y[ok][vrai].mean()) if vrai.any() else None,
                "rmse_si_bas": rmse((q - y)[ok][~vrai]), "rmse_si_haut": rmse((q - y)[ok][vrai]),
                "calibration_deciles": calibration,
            }
        resultat[direction] = bloc
    resultat["moyenne"] = (resultat["directe"]["rmse_globale"] + resultat["inverse"]["rmse_globale"]) / 2
    (Path(run) / "regimes.json").write_text(json.dumps(resultat, indent=2) + "\n")
    return resultat


@torch.no_grad()
def calibration_regime(run):
    """Probabilité de régime haut prédite (moyenne de l'ensemble) contre fréquence observée de
    l'étiquette commune, par décile ; seuils lus dans les statistiques d'entraînement sérialisées."""
    resultat = {}
    for direction, (_, validation) in DIRECTIONS.items():
        test = donnees.lire_dossier_test(validation / "test")
        n = test["n_init"]
        probabilites, seuils = [], None
        for dossier in sorted(d for d in (Path(run) / direction).glob("membre*") if d.is_dir()):
            paquet, config, stats, modeles = monde.charger(dossier)
            if not config.get("_regime") or config["_regime"]["k"] != 2:
                continue
            seuils, cibles = stats["regime_seuils"], [paquet["circuits"][j] for j in config["_regime"]["circuits"]]
            f, _ = variables.construire(test["X"], config["variables"], config["sans"], test["dmop"],
                                        paquet.get("commandes"))
            x = torch.as_tensor(monde.normaliser_entrees(f, stats)[None], dtype=torch.float32)
            yi = torch.as_tensor(monde.normaliser_courants(test["y_init"][None, :n], stats["y_moyenne"],
                                                           stats["y_ecart"]), dtype=torch.float32)
            for m in modeles:
                logits = m.simuler(x[:, n:], m.etat_initial(x[:, :n], yi), m.resume_niveau(x[:, :n], yi),
                                   avec_logits=True)[2]
                probabilites.append(logits.softmax(-1)[0, :, 1].numpy())
        if seuils is None:
            return None
        v = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
        p = pd.Series(np.mean(probabilites, axis=0), index=test["heures"][n:]).reindex(v.index).to_numpy()
        y = v[cibles].to_numpy()
        ok = np.isfinite(y).all(axis=1) & np.isfinite(p)
        etiquette = monde.etiquettes_regime(torch.as_tensor(y[ok]), torch.as_tensor(seuils)).numpy()
        classe = np.clip((p[ok] * 10).astype(int), 0, 9)
        resultat[direction] = {
            "brier": float(np.mean((p[ok] - etiquette) ** 2)),
            "brier_frequence_constante": float(np.mean((etiquette.mean() - etiquette) ** 2)),
            "part_haut_observee": float(etiquette.mean()), "probabilite_moyenne": float(p[ok].mean()),
            "deciles": [{"p_predite": float(p[ok][classe == k].mean()), "frequence": float(etiquette[classe == k].mean()),
                         "n": int((classe == k).sum())} for k in range(10) if (classe == k).any()]}
    (Path(run) / "calibration_regime.json").write_text(json.dumps(resultat, indent=2) + "\n")
    return resultat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, action="append", default=[])
    ap.add_argument("--donnees", action="store_true", help="statistiques des régimes dans runs/regimes/")
    ap.add_argument("--calibration", action="store_true", help="calibration de la tête de régime (RUN/calibration_regime.json)")
    args = ap.parse_args()
    if args.donnees:
        statistiques_donnees(RACINE / "runs" / "regimes")
    for run in args.run:
        if args.calibration:
            print(run, json.dumps(calibration_regime(run)))
        r = erreurs_modele(run)
        print(run, f"directe {r['directe']['rmse_globale']:.6f} inverse {r['inverse']['rmse_globale']:.6f}"
              f" moyenne {r['moyenne']:.6f}")


if __name__ == "__main__":
    main()
