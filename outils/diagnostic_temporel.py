"""Diagnostics des validations autorisées et réinitialisations à dates décalées.

Les poids de chaque direction ont appris exclusivement sur l'autre année.
Chaque réinitialisation fournit exactement 168 heures de courants ; toutes les
autres consommations restent hors du chemin de prédiction. Aucun réentraînement
ni choix de variables sur la période évaluée.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dio import donnees  # noqa: E402
from dio import monde  # noqa: E402


def mesures(verite, prediction):
    masque = np.isfinite(verite)
    if not np.isfinite(prediction).all():
        raise ValueError("Prédictions non finies")
    erreur = (prediction - verite)[masque]
    return {"n": int(masque.sum()), "rmse": float(np.sqrt(np.mean(erreur ** 2))),
            "biais": float(erreur.mean())}


def circuits(verite, prediction):
    lignes = []
    total = np.isfinite(verite.to_numpy()).sum()
    for c in verite:
        y, p = verite[c].to_numpy(), prediction[c].to_numpy()
        ok = np.isfinite(y)
        m = mesures(y, p)
        m.update(circuit=c, contribution_mse=float(np.nansum((p-y)**2)/total),
                 ecart_observe=float(y[ok].std()), ecart_predit=float(p[ok].std()))
        lignes.append(m)
    return pd.DataFrame(lignes).sort_values("contribution_mse", ascending=False)


def diagnostiquer(run, direction, out):
    validation = Path("validation" if direction == "directe" else "validation_inverse")
    entrainement, evaluation = (("annee1", "annee2") if direction == "directe" else ("annee2", "annee1"))
    v = pd.read_csv(validation / "verite_horaire.csv", index_col="ut_ms")
    p = pd.read_csv(run / direction / "pred.csv", index_col="ut_ms")
    if p.index.has_duplicates or list(p.columns) != list(v.columns):
        raise ValueError("Colonnes ou timestamps invalides")
    p = p.reindex(v.index)
    circuits(v, p).to_csv(out / f"{direction}_circuits.csv", index=False)
    # Seuils définis sur l'entraînement ; la vérité évaluée ne sert qu'au diagnostic.
    _, x_train, y_train = donnees.charger_annee(entrainement)
    seuils = np.nanquantile(y_train, [.5, .9], axis=0)
    regimes_circuits = []
    for j, circuit in enumerate(v.columns):
        y_circuit = v[circuit].to_numpy()
        p_circuit = p[circuit].to_numpy()
        for nom, masque in (("bas", y_circuit <= seuils[0, j]),
                            ("intermediaire", (y_circuit > seuils[0, j]) & (y_circuit <= seuils[1, j])),
                            ("haut", y_circuit > seuils[1, j])):
            if masque.any():
                regimes_circuits.append({"circuit": circuit, "regime": nom,
                                         "mediane_train": seuils[0, j], "q90_train": seuils[1, j],
                                         **mesures(y_circuit[masque], p_circuit[masque])})
    pd.DataFrame(regimes_circuits).to_csv(out / f"{direction}_regimes.csv", index=False)
    lignes = []
    jours = (v.index.to_numpy() - v.index[0]) / (24 * donnees.H)
    for debut in range(0, int(jours.max()) + 1, 30):
        ok = (jours >= debut) & (jours < debut + 30)
        lignes.append({"jour": debut, **mesures(v.to_numpy()[ok], p.to_numpy()[ok])})
    pd.DataFrame(lignes).to_csv(out / f"{direction}_periodes.csv", index=False)
    _, x, _ = donnees.charger_annee(evaluation)
    x = x.reindex(v.index)
    conditions = {"saaf_absent": x.saaf_missing > 0,
                  "saaf_partiel": (x.saaf_coverage_fraction < .95) & (x.saaf_missing == 0),
                  "saaf_complet": x.saaf_coverage_fraction >= .95}
    contexte = {nom: mesures(v.to_numpy()[ok], p.to_numpy()[ok])
                for nom, ok in conditions.items() if ok.any()}
    modes = []
    for nom in x_train.columns:
        if not nom.startswith("ftl_") or not nom.endswith("_minutes"):
            continue
        if nom in ("ftl_coverage_minutes", "ftl_overlap_minutes", "ftl_comms_minutes"):
            continue
        frequence = float((x_train[nom] > 0).mean())
        ok = x[nom] > 0
        if ok.any():
            modes.append({"mode": nom, "frequence_train": frequence,
                          "rare_train": frequence < .01,
                          **mesures(v.to_numpy()[ok], p.to_numpy()[ok])})
    pd.DataFrame(modes).to_csv(out / f"{direction}_modes.csv", index=False)
    train = donnees.lire_dmop(entrainement)
    test = donnees.lire_dmop(evaluation)
    comptes = train.subsystem.value_counts()
    inconnus = test.loc[~test.subsystem.isin(comptes.index), "subsystem"].value_counts()
    inconnus.rename_axis("commande").rename("occurrences").to_csv(out / f"{direction}_commandes_inconnues.csv")
    # Seuil de rareté fixé sans consulter les résultats de validation.
    rares = set(comptes[comptes < 20].index)
    regimes = {}
    for nom, selection in (("inconnues", ~test.subsystem.isin(comptes.index)),
                           ("rares_entrainement", test.subsystem.isin(rares))):
        heures = (test.loc[selection, "ut_ms"].to_numpy() // donnees.H) * donnees.H
        ok = v.index.isin(heures)
        if ok.any():
            regimes[nom] = mesures(v.to_numpy()[ok], p.to_numpy()[ok])
    return {"global": mesures(v.to_numpy(), p.to_numpy()), "contextes": contexte,
            "commandes": regimes}


def reinitialisations(run, direction, out, jours_depart=(0, 120, 300, 480)):
    annee = "annee2" if direction == "directe" else "annee1"
    h, x, y = donnees.charger_annee(annee)
    dmop = donnees.lire_dmop(annee)
    membres = json.loads((run / direction / "membres.json").read_text())
    attendue = "annee1" if direction == "directe" else "annee2"
    for i in range(len(membres)):
        paquet = torch.load(run / direction / f"membre{i}/monde.pt", weights_only=True)
        if paquet["annees"] != [attendue]:
            raise ValueError("Poids non indépendants de l'année évaluée")
    lignes = []
    for jour in jours_depart:
        s, ti = jour * 24, 168
        if s + ti >= len(h):
            continue
        # Contexte planifié autorisé, mais aucun courant après les sept jours.
        xi = x.iloc[s:]
        yi = np.full((len(xi), y.shape[1]), np.nan)
        yi[:ti] = y[s:s+ti]
        predictions = [monde.simuler(run / direction / f"membre{i}", xi, yi, ti, dmop)[0]
                       for i in range(len(membres))]
        prediction = np.mean(predictions, axis=0)
        for horizon in (30, 90, 365):
            fin = min(ti + horizon * 24, len(xi))
            lignes.append({"direction": direction, "depart_jour": jour,
                           "depart_ut_ms": int(h[s]), "horizon_demande_jours": horizon,
                           "horizon_effectif_heures": fin-ti,
                           **mesures(y[s+ti:s+fin], prediction[ti:fin])})
    pd.DataFrame(lignes).to_csv(out / f"{direction}_reinitialisations.csv", index=False)
    return lignes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reinitialisations", action="store_true")
    args = ap.parse_args()
    out = args.run / "diagnostic"
    out.mkdir(exist_ok=True)
    resume = {}
    for direction in ("directe", "inverse"):
        resume[direction] = diagnostiquer(args.run, direction, out)
        if args.reinitialisations:
            reinitialisations(args.run, direction, out)
    (out / "resume.json").write_text(json.dumps(resume, indent=2, allow_nan=False) + "\n")
    print(json.dumps(resume, indent=2))


if __name__ == "__main__":
    main()
