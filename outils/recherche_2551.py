"""Recherche NPWD2551, programme 3 (docs/JOURNAL_2551.md) : sélection par validation croisée interne.

    python outils/recherche_2551.py modele --nom p3_xgb
    python outils/recherche_2551.py regime --nom p3_regime
    python outils/recherche_2551.py variables --nom p3_var --modele '{"xgb": {...}}' --jeux '{"base": {}, ...}'

Pour chaque direction (directe, inverse, chrono), tout réglage est choisi sur la seule période d'entraînement :
5 blocs contigus, 168 h d'embargo de chaque côté du bloc écarté, prédictions hors bloc (OOB). La validation
de la direction ne sert qu'à la prédiction finale (pred.csv), notée plus tard en local par outils/bilan_2551.py.
Variables de départ : celles du spécialiste installé (GBM_DEFAUT + orbite, dio/ensemble.SPECIALISTES).
Sorties : runs/NOM/DIRECTION/pred.csv (validation), oob.csv (OOB sur l'entraînement), choix.json (scores
internes de chaque réglage essayé, réglage retenu, durées).

- modele : XGBoost sur GPU, grille profondeur × min_child_weight × taux, arbres par arrêt anticipé sur le bloc
  écarté ; puis, pour le réglage retenu, 1 contre 5 (puis 10) graines et 1 modèle contre 4 sacs temporels ;
- regime : baselines/regime.py, grille du renouvellement du régime ; puis correction du spécialiste installé
  apprise sur ses résidus hors sac (entrees_2551/DIRECTION/hors_sac.csv, pred.csv), avec π du régime ;
- variables : un modèle figé, plusieurs jeux de variables (ajouts, retraits, bords_nan, sans_horloge).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from baselines import gbm_sacs, regime  # noqa: E402
from baselines.boosting import recaler  # noqa: E402
from dio import donnees  # noqa: E402
from dio.ensemble import GBM_DEFAUT, SPECIALISTES  # noqa: E402
from outils.valider import TOUTES  # noqa: E402

C = "NPWD2551"
H = donnees.H
DIRECTIONS = ("directe", "inverse", "chrono")
PLIS, EMBARGO_H, INIT_H = 5, 168, 168
CLES_PREPARER = {"jeu", "ajouts", "sans_horloge", "lissage_cible", "familles_variante", "retirer", "plan",
                 "contexte", "rapide", "bords_nan"}
OPTIONS_S = {k: v for k, v in {**GBM_DEFAUT, **SPECIALISTES[C]["gbm_specialiste"]}.items() if k in CLES_PREPARER}
# Grille XGBoost (taux, profondeur, min_child_weight, arbres au plus).
GRILLE_XGB = ([(0.03, d, m, 4000) for d in (4, 6, 8) for m in (20, 100, 500)]
              + [(0.1, d, 100, 1500) for d in (4, 6, 8)] + [(0.01, d, 100, 8000) for d in (4, 6)])
ARRET_XGB = 200
GAIN_GRAINES = (0.005, 0.002)  # 1 -> 5 graines, puis 5 -> 10 : gain OOB relatif minimal
RESIDU = {"max_iter": 3000, "learning_rate": 0.02, "max_leaf_nodes": 7, "min_samples_leaf": 500,
          "l2_regularization": 10.0, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20,
          "tol": 1e-7}
RENOUVELLEMENTS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rmse(e):
    return float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))


def charger(direction, variantes=None):
    """Variables d'entraînement et de test d'une direction (variantes : modifications de OPTIONS_S)."""
    variantes = dict(variantes or {})
    options = {**OPTIONS_S}
    options["ajouts"] = list(options["ajouts"]) + variantes.pop("ajouts", [])
    options["retirer"] = list(options["retirer"]) + variantes.pop("retirer", [])
    options.update(variantes)
    debut = time.time()
    schema = donnees.charger_schema()
    annees, validation = TOUTES[direction]
    annees = [annees] if isinstance(annees, str) else annees
    heures, table, y, meta = gbm_sacs.preparer(annees, [schema["target_columns"].index(C)], schema=schema, **options)
    test = donnees.lire_dossier_test(validation / "test")
    table_test = gbm_sacs.construire_test(test, meta)[table.columns]
    journal(f"{direction} : {table.shape[1]} variables, {len(table)} heures, {time.time() - debut:.0f} s")
    return {"heures": heures, "f": table.to_numpy(dtype=np.float64), "y": y[:, 0], "colonnes": list(table.columns),
            "test": test, "f_test": table_test.to_numpy(dtype=np.float64),
            "y_init": test["y_init"][:, schema["target_columns"].index(C)]}


def plis(n):
    return gbm_sacs.sacs_temporels(n, PLIS, EMBARGO_H)


def finir(d, prediction, dossier, recalage=True):
    """Recalage sur power_init comme le spécialiste installé (gbm_sacs.predire), borne 0, pred.csv."""
    if recalage:
        prediction = recaler(prediction[:, None], d["y_init"][:, None], d["test"]["n_init"], 1.0, 1440)[:, 0]
    dossier.mkdir(parents=True, exist_ok=True)
    donnees.ecrire_predictions(d["test"]["heures"], np.clip(prediction, 0, None)[:, None],
                               d["test"]["heures_a_predire"], [C], dossier / "pred.csv")


def ecrire_oob(d, oob, dossier, **autres):
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({C: oob, **autres}, index=pd.Index(d["heures"], name="ut_ms")).to_csv(dossier / "oob.csv")


# --- XGBoost ---------------------------------------------------------------------------------------------

def params_xgb(taux, profondeur, mcw, graine=0):
    import torch
    return {"objective": "reg:squarederror", "tree_method": "hist", "eta": taux, "max_depth": profondeur,
            "min_child_weight": mcw, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "seed": graine,
            "device": "cuda" if torch.cuda.is_available() else "cpu", "verbosity": 0}


def xgb_train(f, y, params, arbres, eval_f=None, eval_y=None):
    import xgboost
    k = np.isfinite(y)
    dtrain = xgboost.DMatrix(f[k], y[k])
    if eval_f is None:
        return xgboost.train(params, dtrain, arbres)
    ke = np.isfinite(eval_y)
    deval = xgboost.DMatrix(eval_f[ke], eval_y[ke])
    return xgboost.train(params, dtrain, arbres, evals=[(deval, "e")], early_stopping_rounds=ARRET_XGB,
                         verbose_eval=False)


def xgb_predict(modele, f, arbres=None):
    import xgboost
    plage = (0, arbres) if arbres else (0, 0)
    return modele.predict(xgboost.DMatrix(f), iteration_range=plage)


def cv_xgb(d, reglage):
    """OOB à l'arrêt anticipé de chaque pli ; nombre d'arbres retenu = moyenne des meilleurs."""
    taux, prof, mcw, arbres = reglage
    oob, meilleurs = np.full(len(d["y"]), np.nan), []
    for lignes, bloc in plis(len(d["y"])):
        m = xgb_train(d["f"][lignes], d["y"][lignes], params_xgb(taux, prof, mcw), arbres, d["f"][bloc], d["y"][bloc])
        meilleurs.append(m.best_iteration + 1)
        oob[bloc] = xgb_predict(m, d["f"][bloc], meilleurs[-1])
    return oob, meilleurs


def ensemble_xgb(f, y, reglage, arbres, graines=1, sacs=1):
    """Modèles à nombre d'arbres fixé : graines différentes et/ou sacs temporels (4 blocs, embargo 168 h)."""
    taux, prof, mcw, _ = reglage
    modeles = []
    for g in range(graines):
        decoupe = [(np.arange(len(y)), None)] if sacs == 1 else gbm_sacs.sacs_temporels(len(y), sacs, EMBARGO_H)
        for lignes, _ in decoupe:
            modeles.append(xgb_train(f[lignes], y[lignes], params_xgb(taux, prof, mcw, g), arbres))
    return modeles


def cv_ensemble(d, reglage, arbres_plis, graines, sacs):
    oob = np.full(len(d["y"]), np.nan)
    for (lignes, bloc), arbres in zip(plis(len(d["y"])), arbres_plis):
        modeles = ensemble_xgb(d["f"][lignes], d["y"][lignes], reglage, arbres, graines, sacs)
        oob[bloc] = np.mean([xgb_predict(m, d["f"][bloc]) for m in modeles], axis=0)
    return oob


def modele(nom):
    for direction in DIRECTIONS:
        d = charger(direction)
        dossier = RACINE / "runs" / nom / direction
        scores, debut = {}, time.time()
        for reglage in GRILLE_XGB:
            t = time.time()
            oob, meilleurs = cv_xgb(d, reglage)
            scores[str(reglage)] = {"rmse_oob": rmse(oob - d["y"]), "arbres": meilleurs, "s": round(time.time() - t)}
            journal(direction, reglage, f"OOB {scores[str(reglage)]['rmse_oob']:.4f}", meilleurs,
                    f"{time.time() - t:.0f} s")
            if scores[str(reglage)]["rmse_oob"] <= min(s["rmse_oob"] for s in scores.values()):
                retenu, oob_retenu = reglage, oob
        arbres_plis = scores[str(retenu)]["arbres"]
        base = scores[str(retenu)]["rmse_oob"]
        # Membres : graines 1 -> 5 -> 10 si le gain OOB relatif dépasse GAIN_GRAINES ; sacs temporels 1 contre 4.
        ensembles, graines = {}, 1
        for g, seuil in zip((5, 10), GAIN_GRAINES):
            o = cv_ensemble(d, retenu, arbres_plis, g, 1)
            ensembles[f"graines{g}"] = rmse(o - d["y"])
            precedent = base if graines == 1 else ensembles[f"graines{graines}"]
            journal(direction, f"{g} graines OOB {ensembles[f'graines{g}']:.4f} (avant {precedent:.4f})")
            if ensembles[f"graines{g}"] > precedent * (1 - seuil):
                break
            graines, oob_retenu = g, o
        o = cv_ensemble(d, retenu, arbres_plis, graines, 4)
        ensembles["sacs4"] = rmse(o - d["y"])
        actuel = rmse(oob_retenu - d["y"])
        sacs = 4 if ensembles["sacs4"] < actuel * (1 - GAIN_GRAINES[0]) else 1
        if sacs == 4:
            oob_retenu = o
        journal(direction, f"4 sacs OOB {ensembles['sacs4']:.4f} contre {actuel:.4f} : sacs {sacs}")
        arbres = int(round(np.mean(arbres_plis)))
        modeles = ensemble_xgb(d["f"], d["y"], retenu, arbres, graines, sacs)
        finir(d, np.mean([xgb_predict(m, d["f_test"]) for m in modeles], axis=0), dossier)
        ecrire_oob(d, oob_retenu, dossier)
        (dossier / "choix.json").write_text(json.dumps(
            {"grille": scores, "retenu": list(retenu), "arbres": arbres, "ensembles": ensembles, "graines": graines,
             "sacs": sacs, "rmse_oob": rmse(oob_retenu - d["y"]), "s": round(time.time() - debut)}, indent=1) + "\n")


# --- Régimes et correction des résidus --------------------------------------------------------------------

def cv_regime(d):
    """Composantes OOB par pli (classifieur, régressions), puis lissage de chaque bloc écarté avec ses
    INIT_H premières heures comme power_init (comme au test) ; ces heures ne sont pas notées."""
    n = len(d["y"])
    p, haut, bas = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    blocs = []
    for k, (lignes, bloc) in enumerate(plis(n)):
        t = time.time()
        m = regime.ajuster(d["f"][lignes], d["y"][lignes], d["heures"][lignes])
        p[bloc], haut[bloc], bas[bloc] = regime.composantes(m, d["f"][bloc])
        blocs.append((bloc, m["part_haut"]))
        journal(f"  pli {k} : {time.time() - t:.0f} s")
    scores, pis = {}, {}
    note = np.ones(n, bool)
    for bloc, _ in blocs:
        note[bloc[:INIT_H]] = False
    for r in RENOUVELLEMENTS:
        pi = np.full(n, np.nan)
        for bloc, part in blocs:
            obs = regime.observes_init(d["heures"][bloc], d["y"][bloc], INIT_H)
            pi[bloc] = regime.lisser(p[bloc], d["heures"][bloc], r, part, obs)
        pis[r] = pi
        e = regime.melanger(pi, haut, bas) - d["y"]
        scores[r] = rmse(e[note])
    return scores, pis, (p, haut, bas), note


def regime_residu(nom):
    for direction in DIRECTIONS:
        d = charger(direction)
        debut = time.time()
        scores, pis, (p, haut, bas), note = cv_regime(d)
        r = min(scores, key=scores.get)
        journal(direction, "OOB par renouvellement", {k: round(v, 4) for k, v in scores.items()}, "retenu", r)
        dossier = RACINE / "runs" / nom / direction
        oob = np.where(note, regime.melanger(pis[r], haut, bas), np.nan)
        ecrire_oob(d, oob, dossier, pi=np.where(note, pis[r], np.nan), p=p, haut=haut, bas=bas)
        m = regime.ajuster(d["f"], d["y"], d["heures"])
        pt, ht, bt = regime.composantes(m, d["f_test"])
        pi_t = regime.lisser(pt, d["test"]["heures"], r, m["part_haut"],
                             regime.observes_init(d["test"]["heures"], d["y_init"], d["test"]["n_init"]))
        finir(d, regime.melanger(pi_t, ht, bt), dossier, recalage=False)
        pd.DataFrame({"p": pt, "pi": pi_t, "haut": ht, "bas": bt},
                     index=pd.Index(d["test"]["heures"], name="ut_ms")).to_csv(dossier / "composantes.csv")
        choix = {"renouvellement": scores, "retenu": r, "rmse_oob": scores[r], "s": round(time.time() - debut)}
        (dossier / "choix.json").write_text(json.dumps(choix, indent=1) + "\n")
        residu(d, np.where(note, pis[r], np.nan), pi_t, RACINE / "runs" / f"{nom}_residu" / direction, direction)


def entrees_s(direction, partie, heures):
    """Prédictions de S (hs : hors sac sur l'entraînement, pr : test), entrees_2551/DIRECTION.npz : heures en
    écarts int16, valeurs × 1e4 en int16 (le code envoyé à Kaggle doit rester petit)."""
    a = np.load(RACINE / "entrees_2551" / f"{direction}.npz")
    h = np.concatenate([a[f"{partie}_h0"], a[f"{partie}_h0"] + np.cumsum(a[f"{partie}_dh"].astype(np.int64))])
    v = np.where(a[f"{partie}_nan"], np.nan, a[f"{partie}_v"] / 1e4)
    return pd.Series(v, index=h * H).reindex(heures).to_numpy()


def residu(d, pi_oob, pi_test, dossier, direction):
    """Correction du spécialiste installé S : cible y − S_oob (prédictions hors sac de S sur l'entraînement),
    variables de S + S + π ; prédiction S_test + correction. Score interne : OOB de la correction (mêmes plis)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    s_oob, s_test = entrees_s(direction, "hs", d["heures"]), entrees_s(direction, "pr", d["test"]["heures"])
    f = np.column_stack([d["f"], s_oob, pi_oob])
    f_test = np.column_stack([d["f_test"], s_test, pi_test])
    cible = d["y"] - s_oob
    debut = time.time()
    oob = np.full(len(cible), np.nan)
    for lignes, bloc in plis(len(cible)):
        k = lignes[np.isfinite(cible[lignes])]
        oob[bloc] = HistGradientBoostingRegressor(**RESIDU, random_state=0).fit(f[k], cible[k]).predict(f[bloc])
    k = np.isfinite(cible)
    correction = HistGradientBoostingRegressor(**RESIDU, random_state=0).fit(f[k], cible[k]).predict(f_test)
    dossier.mkdir(parents=True, exist_ok=True)
    corrige = np.clip(s_oob + oob, 0, None)
    ecrire_oob(d, corrige, dossier, s=s_oob)
    heures_test = pd.Index(d["test"]["heures"])
    p = pd.Series(np.clip(s_test + correction, 0, None), index=heures_test).reindex(d["test"]["heures_a_predire"])
    if p.isna().any():
        raise ValueError("Prédiction S manquante")
    p.rename(C).rename_axis("ut_ms").to_frame().to_csv(dossier / "pred.csv")
    choix = {"rmse_oob_s": rmse(s_oob - d["y"]), "rmse_oob_corrige": rmse(corrige - d["y"]),
             "s": round(time.time() - debut)}
    journal(direction, "résidu", choix)
    (dossier / "choix.json").write_text(json.dumps(choix, indent=1) + "\n")


# --- Jeux de variables, modèle figé --------------------------------------------------------------------

def cv_fige(d, spec):
    """spec : {"xgb": [taux, profondeur, mcw, arbres_max]} (arrêt anticipé par pli, comme modele) ou
    {"hgb": "D3"} (réglage de gbm_sacs.REGLAGES, arrêt anticipé interne). Retourne OOB et prédiction test."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    oob = np.full(len(d["y"]), np.nan)
    if "xgb" in spec:
        oob, meilleurs = cv_xgb(d, tuple(spec["xgb"]))
        m = xgb_train(d["f"], d["y"], params_xgb(*spec["xgb"][:3]), int(round(np.mean(meilleurs))))
        return oob, xgb_predict(m, d["f_test"]), meilleurs
    params = {"early_stopping": False, **gbm_sacs.REGLAGES[spec["hgb"]]}
    for lignes, bloc in plis(len(d["y"])):
        k = lignes[np.isfinite(d["y"][lignes])]
        c = utiles(d["f"][k])
        oob[bloc] = HistGradientBoostingRegressor(**params, random_state=0).fit(d["f"][k][:, c], d["y"][k]).predict(
            d["f"][bloc][:, c])
    k = np.isfinite(d["y"])
    c = utiles(d["f"][k])
    m = HistGradientBoostingRegressor(**params, random_state=0).fit(d["f"][k][:, c], d["y"][k])
    return oob, m.predict(d["f_test"][:, c]), None


def utiles(f):
    """Colonnes ayant au moins deux valeurs finies distinctes sur ces lignes (une colonne d'états entièrement
    inconnue dans un pli fait échouer le découpage en classes de HGB)."""
    finies = np.where(np.isfinite(f), f, np.nan)
    return np.nanmax(finies, axis=0, initial=-np.inf) > np.nanmin(finies, axis=0, initial=np.inf)


def jeux_variables(nom, spec, jeux, directions=DIRECTIONS):
    for direction in directions:
        for jeu, variantes in jeux.items():
            debut = time.time()
            d = charger(direction, variantes)
            oob, pred, meilleurs = cv_fige(d, spec)
            dossier = RACINE / "runs" / f"{nom}_{jeu}" / direction
            finir(d, pred, dossier)
            ecrire_oob(d, oob, dossier)
            choix = {"variantes": variantes, "modele": spec, "rmse_oob": rmse(oob - d["y"]), "arbres": meilleurs,
                     "variables": len(d["colonnes"]), "s": round(time.time() - debut)}
            journal(direction, jeu, choix)
            (dossier / "choix.json").write_text(json.dumps(choix, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["modele", "regime", "variables"])
    ap.add_argument("--nom", required=True)
    ap.add_argument("--modele", default='{"hgb": "D3"}')
    ap.add_argument("--jeux", default='{"base": {}}')
    ap.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=DIRECTIONS)
    a = ap.parse_args()
    if a.action == "modele":
        modele(a.nom)
    elif a.action == "regime":
        regime_residu(a.nom)
    else:
        jeux_variables(a.nom, json.loads(a.modele), json.loads(a.jeux), a.directions)


if __name__ == "__main__":
    main()
