"""Recherche NPWD2551, programme 5 (docs/JOURNAL_2551.md) : résolution brute et variables manquantes.

    python outils/programme5.py preparer                       # local : data/prepared/p5/ (cibles fines)
    python outils/programme5.py variables --nom p5v --jeux '{"base": {}, "lentes": {"ajouts": ["lentes"]}}'
    python outils/programme5.py lignes --nom p5_lignes         # un boosting par ligne de chauffage, somme
    python outils/programme5.py fin --nom p5_fin               # boosting sur la grille de 10 min
    python outils/programme5.py thermo --nom p5_thermo         # thermostat à faible capacité, seul et variable de S
    python outils/programme5.py bilan --nom p5_bilan --candidats lentes=p5v_lentes:p5v_base   # local, 5 cas
    python outils/programme5.py arbres --nom p5a --modeles lgb cat   # LightGBM / CatBoost sur les entrées de S

Même protocole que le programme 3 (outils/recherche_2551.py) : par direction, 5 blocs contigus de l'entraînement
avec 168 h d'embargo (oob.csv), puis modèle sur tout l'entraînement et prédiction de la validation (pred.csv),
notée plus tard en local par outils/bilan_2551.py. Réglage figé de sélection : D3 (comme le programme 3) ; les finalistes sont réentraînés en D3L à 5 sacs.
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
from baselines import fin_2551, gbm_sacs, thermostat  # noqa: E402
from outils import recherche_2551 as r  # noqa: E402

SPEC = {"hgb": "D3"}  # comme le programme 3 (runs/p3_var_base) ; finalistes réentraînés en D3L, 5 sacs
FIN = {**gbm_sacs.REGLAGES["D3"], "max_bins": 255}


def ecrire(d, oob, pred, dossier, debut, **infos):
    r.finir(d, pred, dossier)
    r.ecrire_oob(d, oob, dossier)
    choix = {**infos, "rmse_oob": r.rmse(oob - d["y"]), "variables": len(d["colonnes"]), "s": round(time.time() - debut)}
    r.journal(dossier.parent.name, dossier.name, choix)
    (dossier / "choix.json").write_text(json.dumps(choix, indent=1) + "\n")


def lignes(nom, variantes, directions):
    """Une cible par ligne de chauffage (moyenne horaire de A, B, C) ; prédiction = somme des trois."""
    cibles = pd.read_csv(fin_2551.DOSSIER / "lignes.csv", index_col="ut_ms")
    for direction in directions:
        debut = time.time()
        d = r.charger(direction, variantes)
        oob, pred, parts = np.zeros(len(d["y"])), np.zeros(len(d["f_test"])), {}
        for k in cibles.columns:
            y = cibles[k].reindex(d["heures"]).to_numpy()
            o, p, _ = r.cv_fige({**d, "y": y}, SPEC)
            parts[k] = r.rmse(o - y)
            oob, pred = oob + o, pred + p
        ecrire(d, oob, pred, RACINE / "runs" / nom / direction, debut, rmse_oob_lignes=parts, variantes=variantes)


def fin(nom, variantes, directions):
    """Boosting sur la grille de 10 min : variables horaires de S répétées + variables à la minute ; la
    prédiction horaire est la moyenne des 6 cases."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    dix = pd.read_csv(fin_2551.DOSSIER / "dix.csv", index_col="ut_ms")[fin_2551.C]
    for direction in directions:
        debut = time.time()
        d = r.charger(direction, variantes)
        annees = r.TOUTES[direction][0]
        annees = [annees] if isinstance(annees, str) else annees
        t10 = fin_2551.grille(d["heures"])
        minute, n0 = [], 0
        for a in annees:  # variables à la minute : une période continue à la fois
            n = len(r.donnees.charger_annee(a)[0])
            minute.append(fin_2551.variables_minute(r.donnees.lire_contextes(a), t10[n0 * 6:(n0 + n) * 6]))
            n0 += n
        f = np.hstack([np.repeat(d["f"], 6, axis=0), pd.concat(minute).to_numpy()]).astype(np.float32)
        y = dix.reindex(t10).to_numpy()
        t10_test = fin_2551.grille(d["test"]["heures"])
        f_test = np.hstack([np.repeat(d["f_test"], 6, axis=0),
                            fin_2551.variables_minute(d["test"]["tables"], t10_test).to_numpy()]).astype(np.float32)
        r.journal(direction, "grille 10 min", f.shape, f"{np.isfinite(y).mean():.3f} connues")
        oob = np.full(len(y), np.nan)
        case = lambda idx: (idx[:, None] * 6 + np.arange(6)).ravel()  # noqa: E731
        for lignes_h, bloc_h in r.plis(len(d["y"])):
            k = case(lignes_h)
            k = k[np.isfinite(y[k])]
            b = case(bloc_h)
            oob[b] = HistGradientBoostingRegressor(**FIN, random_state=0).fit(f[k], y[k]).predict(f[b])
        k = np.isfinite(y)
        modele = HistGradientBoostingRegressor(**FIN, random_state=0).fit(f[k], y[k])
        ecrire(d, fin_2551.vers_heures(oob), fin_2551.vers_heures(modele.predict(f_test)),
               RACINE / "runs" / nom / direction, debut, rmse_oob_10min=r.rmse(oob - y), arbres=int(modele.n_iter_))


def thermo(nom, directions):
    """Thermostat à faible capacité (baselines/thermostat.py), une sortie par ligne de chauffage. NOM : somme
    des trois sorties, seule (oob.csv : en échantillon, le modèle a une cinquantaine de paramètres) ; NOM_s :
    les trois sorties ajoutées aux variables de S (en échantillon sur l'entraînement), réglage SPEC."""
    cibles = pd.read_csv(fin_2551.DOSSIER / "lignes.csv", index_col="ut_ms")
    for direction in directions:
        debut = time.time()
        d = r.charger(direction, {"ajouts": ["thermique"]})
        th = np.array([c.startswith("th_") for c in d["colonnes"]])
        m = thermostat.ajuster(d["f"][:, th], cibles.reindex(d["heures"]).to_numpy(),
                               [c for c in d["colonnes"] if c.startswith("th_")])
        p, p_test = thermostat.predire(m, d["f"][:, th]), thermostat.predire(m, d["f_test"][:, th])
        infos = {"pi": {k: dict(zip(map(str, m["taus"]), np.round(np.exp(v) / np.exp(v).sum(), 3).tolist()))
                        for k, v in zip(cibles.columns, m["params"]["pi"])},
                 "w": {k: dict(zip(m["series"], np.round(v, 3).tolist())) for k, v in zip(cibles.columns, m["params"]["w"])}}
        ecrire(d, p.sum(axis=1), p_test.sum(axis=1), RACINE / "runs" / nom / direction, debut, **infos)
        debut = time.time()
        d2 = {**d, "f": np.hstack([d["f"][:, ~th], p]), "f_test": np.hstack([d["f_test"][:, ~th], p_test]),
              "colonnes": [c for c in d["colonnes"] if not c.startswith("th_")] + [f"thermo_{k}" for k in cibles.columns]}
        oob, pred, _ = r.cv_fige(d2, SPEC)
        ecrire(d2, oob, pred, RACINE / "runs" / f"{nom}_s" / direction, debut)


def arbres(nom, modeles, variantes, directions):
    """LightGBM et CatBoost (GPU si présent) sur les entrées de S, arbres par arrêt anticipé sur le bloc écarté."""
    for direction in directions:
        d = r.charger(direction, variantes)
        for m in modeles:
            debut = time.time()
            oob, iters = np.full(len(d["y"]), np.nan), []
            for lignes_h, bloc in r.plis(len(d["y"])):
                k = lignes_h[np.isfinite(d["y"][lignes_h])]
                kb = bloc[np.isfinite(d["y"][bloc])]
                modele = ajuster(m, d["f"][k], d["y"][k], d["f"][kb], d["y"][kb])
                iters.append(modele[1])
                oob[bloc] = modele[0](d["f"][bloc])
            k = np.isfinite(d["y"])
            final = ajuster(m, d["f"][k], d["y"][k], None, None, int(np.mean(iters)))
            ecrire(d, oob, final[0](d["f_test"]), RACINE / "runs" / f"{nom}_{m}" / direction, debut, arbres=iters)


def ajuster(m, f, y, fv, yv, arbres=None):
    """(fonction de prédiction, nombre d'arbres). arbres fixé : pas d'arrêt anticipé."""
    if m == "lgb":
        import lightgbm
        modele = lightgbm.LGBMRegressor(n_estimators=arbres or 6000, learning_rate=0.01, num_leaves=15,
                                        min_child_samples=100, reg_lambda=3.0, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.5, verbose=-1, random_state=0)
        if arbres:
            modele.fit(f, y)
            return modele.predict, arbres
        modele.fit(f, y, eval_set=[(fv, yv)], callbacks=[lightgbm.early_stopping(200, verbose=False)])
        return modele.predict, int(modele.best_iteration_)
    import catboost
    import torch
    gpu = torch.cuda.is_available()
    modele = catboost.CatBoostRegressor(iterations=arbres or 6000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                                        task_type="GPU" if gpu else "CPU", verbose=0, random_seed=0,
                                        **({} if arbres else {"early_stopping_rounds": 200}))
    if arbres:
        modele.fit(f, y)
        return modele.predict, arbres
    modele.fit(f, y, eval_set=(fv, yv))
    return modele.predict, int(modele.get_best_iteration() + 1)


def bilan(nom, cands):
    """Mesure externe (outils/bilan_2551.py, 5 cas) de chaque candidat NOM=RUN:REF:PART : livré + PART × (RUN − REF),
    borné à 0. REF : run dont la part est remplacée (défaut : S installé exact, PART 0.5 = remplacement de la part
    boosting ; REF peut contenir {g}, groupe de graines). Règle d'installation inchangée (bilan_2551.regle)."""
    from outils import bilan_2551 as b
    b.SPECIALISTE = "b0_gel/specialiste_exact"
    toutes = b.cas(["directe", "inverse", "chrono"])
    livre = {f"{v}/{g}": b.mesures_cas(b.base(v, g), v) for v, g in toutes}
    sortie = {"livre": {c: m["rmse"] for c, m in livre.items()}}
    print(f"{'':34s}" + "".join(f"{c:>12s}" for c in livre) + "   interne (dir/inv/chr)   règle")
    print(f"{'livré':34s}" + "".join(f"{m['rmse']:12.4f}" for m in livre.values()))
    for nom_c, spec in cands.items():
        run, ref, part = (spec.split(":") + ["", ""])[:3]
        part = float(part or 0.5)
        apres, seul = {}, {}
        for v, g in toutes:
            reference = b.candidat(ref.format(g=g), v) if ref else b.specialiste(v)
            c = b.candidat(run, v)
            seul[f"{v}/{g}"] = b.mesures(c, v, False)["rmse"]
            apres[f"{v}/{g}"] = b.mesures_cas(np.clip(b.base(v, g) + part * (c - reference), 0, None), v)
        interne = []
        for v in ("directe", "inverse", "chrono"):
            f = RACINE / "runs" / run / v / "choix.json"
            interne.append(json.loads(f.read_text()).get("rmse_oob") if f.exists() else None)
        rg = b.regle(livre, apres)
        sortie[nom_c] = {"spec": spec, "seul": seul, "apres": {c: m["rmse"] for c, m in apres.items()},
                         "interne": interne, "regle": rg}
        print(f"{nom_c + ' seul':34s}" + "".join(f"{x:12.4f}" for x in seul.values()))
        print(f"{nom_c + f' ×{part:g}':34s}" + "".join(f"{m['rmse']:12.4f}" for m in apres.values())
              + "   " + "/".join("-" if x is None else f"{x:.4f}" for x in interne)
              + f"   {rg['gain_moyen']:+.5f} {'ACCEPTÉ' if rg['accepte'] else 'refusé : ' + '; '.join(rg['motifs'])}")
    dossier = RACINE / "runs" / nom
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "bilan.json").write_text(json.dumps(sortie, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["preparer", "variables", "lignes", "fin", "arbres", "thermo", "bilan"])
    ap.add_argument("--nom")
    ap.add_argument("--candidats", nargs="*", default=[], help="bilan : NOM=RUN[:REF[:PART]]")
    ap.add_argument("--jeux", default='{"base": {}}')
    ap.add_argument("--variantes", default="{}")
    ap.add_argument("--modeles", nargs="+", default=["lgb", "cat"])
    ap.add_argument("--directions", nargs="+", default=list(r.DIRECTIONS), choices=r.DIRECTIONS)
    a = ap.parse_args()
    variantes = json.loads(a.variantes)
    if a.action == "preparer":
        fin_2551.preparer()
    elif a.action == "variables":
        r.jeux_variables(a.nom, SPEC, json.loads(a.jeux), a.directions)
    elif a.action == "lignes":
        lignes(a.nom, variantes, a.directions)
    elif a.action == "bilan":
        bilan(a.nom, dict(c.split("=", 1) for c in a.candidats))
    elif a.action == "thermo":
        thermo(a.nom, a.directions)
    elif a.action == "fin":
        fin(a.nom, variantes, a.directions)
    else:
        arbres(a.nom, a.modeles, variantes, a.directions)


if __name__ == "__main__":
    main()
