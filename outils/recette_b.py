"""Programme 4 : reproduction de la recette B pour NPWD2551 (docs/JOURNAL_2551.md, baselines/recette_b.py).

    python outils/recette_b.py geler                       # empreintes du livré, spécialiste exact (local)
    python outils/recette_b.py controle                    # entrées année 1 vs controles/recette_b (local)
    python outils/recette_b.py bords                       # vocabulaires, masques, bords de la chrono (local)
    python outils/recette_b.py matrices --sortie /tmp/b_mat              # env principal (Kaggle)
    python outils/recette_b.py entrainer --env-recette --matrices /tmp/b_mat --cellules rec_recette_coll ...

Cellules : ENTREES_REGLAGES_MASQUE, entrées cur (matrice du spécialiste livré, gbm_sacs) ou rec (recette B),
réglages recette (PARAMS_B) ou d3l (livré), masque coll (33 cibles) ou propre (NPWD2551 seule). Chaque
cellule écrit runs/PREFIXE_CELLULE/DIRECTION/ : pred.csv (sans recalage), pred_recale.csv (recalage power_init
du livré, demi-vie 1440 h), oob.csv (hors bloc), info.json (empreintes, lignes, itérations, temps, versions).
--env-recette : relance dans un environnement Python 3.11 épinglé (numpy 1.26.4, pandas 2.3.3,
scikit-learn 1.6.1), créé au besoin avec uv ; les entrées recette y sont reconstruites et comparées à celles
de l'environnement principal. Repli sur l'environnement principal si sa création échoue (consigné).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from baselines import recette_b as rb  # noqa: E402

CONTROLES = RACINE / "controles" / "recette_b"
ENV = Path(os.environ.get("ENV_RECETTE_B", "/tmp/env_recette_b"))
PAQUETS = ["numpy==1.26.4", "pandas==2.3.3", "scikit-learn==1.6.1", "scipy==1.13.1", "joblib==1.6.0",
           "threadpoolctl==3.8.0"]
# Périodes d'entraînement et dossiers de test des trois validations (outils/valider.TOUTES).
DIRECTIONS = {"directe": (["data/annee1"], "validation"), "inverse": (["data/annee2"], "validation_inverse"),
              "chrono": (["data/annee1", "validation_chrono/entrainement"], "validation_chrono")}
ANNEES_GBM = {"directe": ["annee1"], "inverse": ["annee2"], "chrono": ["annee1", "validation_chrono/entrainement"]}


def empreinte(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fichier_sha(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def versions():
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "executable": sys.executable}


def ecrire(chemin, contenu):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=1, default=str) + "\n")


def coupure_chrono():
    return int(pd.read_csv(RACINE / "validation_chrono/test/power_init.csv", usecols=["ut_ms"]).ut_ms.min() // rb.H * rb.H)


# ---------------------------------------------------------------- étape 1 : geler le livré

def geler():
    sortie = RACINE / "runs" / "b0_gel"
    if (sortie / "empreintes.json").exists():
        sys.exit(f"{sortie} existe déjà : rien n'est écrasé")
    from baselines import gbm_sacs
    refs = [f"runs/rc0_livre_{g}/{v}/pred.csv" for g in ("g04", "g59") for v in ("directe", "inverse")]
    refs += ["runs/rc7_chrono_mixte/chrono/pred.csv", 'runs/rc7_chrono_g2/chrono/pred_{"recalage": 1.0}.csv',
             "runs/n2_base_chrono/chrono/pred.csv"] + [f"runs/n2_spec/{v}/pred.csv" for v in DIRECTIONS]
    contenu = {"poids": {str(f.relative_to(RACINE)): fichier_sha(f) for f in sorted((RACINE / "poids").rglob("*"))
                         if f.is_file()},
               "predictions": {r: fichier_sha(RACINE / r) for r in refs},
               "git": subprocess.run(["git", "log", "-1", "--format=%H %s"], cwd=RACINE, capture_output=True,
                                     text=True).stdout.strip(),
               "git_diff_stat": subprocess.run(["git", "diff", "--stat"], cwd=RACINE, capture_output=True,
                                               text=True).stdout}
    for r in refs:
        cible = sortie / "copies" / r.removeprefix("runs/")
        cible.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(RACINE / r, cible)
        cible.chmod(0o444)
    # Spécialiste exact installé dans le livré de chaque direction (identique pour g04 et g59), re-prédit.
    for v in ("directe", "inverse"):
        d = sortie / "specialiste_exact" / v
        d.mkdir(parents=True, exist_ok=True)
        gbm_sacs.predire(RACINE / DIRECTIONS[v][1] / "test", d / "pred.csv",
                         RACINE / f"runs/rc0_livre_g04/{v}/specialiste/gbm_NPWD2551")
    shutil.copytree(RACINE / "runs/n2_base_chrono/chrono", sortie / "specialiste_exact" / "chrono",
                    ignore=shutil.ignore_patterns("*.pkl*", "*.log"))
    contenu["specialiste_exact"] = {v: fichier_sha(sortie / "specialiste_exact" / v / "pred.csv") for v in DIRECTIONS}
    ecrire(sortie / "empreintes.json", contenu)
    print(json.dumps(contenu["specialiste_exact"], indent=1))


# ---------------------------------------------------------------- étape 3 : contrôles

def controle():
    ref = json.loads((CONTROLES / "controle_reference.json").read_text())
    vocab = json.loads((CONTROLES / "vocabulaire_reference.json").read_text())
    colonnes = json.loads((CONTROLES / "colonnes_reference.json").read_text())
    debut = time.time()
    g, x, _, _ = rb.periode(RACINE / "data/annee1", vocab)
    duree = time.time() - debut
    a = x.to_numpy(np.float32)
    echantillon = pd.read_csv(CONTROLES / "controle_entrees_annee1.csv", index_col="ut_ms")
    rapport = {"versions": versions(), "duree_s": round(duree, 1), "shape": list(a.shape), "dtype": str(a.dtype),
               "colonnes_identiques": list(x.columns) == colonnes, "grille": [int(g[0]), int(g[-1])],
               "grille_ok": [int(g[0]), int(g[-1])] == [ref["grid_first"], ref["grid_last"]],
               "sha256_X": empreinte(a), "sha256_X_ok": empreinte(a) == ref["sha256_float32_C"],
               "sha256_ut_ms_ok": empreinte(g.astype(np.int64)) == ref["sha256_ut_ms_int64"]}
    if rapport["colonnes_identiques"]:
        b = x.loc[echantillon.index, echantillon.columns].to_numpy(np.float64)
        e = echantillon.to_numpy(np.float64)
        faux = ~np.isclose(b, e, atol=ref["tolerance_echantillon_atol"], rtol=ref["tolerance_echantillon_rtol"])
        rapport["echantillon"] = {"lignes": len(e), "cellules_hors_tolerance": int(faux.sum()),
                                  "ecart_max": float(np.max(np.abs(b - e)))}
        rapport["divergences"] = [{"colonne": echantillon.columns[j], "ut_ms": int(echantillon.index[i]),
                                   "calcule": float(b[i, j]), "reference": float(e[i, j])}
                                  for i, j in zip(*np.nonzero(faux))][:20]
    else:
        premiere = next((i for i, (c, r) in enumerate(zip(x.columns, colonnes)) if c != r), min(len(x.columns), len(colonnes)))
        rapport["premiere_colonne_divergente"] = {"position": premiere, "calcule": list(x.columns)[premiere:premiere + 3],
                                                  "reference": colonnes[premiere:premiere + 3]}
    appris = rb.apprendre([rb.lire(RACINE / "data/annee1"), rb.lire(RACINE / "data/annee2")])
    rapport["vocabulaire_annees12"] = {k: {"identique": appris[k] == vocab[k], "taille": len(appris[k])} for k in vocab}
    nom = "controle_" + ("epingle" if np.__version__ == "1.26.4" else "principal") + ".json"
    ecrire(RACINE / "runs/b1_controle" / nom, rapport)
    print(json.dumps({k: v for k, v in rapport.items() if k != "divergences"}, indent=1))


def construire_rec(direction, bord=False):
    """Entrées recette d'une validation : vocabulaire de ses seules périodes d'entraînement.
    bord (chrono) : annee2 entière construite puis tranchée à la coupure (référence de la recette)."""
    periodes, test_dir = DIRECTIONS[direction]
    if bord:
        c = coupure_chrono()
        a1, a2 = rb.lire(RACINE / "data/annee1"), rb.lire(RACINE / "data/annee2")
        vocab = rb.apprendre([a1, a2], coupure=c)
        g1, x1, circuits, y1 = rb.periode(RACINE / "data/annee1", vocab)
        g2, x2, _, y2 = rb.periode(RACINE / "data/annee2", vocab, coupure=c)
        g_all, _, _ = rb.cibles(RACINE / "data/annee2/power.csv")
        gt, xt, a_predire, _, _, _ = rb.test(RACINE / test_dir / "test", vocab)
        xt = rb.construire(a2, g_all, vocab).loc[gt]  # même grille, contexte de l'année entière
        return {"vocab": vocab, "g": np.concatenate([g1, g2]), "x": pd.concat([x1, x2]), "y": np.concatenate([y1, y2]),
                "circuits": circuits, "gt": gt, "xt": xt, "a_predire": a_predire}
    vocab = rb.apprendre([rb.lire(RACINE / p) for p in periodes])
    parts = [rb.periode(RACINE / p, vocab) for p in periodes]
    gt, xt, a_predire, _, _, _ = rb.test(RACINE / test_dir / "test", vocab)
    return {"vocab": vocab, "g": np.concatenate([p[0] for p in parts]), "x": pd.concat([p[1] for p in parts]),
            "y": np.concatenate([p[3] for p in parts]), "circuits": parts[0][2], "gt": gt, "xt": xt,
            "a_predire": a_predire}


def comparer(a, b, heures):
    """Cellules différentes entre deux X de mêmes colonnes et lignes : par colonne, écart max, heures touchées."""
    d = ~np.isclose(a.to_numpy(np.float64), b.to_numpy(np.float64), atol=0, rtol=0)
    lignes = np.flatnonzero(d.any(axis=1))
    par_col = {c: int(n) for c, n in zip(a.columns, d.sum(axis=0)) if n}
    blocs = {}
    for c, n in par_col.items():
        blocs[c.split("_")[0] + "_" + c.split("_")[1]] = blocs.get(c.split("_")[0] + "_" + c.split("_")[1], 0) + n
    return {"cellules": int(d.sum()), "lignes": len(lignes), "colonnes": len(par_col),
            "premiere_heure": int(heures[lignes[0]]) if len(lignes) else None,
            "derniere_heure": int(heures[lignes[-1]]) if len(lignes) else None,
            "heures_depuis_debut_max": int(lignes.max()) if len(lignes) else None,
            "ecart_max": float(np.max(np.abs(a.to_numpy(np.float64) - b.to_numpy(np.float64)))) if len(lignes) else 0.0,
            "par_bloc": dict(sorted(blocs.items(), key=lambda kv: -kv[1]))}


def bords():
    """Vocabulaires propres à chaque validation, masques, différences de bord des dossiers découpés."""
    rapport = {"versions": versions(), "coupure_chrono": coupure_chrono()}
    vocabs, voc_ref = {}, json.loads((CONTROLES / "vocabulaire_reference.json").read_text())
    for v in DIRECTIONS:
        r = construire_rec(v)
        vocabs[v] = r["vocab"]
        k = r["circuits"].index(rb.CIBLE)
        rapport[v] = {"lignes_entrainement": len(r["g"]), "colonnes": r["x"].shape[1],
                      "lignes_test": len(r["gt"]), "heures_a_predire": len(r["a_predire"]),
                      "masque_collectif": int(rb.masque(r["y"], r["circuits"]).sum()),
                      "masque_propre": int(rb.masque(r["y"], r["circuits"], False).sum()),
                      "cible_connue_hors_masque_collectif": int((np.isfinite(r["y"][:, k])
                                                                 & ~rb.masque(r["y"], r["circuits"])).sum()),
                      "tailles_vocab": {c: len(x) for c, x in r["vocab"].items()},
                      "vocab_identique_controle": {c: r["vocab"][c] == voc_ref[c] for c in voc_ref}}
        if v in ("directe", "inverse"):  # test = année entière : identique à l'année construite sur sa grille
            annee = "data/annee2" if v == "directe" else "data/annee1"
            g, x, _, _ = rb.periode(RACINE / annee, r["vocab"])
            commun = np.intersect1d(g, r["gt"])
            rapport[v]["bord_test_vs_annee_entiere"] = comparer(r["xt"].loc[commun], x.loc[commun], commun)
        else:
            b = construire_rec(v, bord=True)
            rapport[v]["vocab_identique_bord"] = {c: r["vocab"][c] == b["vocab"][c] for c in r["vocab"]}
            if list(r["x"].columns) == list(b["x"].columns):
                n1 = int((r["g"] < 1278720000000).sum())
                rapport[v]["bord_entrainement_A2a"] = comparer(r["x"].iloc[n1:], b["x"].iloc[n1:], r["g"][n1:])
                rapport[v]["bord_test_A2b"] = comparer(r["xt"], b["xt"], r["gt"])
    ecrire(RACINE / "runs/b1_controle/bords.json", rapport)
    ecrire(RACINE / "runs/b1_controle/vocabulaires.json", vocabs)
    print(json.dumps(rapport, indent=1)[:6000])


# ---------------------------------------------------------------- étapes 4–6 : matrices et entraînements

def matrices(sortie, session):
    """Env principal : entrées actuelles (gbm_sacs, spécialiste livré) et recette, cibles, y_init, par validation."""
    from baselines import gbm_sacs
    from dio import donnees
    from dio.ensemble import GBM_DEFAUT, SPECIALISTES
    schema = donnees.charger_schema()
    regle = {**GBM_DEFAUT, **SPECIALISTES[rb.CIBLE]["gbm_specialiste"]}
    options = {k: regle[k] for k in ("jeu", "ajouts", "familles_variante", "retirer", "plan", "contexte")}
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    resume = {"versions": versions(), "options_cur": options, "config_cur": regle["config"]}
    for v, (_, test_dir) in list(DIRECTIONS.items()) + [("chrono_bord", (None, "validation_chrono"))]:
        debut = time.time()
        r = construire_rec("chrono", bord=True) if v == "chrono_bord" else construire_rec(v)
        m = {"x_rec": r["x"].to_numpy(np.float32), "xt_rec": r["xt"].to_numpy(np.float32), "y_rec": r["y"],
             "g": r["g"], "gt": r["gt"], "a_predire": r["a_predire"]}
        info = {"colonnes_rec": list(r["x"].columns), "circuits": r["circuits"], "vocab": r["vocab"]}
        if v != "chrono_bord":
            heures, table, y, meta = gbm_sacs.preparer(ANNEES_GBM[v], list(range(len(schema["target_columns"]))),
                                                       schema=schema, **options)
            test = donnees.lire_dossier_test(RACINE / test_dir / "test")
            m.update(x_cur=table.to_numpy(np.float64),
                     xt_cur=gbm_sacs.construire_test(test, meta)[list(table.columns)].to_numpy(np.float64),
                     y_cur=y, y_init=test["y_init"][:, schema["target_columns"].index(rb.CIBLE)],
                     n_init=np.array(test["n_init"]))
            info["colonnes_cur"] = list(table.columns)
            info["circuits_cur"] = schema["target_columns"]
            info["controles"] = {"heures_train_identiques": bool(np.array_equal(heures, r["g"])),
                                 "heures_test_identiques": bool(np.array_equal(test["heures"], r["gt"])),
                                 "nan_y_identiques": bool(np.array_equal(
                                     np.isnan(y), np.isnan(r["y"][:, [r["circuits"].index(c)
                                                                      for c in schema["target_columns"]]])))}
        np.savez(sortie / f"{v}.npz", **m)
        (sortie / f"{v}.json").write_text(json.dumps(info))
        info["empreintes"] = {k: empreinte(a) for k, a in m.items()}
        info["shapes"] = {k: list(np.shape(a)) for k, a in m.items()}
        info["duree_s"] = round(time.time() - debut, 1)
        resume[v] = {k: x for k, x in info.items() if k not in ("vocab", "colonnes_rec", "colonnes_cur",
                                                                "circuits_cur")}
        print(v, json.dumps(resume[v].get("controles"), default=str), info["duree_s"], "s", flush=True)
    ecrire(RACINE / "runs/b2_matrices" / f"resume_{session}.json", resume)


def relancer_dans_env(argv):
    """Créer l'env épinglé (Python 3.11) au besoin et y relancer la commande ; rend False si impossible."""
    python = ENV / "bin" / "python"
    try:
        if not python.exists():
            uv = shutil.which("uv")
            if not uv:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
                uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
            subprocess.run([uv, "venv", str(ENV), "--python", "3.11", "-q"], check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(python), *PAQUETS], check=True)
    except (subprocess.CalledProcessError, OSError) as erreur:
        print(f"Environnement épinglé impossible ({erreur}) : repli sur l'environnement principal", flush=True)
        return False
    argv = [a for a in argv if a != "--env-recette"] + ["--dans-env-recette"]
    if subprocess.run([str(python), "-I", str(Path(__file__).resolve()), *argv], cwd=RACINE).returncode:
        sys.exit("échec de la commande dans l'environnement épinglé")
    return True


def entrainer(dossier, cellules, prefixe, dans_env, bord, n_jobs):
    from baselines.boosting import recaler
    dossier = Path(dossier)
    env = "epingle" if dans_env else "principal"
    for v in list(DIRECTIONS) + (["chrono_bord"] if bord else []):
        m = dict(np.load(dossier / f"{v}.npz"))
        info = json.loads((dossier / f"{v}.json").read_text())
        recons = {}
        if dans_env:  # entrées recette reconstruites dans l'env de référence : elles servent à l'entraînement
            r = construire_rec("chrono", bord=True) if v == "chrono_bord" else construire_rec(v)
            nouv = {"x_rec": r["x"].to_numpy(np.float32), "xt_rec": r["xt"].to_numpy(np.float32), "y_rec": r["y"]}
            recons = {k: {"identique_env_principal": empreinte(a) == empreinte(m[k]), "sha256": empreinte(a)}
                      for k, a in nouv.items()}
            m.update(nouv)
        for cellule in (["rec_recette_coll"] if v == "chrono_bord" else cellules):
            entrees, reglage, msq = cellule.split("_")
            x, xt = m[f"x_{entrees}"], m[f"xt_{entrees}"]
            y33 = m[f"y_{entrees}"]
            circuits = info["circuits"] if entrees == "rec" else info["circuits_cur"]
            y = y33[:, circuits.index(rb.CIBLE)]
            garder = rb.masque(y33, circuits, msq == "coll")
            b, t = rb.bornes(len(x)), np.arange(len(x))
            apprises = [int((((t < b[i]) | (t >= b[i + 1])) & garder).sum()) for i in range(5)]
            debut = time.time()
            modeles = rb.ajuster(x, y, garder, rb.REGLAGES[reglage], n_jobs)
            duree = time.time() - debut
            brut = rb.moyenne(modeles, xt)
            sorties = {"pred.csv": np.clip(brut, 0, None)}
            if "y_init" in m:
                sorties["pred_recale.csv"] = np.clip(recaler(brut[:, None], m["y_init"][:, None], int(m["n_init"]),
                                                             1.0, 1440)[:, 0], 0, None)
            nom = f"{prefixe}_bord_{cellule}" if v == "chrono_bord" else f"{prefixe}_{cellule}"
            d = RACINE / "runs" / nom / ("chrono" if v == "chrono_bord" else v)
            d.mkdir(parents=True, exist_ok=True)
            for fichier, p in sorties.items():
                t = pd.DataFrame({rb.CIBLE: p}, index=pd.Index(m["gt"], name="ut_ms")).reindex(m["a_predire"])
                if t.isna().any().any() or not t.index.is_unique or not t.index.is_monotonic_increasing:
                    raise ValueError(f"{nom}/{v} : prédictions manquantes, non finies ou heures non ordonnées")
                t.to_csv(d / fichier)
            pd.DataFrame({rb.CIBLE: rb.hors_bloc(modeles, x), "vrai": y}, index=pd.Index(m["g"], name="ut_ms")).to_csv(
                d / "oob.csv")
            ecrire(d / "info.json", {
                "cellule": cellule, "validation": v, "environnement": env, "versions": versions(),
                "params": modeles[0].get_params(), "iterations": [int(mm.n_iter_) for mm in modeles],
                "arret_anticipe_actif": [bool(mm.do_early_stopping_) for mm in modeles],
                "lignes": len(x), "lignes_apprises_par_sac": apprises,
                "colonnes": x.shape[1], "masque_lignes": int(garder.sum()),
                "masque_collectif_egal_propre": bool(np.array_equal(rb.masque(y33, circuits),
                                                                    rb.masque(y33, circuits, False))), "duree_s": round(duree, 1),
                "empreintes": {"x": empreinte(x), "xt": empreinte(xt), "y": empreinte(y), "masque": empreinte(garder)},
                "entrees_reconstruites": recons})
            print(f"{nom} {v} : {duree:.0f} s, arbres {[int(mm.n_iter_) for mm in modeles]}", flush=True)


# ---------------------------------------------------------------- étape 6–7 : évaluation

CELLULES = [f"{e}_{r}_coll" for e in ("cur", "rec") for r in ("d3l", "recette")]
SEUIL_MOIS, HEURES_MOIS = 0.01, 300


def evaluer(prefixe="b3", installable="b3k_rec_recette_coll"):
    """Scores (livré, S exact, cellules seules, remplacement, mélange transféré), règle fixée avant mesure,
    effets du plan factoriel. Calcul local, aucun entraînement."""
    from outils import bilan_2551 as bl
    bl.SPECIALISTE = "b0_gel/specialiste_exact"
    toutes = bl.cas(["directe", "inverse", "chrono"])
    vals = ["directe", "inverse", "chrono"]

    def runs(nom, fichier="pred.csv"):
        return {v: bl.lire(RACINE / "runs" / nom / v / fichier, v) for v in vals}

    seul = {}
    for c in CELLULES + ["rec_recette_propre"]:
        for fichier, suffixe in (("pred.csv", ""), ("pred_recale.csv", "+recalage")):
            if (RACINE / "runs" / f"{prefixe}_{c}" / "directe" / fichier).exists():
                p = runs(f"{prefixe}_{c}", fichier)
                seul[c + suffixe] = {v: bl.mesures(p[v], v, False) for v in vals}
    S = {v: bl.specialiste(v) for v in vals}
    sortie = {"livre": {f"{v}/{g}": bl.mesures_cas(bl.base(v, g), v) for v, g in toutes},
              "specialiste_exact": {v: bl.mesures(S[v], v, False) for v in vals}, "cellules_seules": seul}
    n2 = {v: bl.lire(RACINE / "runs/n2_spec" / v / "pred.csv", v) for v in vals}
    sortie["controle_S_exact_vs_n2_spec"] = {v: float(np.nanmax(np.abs(S[v] - n2[v]))) for v in vals}
    ctrl = RACINE / "runs" / f"{prefixe}_cur_d3l_coll"
    if (ctrl / "directe" / "pred_recale.csv").exists():  # spécialiste livré réentraîné par ce code
        p = runs(f"{prefixe}_cur_d3l_coll", "pred_recale.csv")
        sortie["controle_cur_d3l_recale_vs_S_exact"] = {v: float(np.nanmax(np.abs(p[v] - S[v]))) for v in vals}
    objets = {"strict_epingle": f"{prefixe}_rec_recette_coll", "strict_principal": installable}
    for nom, run in objets.items():
        if not (RACINE / "runs" / run / "chrono" / "pred.csv").exists():
            continue
        rempl = {f"{v}/{g}": bl.mesures_cas(bl.remplacement(run, v, g), v) for v, g in toutes}
        preds, poids = bl.transferer({nom: run}, nom)
        regle = bl.regle(sortie["livre"], rempl)
        mois = []
        for cas in rempl:
            avant, apres = sortie["livre"][cas]["mois"], rempl[cas]["mois"]
            mois += [(cas, m, apres[m]["rmse"] - avant[m]["rmse"]) for m in avant
                     if avant[m]["n"] >= HEURES_MOIS and apres[m]["rmse"] - avant[m]["rmse"] > SEUIL_MOIS]
        if mois:
            regle["accepte"] = False
            regle["motifs"].append(f"mois en hausse > {SEUIL_MOIS} : {len(mois)}")
        regle["mois_en_hausse"] = mois
        sortie[nom] = {"seul": {v: bl.mesures(bl.candidat(run, v), v, False) for v in vals}, "remplacement": rempl,
                       "melange_transfere": {f"{v}/{g}": bl.mesures(preds[v, g], v, False) for v, g in toutes},
                       "poids_transferes": {c: poids[c]["poids"] for c in poids}, "regle_remplacement": regle,
                       "heures_livre_nul": {f"{v}/{g}": int((bl.base(v, g)[np.isfinite(bl.verite(v).to_numpy())] == 0).sum())
                                            for v, g in toutes}}
    if all(k in sortie for k in objets):
        a, b = runs(objets["strict_epingle"]), runs(objets["strict_principal"])
        sortie["effet_environnement"] = {v: {"ecart_max": float(np.nanmax(np.abs(a[v] - b[v]))),
                                             "rmse_epingle": seul.get("rec_recette_coll", {}).get(v, {}).get("rmse"),
                                             "rmse_principal": sortie["strict_principal"]["seul"][v]["rmse"]}
                                         for v in vals}
    bord = RACINE / "runs" / f"{prefixe}_bord_rec_recette_coll" / "chrono" / "pred.csv"
    if bord.exists():
        sortie["variante_bord_chrono"] = bl.mesures(bl.lire(bord, "chrono"), "chrono", False)
    # Effets du plan factoriel 2 × 2 × 2 (entrées, réglages, recalage) sur la RMSE du spécialiste seul.
    cles = [(e, r, k) for e in ("cur", "rec") for r in ("d3l", "recette") for k in ("", "+recalage")]
    if all(f"{e}_{r}_coll{k}" in seul for e, r, k in cles):
        effets = {}
        for v in vals:
            R = {c: seul[f"{c[0]}_{c[1]}_coll{c[2]}"][v]["rmse"] for c in cles}
            moy = lambda f: float(np.mean([R[c] for c in cles if f(c)]))  # noqa: E731
            effets[v] = {"entrees (rec - cur)": moy(lambda c: c[0] == "rec") - moy(lambda c: c[0] == "cur"),
                         "reglages (recette - d3l)": moy(lambda c: c[1] == "recette") - moy(lambda c: c[1] == "d3l"),
                         "recalage (avec - sans)": moy(lambda c: c[2]) - moy(lambda c: not c[2]),
                         "interaction entrees x reglages": (R["rec", "recette", ""] - R["cur", "recette", ""])
                         - (R["rec", "d3l", ""] - R["cur", "d3l", ""]),
                         "chemin S -> B": {"cur_d3l+recalage": R["cur", "d3l", "+recalage"],
                                           "cur_d3l": R["cur", "d3l", ""], "cur_recette": R["cur", "recette", ""],
                                           "rec_recette (B)": R["rec", "recette", ""]}}
            if "rec_recette_propre" in seul:
                effets[v]["masque (propre - collectif)"] = seul["rec_recette_propre"][v]["rmse"] - R["rec", "recette", ""]
        sortie["effets"] = effets
    oob = {}
    for c in CELLULES + ["rec_recette_propre"]:
        for v in vals:
            f = RACINE / "runs" / f"{prefixe}_{c}" / v / "oob.csv"
            if f.exists():
                t = pd.read_csv(f).dropna()
                oob.setdefault(c, {})[v] = float(np.sqrt(np.mean((t[rb.CIBLE] - t["vrai"]) ** 2)))
    sortie["oob_interne"] = oob
    ecrire(RACINE / "runs/b4_bilan/bilan.json", sortie)
    resume = {"livre": {c: round(r["rmse"], 4) for c, r in sortie["livre"].items()},
              "S": {v: round(r["rmse"], 4) for v, r in sortie["specialiste_exact"].items()},
              "seules": {c: {v: round(r[v]["rmse"], 4) for v in vals} for c, r in seul.items()}}
    for nom in objets:
        if nom in sortie:
            resume[nom] = {"remplacement": {c: round(r["rmse"], 4) for c, r in sortie[nom]["remplacement"].items()},
                           "regle": sortie[nom]["regle_remplacement"]}
    print(json.dumps({**resume, **{k: sortie[k] for k in ("effets", "effet_environnement", "oob_interne",
                                                         "controle_S_exact_vs_n2_spec") if k in sortie}},
                     indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["geler", "controle", "bords", "matrices", "entrainer", "evaluer"])
    ap.add_argument("--sortie", default="/tmp/b_mat")
    ap.add_argument("--matrices", default="/tmp/b_mat")
    ap.add_argument("--cellules", nargs="+", default=["rec_recette_coll"])
    ap.add_argument("--prefixe", default="b3")
    ap.add_argument("--bord", action="store_true", help="variante de bord de la chrono (recette stricte)")
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--session", default="local", help="nom du résumé des matrices (runs/b2_matrices/)")
    ap.add_argument("--env-recette", action="store_true")
    ap.add_argument("--dans-env-recette", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.env_recette and not a.dans_env_recette:
        if relancer_dans_env(sys.argv[1:]):
            return
        a.prefixe += "_repli"
    {"geler": geler, "controle": controle, "bords": bords}.get(a.action, lambda: None)()
    if a.action == "matrices":
        matrices(a.sortie, a.session)
    elif a.action == "evaluer":
        evaluer(a.prefixe)
    elif a.action == "entrainer":
        entrainer(a.matrices, a.cellules, a.prefixe, a.dans_env_recette, a.bord, a.n_jobs)


if __name__ == "__main__":
    main()
