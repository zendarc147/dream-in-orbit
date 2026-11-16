"""Second GRU « à boucle » dans la part réseau de T3 (docs/JOURNAL_GRU_BOUCLE.md).

    uv run python outils/gru_boucle.py kaggle K0       # arguments de lancer_kaggle.py (séparés par \\0) : K0 … K3
    uv run python outils/gru_boucle.py fusionner       # g04 + g59 -> g10 (GB, GB-mm)
    uv run python outils/gru_boucle.py repro           # S0 = T3 : predict.py -> evaluate.py (pred_repro.csv)
    uv run python outils/gru_boucle.py assembler       # S1, S2 (outils/candidat.py, liens durs)
    uv run python outils/gru_boucle.py gru             # GRU seuls : G0, G0 sans recalage, GB, GB-r, GB-mm
    uv run python outils/gru_boucle.py predire         # S1, S2, S2 à 0.6, de bout en bout
    uv run python outils/gru_boucle.py controles       # part boosting, NPWD2532, spécialistes = T3 au bit près
    uv run python outils/gru_boucle.py bilan
    uv run python outils/gru_boucle.py decider
    uv run python outils/gru_boucle.py finaux          # poids annee1 + annee2 -> poids_candidats/gru_boucle/

Liste fermée (fixée avant mesure) : GRU G0 (installé), GB, GB-r (prédiction seule), GB-mm ; structures S0 (= T3),
S1 (GB seul dans la part réseau), S2 (0.5 G0 + 0.5 GB) ; variante de rapport S2 avec un poids boosting de 0.6.
Runs : runs/gb_<v>_<GB|GBmm>_<g04|g59|g10> (Kaggle) ; structures runs/gb_<S1|S2>_<v>/<v>/ ; S0 = runs/st_T3_<v>_g10.
"""
import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.structure import (CHAUFFAGES, COUPURES, VALIDATIONS, candidat, composants, lire, mesures,  # noqa: E402
                              predire_un, rmse, verite)
from outils.valider import TOUTES  # noqa: E402

RUNS = RACINE / "runs"
SORTIE = RUNS / "gb_bilan"
GB = {"variables": "S", "reinjection": True, "premier_segment_verite": True, "normalisation": "standard"}
GBMM = {**GB, "normalisation": "minmax"}
G59 = {"graines": [5, 6, 7, 8, 9]}
ENTRAINEMENTS = ["directe", "inverse", "c62", "c75", "c87", "final"]
SESSIONS = {"K0": ["c62"], "K1": ["directe", "inverse"], "K2": ["c75", "c87"], "K3": ["final"]}
GRUS = {"G0": None, "G0sr": None, "GB": "GB", "GBr": "GB", "GBmm": "GBmm"}
RECALAGE = {"G0": 1.0, "G0sr": 0.0, "GB": 0.0, "GBr": 1.0, "GBmm": 0.0}
STRUCTURES = ["S0", "S1", "S2", "S2b06"]


def annees(w):
    if w in ("directe", "inverse"):
        return [{"directe": "annee1", "inverse": "annee2"}[w]]
    if w == "final":
        return ["annee1", "annee2"]
    return ["annee1", f"validation_glissant/{w}/entrainement"]


def kaggle(session):
    out = []
    for w in SESSIONS[session]:
        for nom, opts in (("GB", GB), ("GBmm", GBMM)):
            for g in ("g04", "g59"):
                o = {**opts, **(G59 if g == "g59" else {})}
                out += ["train.py", "--modele", "gru_recette", "--annees", *annees(w), "--out", f"runs/gb_{w}_{nom}_{g}",
                        "--options", json.dumps(o), ";"]
    print("\0".join(out[:-1]), end="")


def fusionner():
    from outils.fusionner_recette import fusionner as f
    for w in ENTRAINEMENTS:
        for nom in ("GB", "GBmm"):
            a, b, c = (RUNS / f"gb_{w}_{nom}_{g}" for g in ("g04", "g59", "g10"))
            if (a / "recette.pt").is_file() and (b / "recette.pt").is_file() and not (c / "recette.pt").is_file():
                print(c.name, f(a, b, c), "graines", flush=True)


def run_gru(nom, v):
    """Dossier du paquet GRU à 10 graines d'une validation (c50 = modèles de la directe)."""
    if nom in ("G0", "G0sr"):
        return RUNS / composants(v)["gruS"]["g10"]
    w = "directe" if v == "c50" else v
    return RUNS / f"gb_{w}_{GRUS[nom]}_g10"


def dossier_structure(s, v):
    if s == "S0":
        return RUNS / f"st_T3_{v}_g10" / v
    return RUNS / f"gb_{'S2' if s == 'S2b06' else s}_{v}" / v


FICHIER = {"S0": "pred_repro.csv", "S1": "pred.csv", "S2": "pred.csv", "S2b06": "pred_b06.csv"}


def repro():
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(predire_un, [(dossier_structure("S0", v), v, "pred_repro.csv", {}) for v in VALIDATIONS]))
    for v in VALIDATIONS:
        d = dossier_structure("S0", v)
        s = json.loads((d / "scores" / "pred_repro.csv.json").read_text())["rmse"]
        r = json.loads((d / "scores" / "pred.csv.json").read_text())["rmse"]
        print(v, f"repro {s:.6f} journal {r:.6f}", "fichier identique" if filecmp.cmp(d / "pred.csv", d / "pred_repro.csv",
                                                                                         shallow=False) else "FICHIER DIFFÉRENT")


def assembler():
    for v in VALIDATIONS:
        base = ["--base", f"st_T3_{v}_g10", "--directions", v, "--recette2", run_gru("GB", v).name]
        candidat("--nom", f"gb_S1_{v}", *base, "--part2", 1)
        candidat("--nom", f"gb_S2_{v}", *base, "--part2", 0.5)
        print(v, "assemblé", flush=True)


def _gru_un(travail):
    nom, v = travail
    import torch
    torch.set_num_threads(2)
    from dio import recette
    out = SORTIE / "gru" / v / f"{nom}.csv"
    if out.is_file():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    debut = time.time()
    recette.predire(TOUTES[v][1] / "test", out, run_gru(nom, v), recalage=RECALAGE[nom])
    (out.parent / f"{nom}.temps").write_text(f"{time.time() - debut:.1f}\n")
    print(v, nom, f"{time.time() - debut:.0f} s", flush=True)


def gru():
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(_gru_un, [(n, v) for v in VALIDATIONS for n in GRUS]))


def predire():
    t = [(dossier_structure(s, v), v, FICHIER[s], {"poids_gbm": 0.6} if s == "S2b06" else {})
         for v in VALIDATIONS for s in ("S1", "S2", "S2b06")]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(predire_un, t))


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def controles():
    """Part boosting et NPWD2532 de S1/S2 = T3 au bit près ; fichiers des spécialistes identiques."""
    ok = True
    for v in VALIDATIONS:
        t3 = dossier_structure("S0", v)
        for s in ("S1", "S2"):
            d = dossier_structure(s, v)
            g = filecmp.cmp(Path(str(t3 / "pred_repro.csv") + ".parts") / "gbm.csv",
                            Path(str(d / "pred.csv") + ".parts") / "gbm.csv", shallow=False)
            a, b = lire(t3 / "pred_repro.csv", v), lire(d / "pred.csv", v)
            c2532 = np.array_equal(a["NPWD2532"].to_numpy(), b["NPWD2532"].to_numpy(), equal_nan=True)
            spec = all(sha(f) == sha(d / f.relative_to(t3)) for f in sorted((t3 / "specialiste").rglob("*")) if f.is_file())
            boost = all(sha(f) == sha(d / f.relative_to(t3)) for f in sorted((t3 / "gbm").rglob("*")) if f.is_file())
            r1 = filecmp.cmp(t3 / "recette" / "recette.pt", d / "recette" / "recette.pt", shallow=False) \
                if (d / "recette").is_dir() else None
            print(v, s, "part boosting", g, "| NPWD2532", c2532, "| specialiste/", spec, "| gbm/", boost, "| recette/ = G0", r1)
            ok &= g and c2532 and spec and boost
    print("CONTRÔLES OK" if ok else "CONTRÔLES EN ÉCHEC")


# ---------------------------------------------------------------- bilan

def correlations(a, b):
    ea, eb = a - verite(a.attrs["v"]), b - verite(b.attrs["v"])
    k = np.isfinite(ea.to_numpy()) & np.isfinite(eb.to_numpy())
    return {"globale": float(np.corrcoef(ea.to_numpy()[k], eb.to_numpy()[k])[0, 1]),
            **{c: float(ea[c].corr(eb[c])) for c in CHAUFFAGES}}


def temps_entrainement():
    """« Entraînement terminé en N s -> runs/gb_… » des journaux Kaggle rapatriés."""
    t = {}
    for f in sorted((RUNS / "_kaggle").glob("*/*.log")):
        try:
            texte = "".join(l.get("data", "") for l in json.loads(f.read_text() or "[]"))
        except (json.JSONDecodeError, AttributeError):
            texte = f.read_text()
        for n, r in re.findall(r"Entraînement terminé en (\d+) s -> runs/(gb_\S+)", texte):
            t[r] = int(n)
    return t


def bilan():
    r = {"gru": {}, "structures": {}, "correlation_G0_GB": {}, "temps_prediction_gru": {}}
    for v in VALIDATIONS:
        preds = {}
        for n in GRUS:
            f = SORTIE / "gru" / v / f"{n}.csv"
            if f.is_file():
                preds[n] = lire(f, v)
                preds[n].attrs["v"] = v
                r["gru"][f"{n}_{v}"] = mesures(preds[n], v)
                r["temps_prediction_gru"][f"{n}_{v}"] = float((f.parent / f"{n}.temps").read_text())
        if "G0" in preds and "GB" in preds:
            r["correlation_G0_GB"][v] = correlations(preds["G0"], preds["GB"])
            r["correlation_G0_GB"][v + "_GB_GBmm"] = correlations(preds["GB"], preds["GBmm"]) if "GBmm" in preds else None
        for s in STRUCTURES:
            out = dossier_structure(s, v) / FICHIER[s]
            if not out.is_file():
                continue
            p = Path(str(out) + ".parts")
            parts = {k: lire(p / f"{k}.csv", v) for k in ("gru", "gbm", "melange")}
            x = mesures(lire(out, v), v, parts)
            x["temps_prediction"] = json.loads((p / "temps.json").read_text())
            x["predict_s"] = json.loads((out.parent / "scores" / f"{FICHIER[s]}.json").read_text())["predict_s"]
            r["structures"][f"{s}_{v}"] = x
            print(s, v, f"{x['rmse']:.6f}", flush=True)
    r["temps_entrainement"] = temps_entrainement()
    import torch
    r["entrees"] = {}
    for n in ("G0", "GB", "GBmm"):
        p = torch.load(run_gru(n, "directe") / "recette.pt", weights_only=True)
        r["entrees"][n] = {"variables": len(p["variables"]), "reinjectees": len(p["circuits"]) if p["config"]["reinjection"] else 0,
                           "graines": len(p["etats"])}
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "bilan.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    tableaux(r)


def f6(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.6f}"


def tableaux(r):
    """Tableaux markdown du journal (runs/gb_bilan/tableaux.md)."""
    L = []
    tete = "| | " + " | ".join(VALIDATIONS) + " | Moy. dir/inv | Moy. coupures |\n|---|" + "---:|" * (len(VALIDATIONS) + 2)

    def ligne(nom, d):
        xs = [d.get(v) for v in VALIDATIONS]
        t = np.mean(xs[:2]) if None not in xs[:2] else None
        c = np.mean(xs[2:]) if None not in xs[2:] else None
        return f"| {nom} | " + " | ".join(f6(x) for x in xs) + f" | {f6(t)} | {f6(c)} |"

    L += ["### RMSE globale, GRU seuls (part réseau seule)", "", tete]
    for n in GRUS:
        L.append(ligne(n, {v: r["gru"].get(f"{n}_{v}", {}).get("rmse") for v in VALIDATIONS}))
    L += ["", "### RMSE globale, structures (modèle complet)", "", tete]
    for s in STRUCTURES:
        L.append(ligne(s, {v: r["structures"].get(f"{s}_{v}", {}).get("rmse") for v in VALIDATIONS}))
    for obj, titre in (("gru", "part réseau"), ("gbm", "part boosting"), ("melange", "mélange avant spécialistes")):
        L += ["", f"### Structures : {titre}", "", tete]
        for s in STRUCTURES:
            L.append(ligne(s, {v: r["structures"].get(f"{s}_{v}", {}).get("objets", {}).get(obj, {}).get("rmse")
                               for v in VALIDATIONS}))
    L += ["", "### Corrélation des erreurs G0 / GB (GRU seuls)", "",
          "| Validation | globale | NPWD2551 | NPWD2451 | NPWD2851 |", "|---|---:|---:|---:|---:|"]
    for v in VALIDATIONS:
        c = r["correlation_G0_GB"].get(v)
        if c:
            L.append(f"| {v} | {c['globale']:.3f} | " + " | ".join(f"{c[k]:.3f}" for k in CHAUFFAGES) + " |")
    L += ["", "### Circuits suivis : RMSE / MAE / biais, moyenne des 6 validations (et c87)", "",
          "| Objet | NPWD2551 | NPWD2451 | NPWD2851 | NPWD2532 | NPWD2531 |", "|---|---|---|---|---|---|"]
    from outils.structure import SUIVIS
    for groupe, cle in (("gru", GRUS), ("structures", STRUCTURES)):
        for n in cle:
            for sel, etiq in ((VALIDATIONS, "moy. 6"), (["c87"], "c87")):
                xs = [r[groupe].get(f"{n}_{v}") for v in sel]
                if None in xs:
                    continue
                cel = [" / ".join(f"{np.mean([x['circuits'][c][k] for x in xs]):+.4f}" if k == "biais" else
                                  f"{np.mean([x['circuits'][c][k] for x in xs]):.4f}" for k in ("rmse", "mae", "biais"))
                       for c in SUIVIS]
                L.append(f"| {n} ({etiq}) | " + " | ".join(cel) + " |")
    L += ["", "### RMSE par quart d'horizon (q1 / q2 / q3 / q4)", "", "| Objet | " + " | ".join(VALIDATIONS) + " |",
          "|---|" + "---|" * len(VALIDATIONS)]
    for groupe, cle in (("gru", GRUS), ("structures", STRUCTURES)):
        for n in cle:
            L.append(f"| {n} | " + " | ".join(" / ".join(f"{q:.4f}" for q in r[groupe][f"{n}_{v}"]["quarts"])
                                             if f"{n}_{v}" in r[groupe] else "—" for v in VALIDATIONS) + " |")
    L += ["", "### Biais hors saison d'éclipses / régime haut hors saison (moyenne des 4 coupures ; directe ; inverse)", "",
          "| Objet | 2551 hors saison | 2551 haut hors saison | 2451 hors saison | 2451 haut hors saison |",
          "|---|---|---|---|---|"]
    for groupe, cle in (("gru", GRUS), ("structures", STRUCTURES)):
        for n in cle:
            cel = []
            for c in ("NPWD2551", "NPWD2451"):
                for k in ("hors_saison", "haut_hors_saison"):
                    parts_ = []
                    for sel in (COUPURES, ["directe"], ["inverse"]):
                        xs = [r[groupe].get(f"{n}_{v}") for v in sel]
                        parts_.append(f"{np.nanmean([x['circuits'][c][k]['biais'] for x in xs]):+.4f}" if None not in xs else "—")
                    cel.append(" ; ".join(parts_))
            L.append(f"| {n} | " + " | ".join(cel) + " |")
    L += ["", "### Diagnostics de réglage (RMSE globale ; Δ négatif = le réglage aide)", "",
          "| Validation | réinjection : GBmm − G0sr | normalisation : GB − GBmm | recalage : GB − GBr |", "|---|---:|---:|---:|"]
    g = r["gru"]
    for v in VALIDATIONS:
        try:
            L.append(f"| {v} | {g[f'GBmm_{v}']['rmse'] - g[f'G0sr_{v}']['rmse']:+.6f} | "
                     f"{g[f'GB_{v}']['rmse'] - g[f'GBmm_{v}']['rmse']:+.6f} | {g[f'GB_{v}']['rmse'] - g[f'GBr_{v}']['rmse']:+.6f} |")
        except KeyError:
            pass
    L += ["", "### Temps", "", "Entraînement (s, T4, 5 graines par entrée) : " +
          ", ".join(f"{k} {x}" for k, x in sorted(r["temps_entrainement"].items())), "",
          "Prédiction GRU seul (s, 10 graines, local, 4 en parallèle) : " +
          ", ".join(f"{k} {x:.0f}" for k, x in r["temps_prediction_gru"].items()), "",
          "Prédiction de bout en bout (s, local, 3 en parallèle) : " +
          ", ".join(f"{k} {x['predict_s']}" for k, x in r["structures"].items()), "",
          "Entrées : " + json.dumps(r["entrees"])]
    (SORTIE / "tableaux.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- décision

def decider():
    r = json.loads((SORTIE / "bilan.json").read_text())["structures"]
    s0 = {v: r[f"S0_{v}"] for v in VALIDATIONS}
    s2 = {v: r[f"S2_{v}"] for v in VALIDATIONS}
    t0, t2 = (np.mean([x[v]["rmse"] for v in ("directe", "inverse")]) for x in (s0, s2))
    derive = {v: (s2[v]["quarts"][3] > 1.5 * s2[v]["quarts"][0]) and not (s0[v]["quarts"][3] > 1.5 * s0[v]["quarts"][0])
              for v in VALIDATIONS}
    limites = {"sortie non finie ou négative": not all(s2[v]["finies"] for v in VALIDATIONS),
               "moyenne temporelle S2 > S0 + 0.0030": bool(t2 > t0 + 0.0030),
               "une coupure > S0 + 0.0060": any(s2[c]["rmse"] > s0[c]["rmse"] + 0.0060 for c in COUPURES),
               "dérive manifeste (q4 > 1.5 × q1, pas pour S0)": any(derive.values())}
    echec = any(limites.values())
    d = {"S0": {"moyenne_temporelle": t0, **{v: s0[v]["rmse"] for v in VALIDATIONS}},
         "S2": {"moyenne_temporelle": t2, **{v: s2[v]["rmse"] for v in VALIDATIONS}},
         "ecarts_S2_S0": {"moyenne_temporelle": t2 - t0, **{v: s2[v]["rmse"] - s0[v]["rmse"] for v in VALIDATIONS},
                          "moyenne_coupures": float(np.mean([s2[c]["rmse"] - s0[c]["rmse"] for c in COUPURES]))},
         "derive_par_validation": derive, "limites": limites,
         "installe": "S0 (échec technique de S2)" if echec else "S2"}
    for s in ("S1", "S2b06"):
        d[f"ecarts_{s}_S0"] = {v: r[f"{s}_{v}"]["rmse"] - s0[v]["rmse"] for v in VALIDATIONS}
    (SORTIE / "decision.json").write_text(json.dumps(d, indent=1, default=float) + "\n")
    print(json.dumps(d, indent=1, default=float))


# ---------------------------------------------------------------- poids finaux

FINAUX = RACINE / "poids_candidats" / "gru_boucle"
T3 = RACINE / "poids_candidats" / "structure" / "T3_g10"


def finaux():
    """Copies indépendantes (aucun lien dur) de T3_g10 + recette2/ (GB final, 10 graines)."""
    fusionner()
    gb = RUNS / "gb_final_GB_g10" / "recette.pt"
    specs = {"S1": (1.0, 0.5), "S2": (0.5, 0.5), "S2_b06": (0.5, 0.6)}
    for nom, (w2, pg) in specs.items():
        cible = FINAUX / nom
        if cible.exists():
            shutil.rmtree(cible)
        shutil.copytree(T3, cible)
        (cible / "recette2").mkdir()
        shutil.copy2(gb, cible / "recette2" / "recette.pt")
        if w2 == 1:
            shutil.rmtree(cible / "recette")
        reglages = {**json.loads((cible / "ensemble.json").read_text()), "poids_gbm": pg, "recette2": w2,
                    "recalage_recette2": 0.0}
        (cible / "ensemble.json").write_text(json.dumps(reglages, indent=2) + "\n")
        meta = json.loads((RACINE / "poids" / "modele.json").read_text())  # T3 installé (lancer avant installation)
        (cible / "modele.json").unlink()
        meta["options_recette2"] = {**GB, "graines": list(range(10))}
        meta["recalage_recette2"] = 0.0
        meta["note"] = (f"Candidat {nom} (docs/JOURNAL_GRU_BOUCLE.md) : part réseau = "
                        + ("GB seul" if w2 == 1 else "0.5 × GRU S borné (recette/, recalé) + 0.5 × GRU à boucle GB "
                           "(recette2/, réinjection, normalisation standard, sans recalage)")
                        + f" ; poids boosting {pg} ; part boosting et spécialistes de T3.")
        (cible / "modele.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        print(nom, "->", cible.relative_to(RACINE), flush=True)
    cible = FINAUX / "GBmm_final"
    cible.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS / "gb_final_GBmm_g10" / "recette.pt", cible / "recette.pt")
    liens = [f for f in FINAUX.rglob("*") if f.is_file() and os.stat(f).st_nlink > 1]
    print("fichiers à liens multiples :", liens or "aucun")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["kaggle", "fusionner", "repro", "assembler", "gru", "predire", "controles",
                                       "bilan", "decider", "finaux"])
    ap.add_argument("session", nargs="?")
    a = ap.parse_args()
    if a.action == "kaggle":
        kaggle(a.session)
    else:
        globals()[a.action]()


if __name__ == "__main__":
    main()
