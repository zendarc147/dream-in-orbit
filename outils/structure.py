"""Structures sobres de l'ensemble (docs/JOURNAL_STRUCTURE.md) : assembler, prédire, bilan, décision.

    python outils/structure.py assembler [--validations directe ...] [--candidats L T3 ...]
    python outils/structure.py predire [--travaux 4]
    python outils/structure.py bilan
    python outils/structure.py decider
    python outils/structure.py finaux    # poids annee1 + annee2 de chaque candidat -> poids_candidats/structure/

Candidats (liste fermée) : L (livré), C1, T1, T2, T3 ; variantes : GRU S à 10 graines (groupe g10), T3 sans recalage
(T3sr : dossier de T3, predict.py --options '{"recalage": 0.0}'). Validations : directe, inverse, c50, c62, c75, c87.
Dossiers : runs/st_<cand>_<validation>_<groupe>/<validation>/ (assemblés par outils/candidat.py, liens durs).
Composants : directe/inverse et c50 = modèles existants (c50 = modèles de la directe) ; c62/c75/c87 = runs/st_<c>_*
(Kaggle, runs/st_kaggle.py).
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.valider import TOUTES  # noqa: E402

H = 3_600_000
VALIDATIONS = ["directe", "inverse", "c50", "c62", "c75", "c87"]
COUPURES = ["c50", "c62", "c75", "c87"]
CANDIDATS = ["L", "C1", "T1", "T2", "T3"]
SUIVIS = ["NPWD2551", "NPWD2451", "NPWD2851", "NPWD2532", "NPWD2531"]
CHAUFFAGES = ["NPWD2551", "NPWD2451", "NPWD2851"]
ORDRE = ["T3sr_g10", "T3sr_g5", "T3_g10", "T3_g5", "T2_g10", "T2_g5", "T1_g10", "T1_g5", "C1_g5"]
SORTIE = RACINE / "runs" / "st_bilan"


def composants(v):
    """Runs des composants d'une validation (noms sous runs/), par groupe de graines."""
    if v in ("directe", "inverse", "c50"):
        d = "directe" if v == "c50" else v
        return {"L": {"g04": f"rc0_livre_g04/{d}", "g59": f"rc0_livre_g59/{d}"},
                "gruS": {"g04": f"rc3_gruA_norm/{d}", "g59": f"rc4_gruA_mm_g59/{d}", "g10": f"st_gruS10/{d}"},
                "S": f"rc1_gbm_S/{d}", "B": f"st_B_{d}"}
    return {"L": {"g04": f"st_{v}_L_g04", "g59": f"st_{v}_L_g59"},
            "gruS": {"g04": f"st_{v}_gruS_g04", "g59": f"st_{v}_gruS_g59", "g10": f"st_{v}_gruS10"},
            "S": f"st_{v}_gbmS", "B": f"st_{v}_B"}


def candidat(*args):
    subprocess.run([sys.executable, "outils/candidat.py", *map(str, args)], cwd=RACINE, check=True,
                   stdout=subprocess.DEVNULL)


def nom(cand, v, g):
    return f"st_{cand}_{v}_{g}"


def assembler(validations, candidats, controles=False):
    for v in validations:
        k = composants(v)
        if v not in ("directe", "inverse", "c50"):  # L g59 et GRU S 10 graines des coupures
            if not (RACINE / "runs" / k["L"]["g59"]).is_dir():
                candidat("--nom", k["L"]["g59"], "--base", k["L"]["g04"], "--directions", v,
                         "--mixte", f"st_{v}_mixte_g59", "--recette", f"st_{v}_g2_g59")
            if not (RACINE / "runs" / k["gruS"]["g10"] / "recette.pt").is_file():
                from outils.fusionner_recette import fusionner
                fusionner(RACINE / "runs" / k["gruS"]["g04"], RACINE / "runs" / k["gruS"]["g59"],
                          RACINE / "runs" / k["gruS"]["g10"])
        for g in ("g04", "g59", "g10"):
            base = ["--base", k["L"]["g04" if g == "g10" else g], "--directions", v]
            gru = ["--recette", k["gruS"][g]]
            specs = {"L": [], "C1": gru + ["--part", 0.5], "T1": gru + ["--part", 1],
                     "T2": gru + ["--part", 1, "--gbm2", k["S"], "--poids-gbm2", 0.5, "--b", k["B"], "--part-b", 0.5],
                     "T3": gru + ["--part", 1, "--gbm", k["S"], "--b", k["B"], "--part-b", 1]}
            if controles and v == "directe" and g == "g04":  # contrôles : T2 et T3 sans B (seule 2551 doit changer)
                specs["T2ctl"] = specs["T2"][:-4]
                specs["T3ctl"] = specs["T3"][:-4]
            for c, spec in specs.items():
                if c not in candidats and not c.endswith("ctl"):
                    continue
                if g == "g10" and c in ("L", "C1"):
                    continue
                candidat("--nom", nom(c, v, g), *base, *spec)
        print(v, "assemblé", flush=True)


def travaux():
    sortie = []
    for d in sorted((RACINE / "runs").glob("st_*_*_g*")):
        parts = d.name.split("_")
        if len(parts) != 4 or parts[1] not in CANDIDATS + ["T2ctl", "T3ctl"] or parts[2] not in VALIDATIONS:
            continue
        v = parts[2]
        sortie.append((d / v, v, "pred.csv", {}))
        if parts[1] == "T3":
            sortie.append((d / v, v, "pred_sr.csv", {"recalage": 0.0}))
    return sortie


def predire_un(travail):
    dossier, v, fichier, options = travail
    out = dossier / fichier
    if out.is_file() and (dossier / "scores" / f"{fichier}.json").is_file():
        return
    test = TOUTES[v][1]
    debut = time.time()
    r = subprocess.run([sys.executable, "predict.py", "--test_dir", test / "test", "--out", out, "--poids", dossier,
                        "--options", json.dumps({**options, "parts": True})], cwd=RACINE, capture_output=True, text=True)
    if r.returncode:
        print("ÉCHEC", out, r.stderr[-2000:], flush=True)
        return
    duree = time.time() - debut
    e = subprocess.run([sys.executable, "evaluate.py", "--pred", out, "--verite", test / "verite_horaire.csv",
                        "--heures", test / "test" / "heures_a_predire.csv"], cwd=RACINE, capture_output=True, text=True)
    (dossier / "scores").mkdir(exist_ok=True)
    (dossier / "scores" / f"{fichier}.json").write_text(json.dumps(
        {"rmse": float(e.stdout.split("RMSE = ")[1].split()[0]), "predict_s": round(duree)}) + "\n")
    print(out.relative_to(RACINE), e.stdout.strip(), f"{duree:.0f} s", flush=True)


# ---------------------------------------------------------------- bilan

_CACHE = {}


def verite(v):
    if ("y", v) not in _CACHE:
        _CACHE["y", v] = pd.read_csv(TOUTES[v][1] / "verite_horaire.csv", index_col="ut_ms")
    return _CACHE["y", v]


def lire(chemin, v):
    return pd.read_csv(chemin, index_col="ut_ms").reindex(verite(v).index)[verite(v).columns]


def ombre(v):
    """True aux heures d'un jour sans ombre de Mars (contexte du test seul)."""
    if ("o", v) not in _CACHE:
        from dio import donnees
        x = donnees.lire_dossier_test(TOUTES[v][1] / "test")["X"].reindex(verite(v).index)
        jour = x.index // (24 * H)
        _CACHE["o", v] = (x["evtf_MAR_UMBRA_minutes"].groupby(jour).transform("sum") == 0).to_numpy()
    return _CACHE["o", v]


def seuils_hauts():
    """Médiane des moyennes hebdomadaires vraies de chaque circuit sur annee1 + annee2 (rapport seulement)."""
    if "seuils" not in _CACHE:
        y = pd.concat([pd.read_csv(RACINE / "data" / "prepared" / a / "y.csv", index_col="ut_ms") for a in ("annee1", "annee2")])
        _CACHE["seuils"] = y.groupby(y.index // (168 * H)).mean().median()
    return _CACHE["seuils"]


def rmse(e):
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def episodes(e, index):
    """Épisodes de sous-estimation prolongée : ≥ 3 jours consécutifs d'erreur journalière moyenne < −0.1."""
    t = pd.Series(e, index=index).dropna()
    jours = t.groupby(t.index // (24 * H)).mean()
    sous = np.r_[(jours < -0.1).to_numpy(), False]
    liste, debut = [], None
    for i, b in enumerate(sous):
        if b and debut is None:
            debut = i
        elif not b and debut is not None:
            if i - debut >= 3:
                liste.append((debut, i))
            debut = None
    dans = np.isin((t.index // (24 * H)).to_numpy(), np.concatenate([jours.index[a:b] for a, b in liste]) if liste else [])
    return {"nombre": len(liste), "jours": int(sum(b - a for a, b in liste)),
            "part_mse": float(np.sum(t[dans] ** 2) / np.sum(t ** 2)) if len(t) else 0.0}


def sans_biais_hebdo(e, index):
    t = pd.DataFrame(e, index=index)
    return t - t.groupby(t.index // (168 * H)).transform("mean")


def mesures(p, v, parts=None):
    y = verite(v)
    e = p - y
    k = np.isfinite(y.to_numpy())
    plat = e.to_numpy()[k]
    n = len(e)
    s = {"rmse": rmse(plat), "finies": bool(np.isfinite(p.to_numpy()).all() and (p.to_numpy() >= 0).all()),
         "quarts": [rmse(e.iloc[i * n // 4:(i + 1) * n // 4].to_numpy()) for i in range(4)],
         "rmse_sans_biais_hebdo": rmse(sans_biais_hebdo(e, e.index).to_numpy())}
    mois = pd.to_datetime(e.index, unit="ms").strftime("%Y-%m")
    par_mois = {m: rmse(g.to_numpy()) for m, g in e.groupby(mois) if g.notna().to_numpy().sum() >= 300 * 33}
    s["pire_mois"] = max(par_mois.items(), key=lambda x: x[1]) if par_mois else None
    carre = (e ** 2).sum(axis=1, min_count=1)
    nb = e.notna().sum(axis=1)
    j = pd.DataFrame({"c": carre, "n": nb}).groupby(e.index // (24 * H)).sum()
    f = j.rolling(28).sum().dropna()
    f = f[f.n > 0]
    if len(f):
        w = (f.c / f.n).idxmax()
        s["pire_28j"] = [str(pd.to_datetime((w - 27) * 24 * H, unit="ms").date()), float(np.sqrt(f.c[w] / f.n[w]))]
    hors = ombre(v)
    seuils = seuils_hauts()
    s["circuits"] = {}
    for c in SUIVIS:
        ec = e[c].to_numpy()
        m = np.isfinite(ec)
        d = {"rmse": rmse(ec), "mae": float(np.mean(np.abs(ec[m]))), "biais": float(ec[m].mean())}
        if c in CHAUFFAGES:
            semaine = y[c].groupby(y.index // (168 * H)).transform("mean").to_numpy() > seuils[c]
            for etiq, masque in (("hors_saison", hors), ("haut_hors_saison", hors & semaine)):
                mm = masque & m
                d[etiq] = {"heures": int(mm.sum()), "rmse": rmse(ec[mm]),
                           "biais": float(ec[mm].mean()) if mm.any() else float("nan")}
            d["episodes"] = episodes(ec, e.index)
            d["rmse_sans_biais_hebdo"] = rmse(sans_biais_hebdo(e[[c]], e.index).to_numpy())
        s["circuits"][c] = d
    if parts:
        s["objets"] = {}
        for nom_, q in parts.items():
            eq = q - y
            s["objets"][nom_] = {"rmse": rmse(eq.to_numpy()), **{c: rmse(eq[c].to_numpy()) for c in CHAUFFAGES}}
        eg, eb = (parts["gru"] - y), (parts["gbm"] - y)
        km = np.isfinite(eg.to_numpy()) & np.isfinite(eb.to_numpy())
        s["correlation_gru_gbm"] = {"globale": float(np.corrcoef(eg.to_numpy()[km], eb.to_numpy()[km])[0, 1]),
                                    **{c: float(eg[c].corr(eb[c])) for c in CHAUFFAGES}}
    return s


def bilan():
    resultats = {}
    for dossier, v, fichier, _ in travaux():
        out = dossier / fichier
        if not out.is_file():
            continue
        cle = dossier.parent.name.removeprefix("st_") + ("_sr" if fichier == "pred_sr.csv" else "")
        pd_parts = Path(str(out) + ".parts")
        parts = {k: lire(pd_parts / f"{k}.csv", v) for k in ("gru", "gbm", "melange")} if pd_parts.is_dir() else None
        s = mesures(lire(out, v), v, parts)
        if pd_parts.is_dir():
            s["temps_prediction"] = json.loads((pd_parts / "temps.json").read_text())
        resultats[cle] = s
        print(cle, f"{s['rmse']:.6f}", flush=True)
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "bilan.json").write_text(json.dumps(resultats, indent=1) + "\n")


# ---------------------------------------------------------------- sobriété des composants

def _arbres(estimateurs):
    """(colonnes lues par au moins une coupure, nombre d'arbres) d'une liste d'HGB."""
    lues, n = set(), 0
    for m in estimateurs:
        for iteration in getattr(m, "_predictors", []):
            for arbre in iteration:
                noeuds = arbre.nodes
                lues |= set(noeuds["feature_idx"][~noeuds["is_leaf"].astype(bool)].tolist())
                n += 1
    return lues, n


def sobriete_composant(dossier):
    import lzma
    import pickle
    import torch
    d = Path(dossier)
    if (d / "recette.pt").is_file():
        p = torch.load(d / "recette.pt", weights_only=True)
        return {"type": "GRU recette", "entrees": len(p["variables"]), "graines": len(p["etats"])}
    if (d / "membres.json").is_file():
        m = [torch.load(x / "monde.pt", weights_only=True) for x in sorted(d.glob("membre[0-9]*"))]
        return {"type": "GRU mixte", "entrees_par_membre": [len(x["variables"]) for x in m],
                "graines": sum(len(x["etats"]) for x in m)}
    if (d / "modele_b.pkl.xz").is_file():
        with lzma.open(d / "modele_b.pkl.xz", "rb") as f:
            p = pickle.load(f)
        lues, n = _arbres(p["modeles"])
        return {"type": "recette B", "colonnes": len(p["colonnes"]), "colonnes_lues": len(lues), "arbres": n}
    chemin = d / "gbm_sacs.pkl"
    with (open(chemin, "rb") if chemin.exists() else lzma.open(chemin.with_suffix(".pkl.xz"), "rb")) as f:
        p = pickle.load(f)
    lues, n = _arbres([m for sac in p["modeles"] for m in sac])
    return {"type": "gbm_sacs", "colonnes": len(p["colonnes"]), "colonnes_lues": len(lues), "arbres": n,
            "circuits": len(p["circuits"])}


def sobriete():
    """Composants de la directe (même définition partout) : colonnes lues, arbres, entrées."""
    d = "directe"
    comp = {"GRU mixte": f"rc0_livre_g04/{d}", "G2": f"rc0_livre_g04/{d}/recette", "GRU S (5 g)": f"rc3_gruA_norm/{d}",
            "R": f"rc0_livre_g04/{d}/gbm", "S": f"rc1_gbm_S/{d}", "D3L (NPWD2551)": f"rc0_livre_g04/{d}/specialiste/gbm_NPWD2551",
            "B (NPWD2551)": "st_B_directe", "physique (NPWD2532)": f"rc0_livre_g04/{d}/specialiste/physique"}
    sortie = {k: sobriete_composant(RACINE / "runs" / v) for k, v in comp.items() if (RACINE / "runs" / v).exists()}
    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "sobriete.json").write_text(json.dumps(sortie, indent=1) + "\n")
    print(json.dumps(sortie, indent=1))


# ---------------------------------------------------------------- poids finaux de chaque candidat

FINAUX = RACINE / "poids_candidats" / "structure"


def finaux():
    """Dossier de poids complet (annee1 + annee2) de chaque candidat, installé ou non : base = poids/ (livré :
    GRU mixte, G2, R, D3L, physique, marqueurs) ; composants nouveaux = runs/st_final_* (Kaggle S4)."""
    import os
    import shutil
    if not (RACINE / "runs" / "st_final_gruS10" / "recette.pt").is_file():
        from outils.fusionner_recette import fusionner
        fusionner(RACINE / "runs" / "st_final_gruS_g04", RACINE / "runs" / "st_final_gruS_g59",
                  RACINE / "runs" / "st_final_gruS10")
    base = ["--base", "../poids", "--directions", "final"]
    specs = {"L": [], "C1_g5": ["--recette", "st_final_gruS_g04", "--part", 0.5]}
    for g, gru in (("g10", "st_final_gruS10"), ("g5", "st_final_gruS_g04")):
        r = ["--recette", gru, "--part", 1]
        specs[f"T1_{g}"] = r
        specs[f"T2_{g}"] = r + ["--gbm2", "st_final_gbmS", "--poids-gbm2", 0.5, "--b", "st_final_B", "--part-b", 0.5]
        specs[f"T3_{g}"] = specs[f"T3sr_{g}"] = r + ["--gbm", "st_final_gbmS", "--b", "st_final_B", "--part-b", 1]
    for c, spec in specs.items():
        cible = FINAUX / c
        if cible.exists():
            shutil.rmtree(cible)
        candidat("--nom", f"st_fin_{c}", *base, *spec)
        cible.parent.mkdir(parents=True, exist_ok=True)
        os.rename(RACINE / "runs" / f"st_fin_{c}" / "final", cible)
        (RACINE / "runs" / f"st_fin_{c}").rmdir()
        if c.startswith("T3sr"):  # même dossier que T3, prédiction sans recalage de la part boosting
            meta = json.loads((cible / "modele.json").read_text())
            meta["options_prediction"] = {**meta.get("options_prediction", {}), "recalage": 0.0}
            (cible / "modele.json").unlink()
            (cible / "modele.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        print(c, "->", cible.relative_to(RACINE), flush=True)


# ---------------------------------------------------------------- décision mécanique

def score(r, cand, v, g):
    return r.get(f"{cand}_{v}_{g}", {}).get("rmse")


def decider():
    r = json.loads((SORTIE / "bilan.json").read_text())
    r = {k.replace("T3_", "T3sr_", 1).removesuffix("_sr") if k.endswith("_sr") else k: x for k, x in r.items()}
    temporel = [(v, g) for v in ("directe", "inverse") for g in ("g04", "g59")]
    livre_t = {vg: score(r, "L", *vg) for vg in temporel}
    livre_c = {c: np.mean([score(r, "L", c, g) for g in ("g04", "g59")]) for c in COUPURES}
    lignes, choix = [], None
    for cle in ORDRE:
        cand, version = cle.split("_")
        if version == "g10":
            t = {vg: score(r, cand, vg[0], "g10") for vg in temporel}
            cs = {c: score(r, cand, c, "g10") for c in COUPURES}
        else:
            t = {vg: score(r, cand, *vg) for vg in temporel}
            cs = {c: (np.mean([score(r, cand, c, g) for g in ("g04", "g59")])
                      if all(score(r, cand, c, g) is not None for g in ("g04", "g59")) else None) for c in COUPURES}
        if any(x is None for x in list(t.values()) + list(cs.values())):
            lignes.append({"candidat": cle, "complet": False})
            continue
        dt = {f"{v}_{g}": t[(v, g)] - livre_t[(v, g)] for v, g in temporel}
        dc = {c: cs[c] - livre_c[c] for c in COUPURES}
        groupes = ("g10",) if version == "g10" else ("g04", "g59")
        finies = all(r[f"{cand}_{v}_{g}"]["finies"] for v in VALIDATIONS for g in groupes)
        gardes = {"moyenne_4 <= +0.0010": float(np.mean(list(dt.values()))) <= 0.0010,
                  "chacune des 4 <= +0.0020": max(dt.values()) <= 0.0020,
                  "moyenne coupures <= +0.0020": float(np.mean(list(dc.values()))) <= 0.0020,
                  "chaque coupure <= +0.0040": max(dc.values()) <= 0.0040,
                  "sorties finies": finies}
        admissible = all(gardes.values())
        lignes.append({"candidat": cle, "complet": True, "moyenne_4": float(np.mean(list(t.values()))),
                       "delta_moyenne_4": float(np.mean(list(dt.values()))), "deltas_4": dt,
                       "moyenne_coupures": float(np.mean(list(cs.values()))),
                       "delta_moyenne_coupures": float(np.mean(list(dc.values()))), "deltas_coupures": dc,
                       "gardes": gardes, "admissible": admissible})
        if admissible and choix is None:
            choix = cle
    decision = {"livre": {"temporel": {f"{v}_{g}": x for (v, g), x in livre_t.items()},
                          "moyenne_4": float(np.mean(list(livre_t.values()))), "coupures": livre_c,
                          "moyenne_coupures": float(np.mean(list(livre_c.values())))},
                "candidats": lignes, "installe": choix or "L (livré conservé)"}
    (SORTIE / "decision.json").write_text(json.dumps(decision, indent=1, default=float) + "\n")
    for x in lignes:
        if x["complet"]:
            print(f"{x['candidat']:9s} Δ4 {x['delta_moyenne_4']:+.6f} max {max(x['deltas_4'].values()):+.6f} "
                  f"Δcoup {x['delta_moyenne_coupures']:+.6f} max {max(x['deltas_coupures'].values()):+.6f} "
                  f"{'ADMISSIBLE' if x['admissible'] else 'refusé : ' + ', '.join(k for k, ok in x['gardes'].items() if not ok)}")
        else:
            print(f"{x['candidat']:9s} incomplet")
    print("Installé :", decision["installe"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["assembler", "predire", "bilan", "decider", "sobriete", "finaux"])
    ap.add_argument("--validations", nargs="+", default=VALIDATIONS)
    ap.add_argument("--candidats", nargs="+", default=CANDIDATS)
    ap.add_argument("--controles", action="store_true")
    ap.add_argument("--travaux", type=int, default=4)
    ap.add_argument("--filtre", default="", help="sous-chaîne du nom des dossiers à prédire")
    a = ap.parse_args()
    if a.action == "assembler":
        assembler(a.validations, a.candidats, a.controles)
    elif a.action == "predire":
        t = [x for x in travaux() if a.filtre in str(x[0]) and x[1] in a.validations]
        with ThreadPoolExecutor(a.travaux) as pool:
            list(pool.map(predire_un, t))
    elif a.action == "bilan":
        bilan()
    elif a.action == "sobriete":
        sobriete()
    elif a.action == "finaux":
        finaux()
    else:
        decider()


if __name__ == "__main__":
    main()
