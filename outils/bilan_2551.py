"""Bilan de NPWD2551 sur les trois validations internes (docs/JOURNAL_2551.md, programme 2).

    python outils/bilan_2551.py --nom n2_diag --validations directe inverse
    python outils/bilan_2551.py --nom n2_A --candidats xgb=n2_xgb20 ea=n2_xgb20_ea lr=n2_lent+n2_rapide
    python outils/bilan_2551.py --nom n2_zero --zero

Base = modèle livré : directe et inverse, prédictions de bout en bout runs/rc0_livre_g04|g59 ; chrono,
0.5 × (0.7 GRU mixte + 0.3 GRU recette G2) + 0.5 × boosting spécialiste D3L + orbite (runs/rc7_chrono_*,
runs/n2_base_chrono), même structure que le livré. Un candidat s'écrit NOM=RUN ou NOM=RUN1+RUN2 (somme, ex.
niveau lent + composante rapide) ; ses prédictions sont runs/RUN/DIRECTION/pred.csv.
5 cas : directe et inverse × graines de la base 0–4 / 5–9, et chrono. Vérité inconnue : jamais notée.

Mélanges : base + 1 à 3 candidats, poids positifs de somme 1 au pas de 0.1. Poids d'une validation choisis
sur les deux autres (moyenne des groupes de graines) ; poids communs choisis sur les trois (installation).
Critère fixé avant les mesures : avec les poids transférés, RMSE en baisse dans les 5 cas, aucune moitié de
période en hausse de plus de 0.002, gain moyen ≥ 0.001.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from outils.specialistes import grille, rmse  # noqa: E402
from outils.valider import TOUTES  # noqa: E402

C = "NPWD2551"
H = 3_600_000
GROUPES = {"directe": ("g04", "g59"), "inverse": ("g04", "g59"), "chrono": ("-",)}
SEUIL_EPISODE, JOURS_EPISODE = -0.1, 3
SEUIL_ON, FENETRE_BASCULE = 0.15, 14
_CACHE = {}
SPECIALISTE = "n2_spec"  # prédictions du spécialiste S seul (programme 4 : b0_gel/specialiste_exact, S installé exact)


def verite(v):
    if ("y", v) not in _CACHE:
        _CACHE["y", v] = pd.read_csv(TOUTES[v][1] / "verite_horaire.csv", index_col="ut_ms")[C]
    return _CACHE["y", v]


def lire(chemin, v):
    return pd.read_csv(chemin, index_col="ut_ms")[C].reindex(verite(v).index).to_numpy()


def base(v, g):
    if v != "chrono":
        return lire(RACINE / "runs" / f"rc0_livre_{g}" / v / "pred.csv", v)
    r = RACINE / "runs"
    gru = 0.7 * lire(r / "rc7_chrono_mixte/chrono/pred.csv", v) + 0.3 * lire(
        r / 'rc7_chrono_g2/chrono/pred_{"recalage": 1.0}.csv', v)
    return np.clip(0.5 * gru + 0.5 * lire(r / "n2_base_chrono/chrono/pred.csv", v), 0, None)


def candidat(spec, v):
    return np.clip(sum(lire(RACINE / "runs" / run / v / "pred.csv", v) for run in spec.split("+")), 0, None)


def regimes(v):
    """Régimes de chaque heure notée, tirés du contexte du test seul (et du niveau vrai, diagnostic)."""
    if ("r", v) in _CACHE:
        return _CACHE["r", v]
    from dio import donnees
    test = donnees.lire_dossier_test(TOUTES[v][1] / "test")
    x = test["X"].reindex(verite(v).index)
    jour = x.index // (24 * H)
    ombre_jour = x["evtf_MAR_UMBRA_minutes"].groupby(jour).transform("sum")
    ftl = [c for c in x.columns if c.startswith("ftl_") and c.endswith("_minutes")
           and c not in ("ftl_coverage_minutes", "ftl_overlap_minutes", "ftl_comms_minutes")]
    pointage = x[ftl].idxmax(axis=1).str.removeprefix("ftl_").str.removesuffix("_minutes")
    dmop = test["dmop"]
    temps = np.sort(dmop.ut_ms[dmop.subsystem.str.match(r"ATTT30[56]")].to_numpy(dtype=np.int64))
    h = x.index.to_numpy(dtype=np.int64)
    attt = (np.searchsorted(temps, h + 25 * H) - np.searchsorted(temps, h - 24 * H)) > 0
    y = verite(v)
    niveau = y.groupby(y.index // (168 * H)).transform("mean")
    _CACHE["r", v] = pd.DataFrame({
        "saison d'éclipses": np.where(ombre_jour > 0, "oui", "non"), "pointage": pointage.to_numpy(),
        "ATTT ±24 h": np.where(attt, "oui", "non"),
        "niveau vrai hebdo": np.where(niveau > 0.15, "haut", "bas")}, index=x.index)
    return _CACHE["r", v]


def mesures(p, v, details=True):
    y = verite(v).to_numpy()
    k = np.isfinite(y)
    e = p[k] - y[k]
    n = len(e)
    s = {"rmse": rmse(e), "mae": float(np.mean(np.abs(e))), "biais": float(e.mean()),
         "moities": [rmse(e[:n // 2]), rmse(e[n // 2:])],
         "quarts": [rmse(e[i * n // 4:(i + 1) * n // 4]) for i in range(4)]}
    mois = pd.Series(e).groupby(pd.to_datetime(verite(v).index[k], unit="ms").strftime("%Y-%m").to_numpy())
    s["mois"] = {m: {"n": len(g), "rmse": rmse(g.to_numpy()), "mae": float(np.mean(np.abs(g))), "biais": float(g.mean())}
                 for m, g in mois}
    if not details:
        return s
    t = pd.Series(e, index=verite(v).index[k])
    for nom, duree in (("semaine", 168), ("jour", 24)):
        niveau = t.groupby(t.index // (duree * H)).transform("mean")
        s[f"mse_niveau_{nom}"] = float(np.mean(niveau ** 2))
        s[f"mse_intra_{nom}"] = float(np.mean((t - niveau) ** 2))
    jours = t.groupby(t.index // (24 * H)).mean()
    sous = (jours < SEUIL_EPISODE).to_numpy()
    episodes, debut = [], None
    for i, b in enumerate(np.r_[sous, False]):
        if b and debut is None:
            debut = i
        elif not b and debut is not None:
            if i - debut >= JOURS_EPISODE:
                episodes.append((debut, i))
            debut = None
    dans = np.zeros(len(t), bool)
    jour_heure = (t.index // (24 * H)).to_numpy()
    for a, b in episodes:
        dans |= np.isin(jour_heure, jours.index[a:b])
    s["episodes_sous_estimation"] = {
        "nombre": len(episodes), "jours": int(sum(b - a for a, b in episodes)),
        "part_mse": float(np.sum(e[dans] ** 2) / np.sum(e ** 2)) if n else 0.0,
        "liste": [{"debut_jour": int(jours.index[a]), "jours": b - a, "erreur_moy": float(jours.iloc[a:b].mean())}
                  for a, b in episodes]}
    r = regimes(v).loc[t.index]
    s["regimes"] = {col: {str(val): {"n": int(m.sum()), "rmse": rmse(e[m]), "biais": float(e[m].mean()),
                                     "part_mse": float(np.sum(e[m] ** 2) / np.sum(e ** 2))}
                          for val in r[col].unique() for m in [(r[col] == val).to_numpy()] if m.sum() >= 24}
                    for col in r.columns}
    semaines = t.groupby(t.index // (168 * H))
    s["semaines"] = {int(w): {"rmse": rmse(g.to_numpy()), "biais": float(g.mean())} for w, g in semaines}
    return s


def cas(validations):
    return [(v, g) for v in validations for g in GROUPES[v]]


def debut_chrono():
    """Première heure du power_init de la chrono : A2a = heures de la directe avant cet instant."""
    if "chrono0" not in _CACHE:
        _CACHE["chrono0"] = int(pd.read_csv(TOUTES["chrono"][1] / "test" / "power_init.csv", usecols=["ut_ms"]).ut_ms.min())
    return _CACHE["chrono0"]


# Poids d'une validation choisis sur des périodes disjointes de la sienne (programme 3) : la chrono teste A2b,
# déjà dans le test de la directe ; elle prend ses poids sur l'inverse (A1) et la première moitié de la directe.
SOURCES = {"inverse": [("directe", None)], "directe": [("inverse", None)],
           "chrono": [("inverse", None), ("directe", "A2a")]}


def melanges(cands, validations, taille_max):
    """Pour chaque sous-ensemble de candidats : poids transférés (périodes disjointes, SOURCES) et communs."""
    erreurs, a2a = {}, {}
    for v, g in cas(["directe", "inverse", "chrono"]):
        y = verite(v).to_numpy()
        k = np.isfinite(y)
        colonnes = [base(v, g)] + [candidat(spec, v) for spec in cands.values()]
        erreurs[v, g] = np.stack(colonnes, 1)[k] - y[k][:, None]
        if v == "directe":
            a2a[g] = verite(v).index.to_numpy()[k] < debut_chrono()
    noms = list(cands)
    sortie = {}
    for taille in range(1, min(taille_max, len(noms)) + 1):
        for groupe in itertools.combinations(range(len(noms)), taille):
            idx = [0] + [i + 1 for i in groupe]
            poids = grille(len(idx))

            def courbe(v, partie=None):
                return np.mean([[rmse((erreurs[v, g][a2a[g]] if partie else erreurs[v, g])[:, idx] @ w)
                                 for w in poids] for g in GROUPES[v]], axis=0)

            courbes = {(v, partie): courbe(v, partie) for sources in SOURCES.values() for v, partie in sources}
            commun = poids[int(np.argmin(sum(courbe(v) for v in ("directe", "inverse", "chrono"))))]
            res, gains = {}, []
            for v, g in cas(validations):
                w = poids[int(np.argmin(sum(courbes[src] for src in SOURCES[v])))]
                e = erreurs[v, g]
                n = len(e)
                avant, apres = rmse(e[:, 0]), rmse(e[:, idx] @ w)
                demis = [rmse(e[s][:, idx] @ w) - rmse(e[s, 0]) for s in (slice(0, n // 2), slice(n // 2, None))]
                gains.append(avant - apres)
                res[f"{v}/{g}"] = {"base": avant, "transfere": apres, "poids": w.tolist(), "moities": demis,
                                   "commun": rmse(e[:, idx] @ commun)}
            nom = "+".join(noms[i] for i in groupe)
            sortie[nom] = {"poids_communs": dict(zip(["base"] + [noms[i] for i in groupe], commun.tolist())),
                           "cas": res, "gain_moyen": float(np.mean(gains))}
    return sortie


def transferer(cands, nom):
    """Prédictions du mélange base + candidat nom avec les poids transférés (SOURCES), par cas."""
    poids = melanges({nom: cands[nom]}, ["directe", "inverse", "chrono"], 1)[nom]["cas"]
    return {(v, g): np.clip(np.array(poids[f"{v}/{g}"]["poids"]) @ np.stack([base(v, g), candidat(cands[nom], v)]),
                            0, None) for v, g in cas(["directe", "inverse", "chrono"])}, poids


def specialiste(v):
    return lire(RACINE / "runs" / SPECIALISTE / v / "pred.csv", v)


def remplacement(spec, v, g):
    """Forme de l'installation de D3L : la part boosting (0.5 × S) remplacée par 0.5 × candidat."""
    return np.clip(base(v, g) + 0.5 * (candidat(spec, v) - specialiste(v)), 0, None)


def bascules(on):
    """Indices des jours où le régime vrai change durablement (séquences de moins de 3 jours absorbées)."""
    etat = on.copy()
    debut = 0
    for i in range(1, len(etat) + 1):
        if i == len(etat) or etat[i] != etat[debut]:
            if i - debut < JOURS_EPISODE and debut > 0:
                etat[debut:i] = etat[debut - 1]
            debut = i
    return np.flatnonzero(np.diff(etat.astype(int)) != 0) + 1


def decomposer(p, v):
    """MSE = part journalière (niveau / date de bascule / régime faux isolé) + part horaire (heure − moyenne du
    jour). Jour haut si sa moyenne vraie dépasse SEUIL_ON ; « bascule » : jour au régime prédit faux à
    ±FENETRE_BASCULE jours d'une bascule durable vraie. Mesures dans ces fenêtres et hors."""
    y = verite(v)
    k = np.isfinite(y.to_numpy())
    t = pd.DataFrame({"e": p[k] - y.to_numpy()[k], "y": y.to_numpy()[k], "p": p[k]}, index=y.index[k])
    jour = t.index // (24 * H)
    j = t.groupby(jour).mean()
    vrai, predit = (j.y > SEUIL_ON).to_numpy(), (j.p > SEUIL_ON).to_numpy()
    idx = bascules(vrai)
    proche = np.zeros(len(j), bool)
    numero = np.arange(len(j))
    for b in idx:
        proche |= np.abs(numero - b) <= FENETRE_BASCULE
    categorie = np.where(vrai == predit, "niveau", np.where(proche, "bascule", "regime_faux"))
    cat_h = pd.Series(categorie, index=j.index).reindex(jour).to_numpy()
    ej = pd.Series(j.e.to_numpy(), index=j.index).reindex(jour).to_numpy()
    intra = t.e.to_numpy() - ej
    n, mse = len(t), float(np.mean(t.e ** 2))
    parts = {c: float(np.sum(ej[cat_h == c] ** 2) / n) for c in ("niveau", "bascule", "regime_faux")}
    parts["horaire"] = float(np.sum(intra ** 2) / n)
    proche_h = pd.Series(proche, index=j.index).reindex(jour).to_numpy()
    e = t.e.to_numpy()
    fen = {nom: {"heures": int(m.sum()), "rmse": rmse(e[m]), "mae": float(np.mean(np.abs(e[m]))),
                 "biais": float(e[m].mean())} for nom, m in (("transitions", proche_h), ("hors", ~proche_h)) if m.any()}
    return {"mse": mse, "parts": parts, "parts_relatives": {c: x / mse for c, x in parts.items()},
            "jours": {c: int((categorie == c).sum()) for c in ("niveau", "bascule", "regime_faux")},
            "bascules": [int(j.index[b]) for b in idx], "fenetres": fen}


def mesures_cas(p, v):
    s = mesures(p, v, False)
    s["decomposition"] = decomposer(p, v)
    return s


PERIODES = {"A1": ["inverse"], "A2": ["directe"], "A2b": ["chrono"]}


def regle(avant, apres):
    """Règle d'installation du programme 3 (fixée avant les mesures) ; avant, apres : {cas: mesures_cas}."""
    gains = {c: avant[c]["rmse"] - apres[c]["rmse"] for c in avant}
    motifs = []
    if min(gains.values()) <= 0:
        motifs.append("RMSE en hausse dans un cas")
    for periode, vs in PERIODES.items():
        cs = [c for c in avant if c.split("/")[0] in vs]
        if np.mean([gains[c] for c in cs]) < 0.0005:
            motifs.append(f"gain < 0.0005 sur {periode}")
        if np.mean([apres[c]["mae"] - avant[c]["mae"] for c in cs]) > 0:
            motifs.append(f"MAE en hausse sur {periode}")
    if np.mean(list(gains.values())) < 0.0015:
        motifs.append("gain moyen < 0.0015")
    for c in avant:
        if max(np.subtract(apres[c]["moities"], avant[c]["moities"])) > 0.002:
            motifs.append(f"moitié en hausse > 0.002 ({c})")
        if abs(apres[c]["biais"]) - abs(avant[c]["biais"]) > 0.01:
            motifs.append(f"|biais| en hausse > 0.01 ({c})")
        fa, fb = avant[c]["decomposition"]["fenetres"], apres[c]["decomposition"]["fenetres"]
        if "transitions" in fa and fb["transitions"]["rmse"] - fa["transitions"]["rmse"] > 0.005:
            motifs.append(f"transitions en hausse > 0.005 ({c})")
    return {"gain_moyen": float(np.mean(list(gains.values()))), "gains": gains, "accepte": not motifs,
            "motifs": sorted(set(motifs))}


def evaluer(cands):
    """Livré, S seul, et pour chaque candidat : seul, remplacement de la part boosting, mélange transféré."""
    toutes = cas(["directe", "inverse", "chrono"])
    livre = {f"{v}/{g}": mesures_cas(base(v, g), v) for v, g in toutes}
    sortie = {"livre": livre, "specialiste": {f"{v}/{g}": mesures_cas(specialiste(v), v) for v, g in toutes}}
    for nom, spec in cands.items():
        seul = {f"{v}/{g}": mesures_cas(candidat(spec, v), v) for v, g in toutes}
        rempl = {f"{v}/{g}": mesures_cas(remplacement(spec, v, g), v) for v, g in toutes}
        preds, poids = transferer(cands, nom)
        mel = {f"{v}/{g}": mesures_cas(preds[v, g], v) for v, g in toutes}
        sortie[nom] = {"seul": seul, "remplacement": rempl, "melange": mel,
                       "poids_transferes": {c: poids[c]["poids"] for c in poids},
                       "regle_remplacement": regle(livre, rempl), "regle_melange": regle(livre, mel)}
    return sortie


def afficher(ev):
    cs = list(ev["livre"])
    print(f"{'':26s}" + "".join(f"{c:>12s}" for c in cs))

    def ligne(nom, r):
        print(f"{nom:26s}" + "".join(f"{r[c]['rmse']:12.4f}" for c in cs))
        print(f"{'  MAE / biais':26s}" + "".join(f"{r[c]['mae']:6.3f}/{r[c]['biais']:+.3f}" for c in cs))
        print(f"{'  transitions / hors':26s}" + "".join(
            f"{r[c]['decomposition']['fenetres'].get('transitions', {}).get('rmse', np.nan):6.3f}/"
            f"{r[c]['decomposition']['fenetres']['hors']['rmse']:.3f}" for c in cs))
        print(f"{'  niv/basc/faux/horaire %':26s}" + "".join(
            "{:>3.0f}/{:>2.0f}/{:>2.0f}/{:>2.0f}".format(*[100 * r[c]['decomposition']['parts_relatives'][k]
                                                       for k in ('niveau', 'bascule', 'regime_faux', 'horaire')])
            for c in cs))
    ligne("livré", ev["livre"])
    ligne("spécialiste S seul", ev["specialiste"])
    for nom, r in ev.items():
        if nom in ("livre", "specialiste"):
            continue
        for forme in ("seul", "remplacement", "melange"):
            ligne(f"{nom} {forme}", r[forme])
        for forme in ("remplacement", "melange"):
            rg = r[f"regle_{forme}"]
            print(f"  règle {forme}: gain moyen {rg['gain_moyen']:+.5f} {'ACCEPTÉ' if rg['accepte'] else 'refusé'}"
                  f" {'; '.join(rg['motifs'])}")
        print(f"  poids transférés (base, candidat) : {r['poids_transferes']}")


def porte(validations=("directe", "inverse", "chrono")):
    """Porte asymétrique du commutateur (programme 3 bis) : aux heures où le commutateur reconstruit est coupé,
    2551 = min(livré, niveau bas) (forme principale) ou = niveau bas (forme secondaire). Commutateur et niveau
    bas appris sur l'entraînement de la direction seulement ; livré inchangé ailleurs."""
    from baselines import gbm_sacs
    from dio import donnees
    schema = donnees.charger_schema()
    i = schema["target_columns"].index(C)
    sortie = {}
    for v in validations:
        annees = TOUTES[v][0]
        annees = [annees] if isinstance(annees, str) else annees
        periodes = [donnees.charger_annee(a, schema) for a in annees]
        tables = [donnees.lire_dmop(a) for a in annees]
        ys = [y[:, i] for _, _, y in periodes]
        comm = gbm_sacs.apprendre_commutateur(tables, [h for h, _, _ in periodes], ys)
        bas = np.concatenate([y[gbm_sacs.regime_journalier(h, y) == 0] for (h, _, _), y in zip(periodes, ys)])
        niveau = float(np.nanmedian(bas))
        test = donnees.lire_dossier_test(TOUTES[v][1] / "test")
        etat = gbm_sacs.variables_commutateur(test["dmop"], test["heures"], comm, gbm_sacs.regime_initial(test))
        coupe = (etat["comm_haut"].reindex(verite(v).index) == 0).to_numpy()
        y = verite(v).to_numpy()
        k = np.isfinite(y)
        jours = gbm_sacs.regime_journalier(verite(v).index.to_numpy(), y)
        info = {"commutateur": {f: {c: round(p, 3) for c, p in m.items()} for f, m in comm.items()},
                "niveau_bas": niveau, "heures_coupees": int((coupe & k).sum()), "heures": int(k.sum()),
                "part_jours_hauts_si_coupe": float(np.nanmean(jours[coupe & k])) if (coupe & k).any() else None,
                "y_moyen_si_coupe": float(np.mean(y[coupe & k])) if (coupe & k).any() else None}
        for g in GROUPES[v]:
            p = base(v, g)
            sortie[f"{v}/{g}"] = {"info": info, "livre": mesures_cas(p, v),
                                  "min": mesures_cas(np.where(coupe, np.minimum(p, niveau), p), v),
                                  "remplacement": mesures_cas(np.where(coupe, niveau, p), v),
                                  "moyenne_livre_si_coupe": float(np.mean(p[coupe & k])) if (coupe & k).any() else None}
    livre = {c: r["livre"] for c, r in sortie.items()}
    regles = {forme: regle(livre, {c: r[forme] for c, r in sortie.items()}) for forme in ("min", "remplacement")}
    return sortie, regles


def zero(validations):
    """Règle 2551 = 0 pendant k h après une commande (ATTT306C, ATTT305O), appliquée à la base."""
    from dio import donnees
    sortie = {}
    for v in validations:
        test = donnees.lire_dossier_test(TOUTES[v][1] / "test")
        h = verite(v).index.to_numpy(dtype=np.int64)
        for code in ("ATTT306C", "ATTT305O"):
            temps = np.sort(test["dmop"].ut_ms[test["dmop"].subsystem == code].to_numpy(dtype=np.int64))
            for k in (1, 3, 6):
                masque = (np.searchsorted(temps, h + H) - np.searchsorted(temps, h - (k - 1) * H)) > 0
                for g in GROUPES[v]:
                    p = base(v, g)
                    sortie[f"{v}/{g}/{code}/{k}h"] = {
                        "heures": int((masque & np.isfinite(verite(v).to_numpy())).sum()),
                        "base": mesures(p, v, False)["rmse"], "regle": mesures(np.where(masque, 0.0, p), v, False)["rmse"]}
    return sortie


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--validations", nargs="+", default=["directe", "inverse", "chrono"])
    ap.add_argument("--candidats", nargs="*", default=[], help="NOM=RUN ou NOM=RUN1+RUN2")
    ap.add_argument("--taille", type=int, default=3, help="nombre maximal de candidats par mélange")
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--porte", action="store_true", help="programme 3 bis : porte asymétrique du commutateur")
    ap.add_argument("--evaluer", action="store_true",
                    help="programme 3 : livré, S, candidats seuls / remplacement / mélange transféré, règle")
    a = ap.parse_args()
    cands = dict(c.split("=", 1) for c in a.candidats)
    sortie = {"validations": a.validations, "candidats": cands}
    if a.porte:
        sortie["porte"], sortie["regles"] = porte()
        for c, r in sortie["porte"].items():
            inf = r["info"]
            print(f"{c:12s} coupé {inf['heures_coupees']:5d}/{inf['heures']} h, niveau bas {inf['niveau_bas']:.4f}, "
                  f"y moyen si coupé {inf['y_moyen_si_coupe']}, livré moyen si coupé {r['moyenne_livre_si_coupe']}")
            for forme in ("livre", "min", "remplacement"):
                m = r[forme]
                print(f"   {forme:13s} RMSE {m['rmse']:.4f} MAE {m['mae']:.4f} biais {m['biais']:+.4f} moitiés "
                      f"{m['moities'][0]:.4f}/{m['moities'][1]:.4f} transitions "
                      f"{m['decomposition']['fenetres'].get('transitions', {}).get('rmse', float('nan')):.4f}")
        for forme, rg in sortie["regles"].items():
            print(f"règle {forme}: gain moyen {rg['gain_moyen']:+.5f} {'ACCEPTÉ' if rg['accepte'] else 'refusé'} "
                  f"{'; '.join(rg['motifs'])}")
        print("commutateurs appris :", {c.split('/')[0]: r["info"]["commutateur"] for c, r in sortie["porte"].items()})
    elif a.evaluer:
        sortie["evaluation"] = evaluer(cands)
        afficher(sortie["evaluation"])
    elif a.zero:
        sortie["zero"] = zero(a.validations)
        for k, r in sortie["zero"].items():
            print(f"{k:28s} heures {r['heures']:4d}  base {r['base']:.4f} -> règle {r['regle']:.4f}")
    else:
        sortie["seuls"] = {"base": {f"{v}/{g}": mesures(base(v, g), v) for v, g in cas(a.validations)}}
        for nom, spec in cands.items():
            sortie["seuls"][nom] = {f"{v}/{g}": mesures(candidat(spec, v), v, False) for v, g in cas(a.validations)}
        print(f"{'seul':14s}" + "".join(f"{v[:3]}/{g:4s} " for v, g in cas(a.validations)) + "  (RMSE / MAE / biais)")
        for nom, r in sortie["seuls"].items():
            print(f"{nom:14s}" + "".join(f"{s['rmse']:.4f}    " for s in r.values()))
            print(f"{'':14s}" + "".join(f"{s['mae']:.4f}    " for s in r.values()))
            print(f"{'':14s}" + "".join(f"{s['biais']:+.4f}   " for s in r.values()))
        if cands:
            sortie["melanges"] = melanges(cands, a.validations, a.taille)
            for nom, r in sorted(sortie["melanges"].items(), key=lambda kv: -kv[1]["gain_moyen"]):
                print(f"{nom:22s} gain {r['gain_moyen']:+.5f} communs {r['poids_communs']}")
                for k, c in r["cas"].items():
                    print(f"   {k:12s} {c['base']:.4f} -> {c['transfere']:.4f} poids {c['poids']} "
                          f"moitiés {c['moities'][0]:+.4f} {c['moities'][1]:+.4f}")
    dossier = RACINE / "runs" / a.nom
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "bilan.json").write_text(json.dumps(sortie, indent=1) + "\n")
    print(f"runs/{a.nom}/bilan.json")


if __name__ == "__main__":
    main()
