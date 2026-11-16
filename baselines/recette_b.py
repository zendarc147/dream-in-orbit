"""Recette B : boosting horaire indépendant pour NPWD2551 (tests/controles/recette_b/RECETTE_AUTONOME.md).

Module indépendant : numpy, pandas et scikit-learn seulement, aucune fonction de dio/ ni de
baselines/gbm_sacs.py. Les numéros de section renvoient à la recette.

Interprétations notées (docs/journaux/JOURNAL_2551.md, programme 4) : tris par ut_ms stables (kind="stable"), sauf
l'altitude (np.argsort par défaut, comme demandé) ; changements de variante datés à leur heure même hors grille ;
récences en float32 avant log1p (§8), distances aux changements en float64. Avec ces choix, X année 1 sur le
vocabulaire de contrôle redonne exactement l'empreinte sha256 de tests/controles/recette_b/controle_reference.json.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

H = 3_600_000
CIBLE = "NPWD2551"
PLAFOND = 720
ANGLES = ("sa", "sx", "sy", "sz")
CONTEXTES = ("saaf", "ltdata", "evtf", "ftl", "dmop")
PAIRES = (("PENS", "PENE"), ("MOCS", "MOCE"), ("PPNS", "PPNE"), ("PDNS", "PDNE"), ("UPBS", "UPBE"), ("UDBS", "UDBE"))
DECALES = ("ev_umbra", "ev_penumbra", "saaf_cos_sa", "saaf_cos_sx", "saaf_cos_sy", "saaf_cos_sz", "ftl_comms",
           "ftl_EARTH", "ftl_SLEW", "ev_log_alt")
EXCLUES = r"lt_earthmars|lt_sunmarsearth|^roll_|^ewm_|^dm_s_|^dm_c_|^dm_r_"
SEUIL = 50

# §11 : paramètres exacts (tests/controles/recette_b/parametres_reference.json).
PARAMS_B = {"loss": "squared_error", "learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 15, "max_depth": None,
            "min_samples_leaf": 100, "l2_regularization": 3.0, "max_bins": 255, "categorical_features": "from_dtype",
            "max_features": 1.0, "quantile": None, "verbose": 0, "monotonic_cst": None, "interaction_cst": None,
            "warm_start": False, "early_stopping": "auto", "scoring": "loss", "validation_fraction": 0.1,
            "n_iter_no_change": 10, "tol": 1e-7, "random_state": 0}


# ---------------------------------------------------------------- §1 grille, cibles, lecture

def grille(debut, fin):
    return np.arange(debut // H * H, fin // H * H + H, H, dtype=np.int64)


def cibles(power_csv):
    """(grille, noms des 33 circuits, y float32 avec NaN) : moyenne horaire arithmétique par circuit."""
    p = pd.read_csv(power_csv)
    p["ut_ms"] = p.ut_ms // H * H
    y = p.groupby("ut_ms").mean()
    g = grille(int(y.index.min()), int(y.index.max()))
    return g, list(y.columns), y.reindex(g).to_numpy(dtype=np.float32)


def lire(dossier):
    return {nom: pd.read_csv(Path(dossier) / f"{nom}.csv") for nom in CONTEXTES}


def couper(tables, coupure=None, depuis=None):
    """Tables restreintes aux lignes de temps < coupure (ou ≥ depuis) ; ltdata garde son jour entamé."""
    sortie = {}
    for nom, t in tables.items():
        temps = t["utb_ms"] if nom == "ftl" else t["ut_ms"]
        jour = 24 * H
        if coupure is not None:
            garder = temps < ((coupure // jour + 1) * jour if nom == "ltdata" else coupure)
        else:
            garder = temps >= ((depuis // jour) * jour if nom == "ltdata" else depuis)
        sortie[nom] = t[garder].reset_index(drop=True)
    return sortie


def _interp(table, g):
    """§1 : union des timestamps, interpolation selon l'index (bords maintenus), grille, ffill/bfill."""
    if table.empty:
        return pd.DataFrame(np.nan, index=g, columns=table.columns)
    tout = table.reindex(np.union1d(table.index.to_numpy(dtype=np.int64), g))
    return tout.interpolate(method="index", limit_direction="both").reindex(g).ffill().bfill()


# ---------------------------------------------------------------- §2 SAAF, §3 LTDATA

def saaf(t, g):
    t = t.copy()
    h = t.ut_ms // H * H
    for a in ANGLES:
        t[f"cos_{a}"] = np.cos(np.deg2rad(t[a]))
    groupes = t.groupby(h)
    table = pd.DataFrame({**{f"saaf_{a}": groupes[a].mean() for a in ANGLES},
                          **{f"saaf_cos_{a}": groupes[f"cos_{a}"].mean() for a in ANGLES},
                          "saaf_sa_std": groupes["sa"].std().fillna(0.0)})
    return _interp(table, g)


def ltdata(t, g):
    table = t.set_index("ut_ms").sort_index()
    table.columns = [f"lt_{c}" for c in table.columns]
    return _interp(table.astype(float), g)


# ---------------------------------------------------------------- §4 intervalles

def fraction(g, debuts, fins):
    """Part de [h, h + H) couverte par l'union des intervalles [s, e) (fusion des chevauchants et contigus)."""
    debuts, fins = np.asarray(debuts, dtype=np.int64), np.asarray(fins, dtype=np.int64)
    garder = fins > debuts
    debuts, fins = debuts[garder], fins[garder]
    if not len(debuts):
        return np.zeros(len(g))
    ordre = np.argsort(debuts, kind="stable")
    debuts, fins = debuts[ordre], fins[ordre]
    s, e = [debuts[0]], [fins[0]]
    for a, b in zip(debuts[1:], fins[1:]):
        if a <= e[-1]:
            e[-1] = max(e[-1], b)
        else:
            s.append(a)
            e.append(b)
    s, e = np.array(s, dtype=np.int64), np.array(e, dtype=np.int64)
    cumul = np.concatenate([[0], np.cumsum(e - s)])

    def couvert(t):
        k = np.searchsorted(s, t, side="right") - 1
        sortie = np.zeros(len(t), dtype=np.int64)
        ok = k >= 0
        sortie[ok] = cumul[k[ok]] + np.minimum(t[ok] - s[k[ok]], e[k[ok]] - s[k[ok]])
        return sortie

    return (couvert(g + H) - couvert(g)) / H


def apparier(temps, debut, fin):
    """§4 : sur les événements triés par ut_ms, un START ouvre ou remplace l'ouverture ; le END suivant la ferme ;
    END sans START et START sans END ignorés. Rend (débuts, fins)."""
    ordre = np.argsort(temps, kind="stable")
    paires, ouvert = [], None
    for t, d, f in zip(np.asarray(temps)[ordre], np.asarray(debut)[ordre], np.asarray(fin)[ordre]):
        if d:
            ouvert = t
        elif f and ouvert is not None:
            paires.append((ouvert, t))
            ouvert = None
    p = np.array(paires, dtype=np.int64).reshape(-1, 2)
    return p[:, 0], p[:, 1]


def _compter(temps, g):
    """Nombre d'instants de chaque heure de la grille (0 hors événement)."""
    i = (np.asarray(temps, dtype=np.int64) // H * H - g[0]) // H
    i = i[(i >= 0) & (i < len(g))]
    return np.bincount(i, minlength=len(g)).astype(float)


# ---------------------------------------------------------------- §5 EVTF

def evtf(t, g):
    d = t.description.astype(str)
    temps = t.ut_ms.to_numpy(dtype=np.int64)

    def serie(debut, fin):
        cles = debut | fin
        return fraction(g, *apparier(temps[cles], debut[cles].to_numpy(), fin[cles].to_numpy()))

    sortie = {
        "ev_umbra": serie(d == "MAR_UMBRA_START", d == "MAR_UMBRA_END"),
        "ev_penumbra": serie(d == "MAR_PENUMBRA_START", d == "MAR_PENUMBRA_END"),
        "ev_moon_shadow": serie(d.str.match(r"^(PHO|DEI)_(UMBRA|PENUMBRA)_START$"),
                                d.str.match(r"^(PHO|DEI)_(UMBRA|PENUMBRA)_END$")),
        "ev_occ_mars": serie(d.str.match(r"^OCC_MARS_START"), d.str.match(r"^OCC_MARS_END")),
        "ev_occ_mars_km": serie(d.str.match(r"^OCC_MARS_\d+KM_START"), d.str.match(r"^OCC_MARS_\d+KM_END")),
        "ev_mrb_range": serie(d.str.match(r"^MRB_/_RANGE_\d+KM_START"), d.str.match(r"^MRB_/_RANGE_\d+KM_END")),
    }
    station = d.str.extract(r"^([A-Z0-9]{3})_(AOS|LOS)_")
    debuts, fins, somme = [], [], np.zeros(len(g))
    for s in sorted(station[0].dropna().unique()):
        m = (station[0] == s).to_numpy()
        a, b = apparier(temps[m], (station[1][m] == "AOS").to_numpy(), (station[1][m] == "LOS").to_numpy())
        debuts.append(a)
        fins.append(b)
        somme += fraction(g, a, b)
    sortie["ev_gs_any"] = fraction(g, np.concatenate(debuts or [[]]), np.concatenate(fins or [[]]))
    sortie["ev_gs_sum"] = somme
    sortie.update(orbite(t, g))
    sortie["ev_log_alt"] = log_altitude(t, g)
    sortie["ev_n_events"] = _compter(temps, g)
    return pd.DataFrame(sortie, index=g)


def orbite(t, g):
    d = t.description.astype(str)
    p = np.sort(t.ut_ms[d.str.startswith("PERICENTRE_PASSAGE")].to_numpy(dtype=np.int64))
    noms = ("ev_since_peri_h", "ev_to_peri_h", "ev_orbit_phase_cos", "ev_orbit_phase_sin", "ev_orbit_period_h")
    if len(p) < 2:
        return {n: np.zeros(len(g)) for n in noms}
    m = g + H // 2
    i = np.searchsorted(p, m, side="right") - 1
    avant, apres = p[np.clip(i, 0, len(p) - 1)], p[np.clip(i + 1, 0, len(p) - 1)]
    periode = np.where(apres > avant, apres - avant, np.median(np.diff(p)))
    depuis = np.where(i >= 0, m - avant, 0)
    phase = np.clip(depuis / periode, 0, 1)
    return dict(zip(noms, (depuis / H, np.maximum(apres - m, 0) / H, np.cos(2 * np.pi * phase),
                           np.sin(2 * np.pi * phase), periode / H)))


def log_altitude(t, g):
    d = t.description.astype(str)
    km = d.str.extract(r"^(\d+)_KM_(ASCEND|DESCEND)$")[0]
    m = km.notna().to_numpy()
    peri = np.sort(t.ut_ms[d.str.startswith("PERICENTRE_PASSAGE")].to_numpy(dtype=np.int64))
    apo = np.sort(t.ut_ms[d.str.startswith("APOCENTRE_PASSAGE")].to_numpy(dtype=np.int64))
    temps = np.concatenate([t.ut_ms.to_numpy(dtype=np.int64)[m], peri, apo])
    alt = np.concatenate([km[m].astype(float).to_numpy(), np.full(len(peri), 330.0), np.full(len(apo), 10100.0)])
    if len(temps) < 2:
        return np.full(len(g), np.log(10100.0))
    ordre = np.argsort(temps)
    temps, logs = temps[ordre], np.log(alt[ordre])
    return np.mean([np.interp(g + k * H // 8, temps, logs) for k in (1, 3, 5, 7)], axis=0)


# ---------------------------------------------------------------- §6 vocabulaire

def separer(cmd):
    """split_stem : dernière majuscule A–Z après le 4e caractère remplacée par « _ » ; (stem, variante)."""
    for j in range(len(cmd) - 1, 3, -1):
        if "A" <= cmd[j] <= "Z":
            return cmd[:j] + "_" + cmd[j + 1:], cmd[j]
    return cmd, ""


def _commandes(dmop):
    return dmop.subsystem.astype(str).str.strip('"')


def apprendre(periodes, coupure=None):
    """periodes : dicts de tables d'entraînement. Seuils appliqués dans chaque période, puis union triée.
    coupure : seules les commandes d'heure < coupure et les FTL de début < coupure (validation chronologique)."""
    prefixes, stems, couples, ftl, cmds = set(), set(), set(), set(), set()
    for t in periodes:
        dmop, f = t["dmop"], t["ftl"]
        if coupure is not None:
            dmop = dmop[dmop.ut_ms // H * H < coupure]
            f = f[f.utb_ms < coupure]
        cmd = _commandes(dmop)
        prefixes |= set(cmd.str[:4])
        propres = cmd[~cmd.str.contains(".", regex=False)]
        sv = propres.map(separer)
        c = sv.str[0].value_counts()
        stems |= set(c[c >= SEUIL].index)
        c = sv.value_counts()
        frequents = [k for k, n in c.items() if n >= SEUIL]
        par_stem = pd.Series([s for s, _ in frequents]).value_counts()
        couples |= {k for k in frequents if par_stem.get(k[0], 0) >= 2}
        c = propres.value_counts()
        cmds |= set(c[c >= SEUIL].index)  # information seulement : colonnes supprimées par la recette
        ftl |= set(f.type.astype(str).str.strip('"'))
    return {"cmds": sorted(cmds), "prefixes": sorted(prefixes), "switches": [list(k) for k in sorted(couples)],
            "stems": sorted(stems), "ftl_types": sorted(ftl)}


# ---------------------------------------------------------------- §7 FTL

def pointage(t, g, types):
    typ = t.type.astype(str).str.strip('"')
    s, e = t.utb_ms.to_numpy(dtype=np.int64), t.ute_ms.to_numpy(dtype=np.int64)
    sortie = {f"ftl_{k}": fraction(g, s[(typ == k).to_numpy()], e[(typ == k).to_numpy()]) for k in types}
    sortie["ftl_other"] = np.clip(1 - sum(sortie.values(), np.zeros(len(g))), 0, 1)
    comms = t.flagcomms.astype(str).str.upper().str.contains("TRUE").to_numpy()
    sortie["ftl_comms"] = fraction(g, s[comms], e[comms])
    sortie["ftl_n_switch"] = _compter(s, g)
    return pd.DataFrame(sortie, index=g)


# ---------------------------------------------------------------- §8 DMOP

def _comptes(i_heure, etiquettes, noms, n):
    """Matrice n × len(noms) des comptes horaires ; i_heure : indice de grille de chaque ligne (hors grille ignoré)."""
    position = {k: j for j, k in enumerate(noms)}
    j = np.array([position.get(e, -1) for e in etiquettes], dtype=np.int64)
    ok = (j >= 0) & (i_heure >= 0) & (i_heure < n)
    sortie = np.zeros((n, len(noms)))
    np.add.at(sortie, (i_heure[ok], j[ok]), 1)
    return sortie


def recence(comptes):
    """§8 : à l'heure i, min(i − last, 720), last = dernière heure < i avec une occurrence (−inf au départ),
    puis log1p ; float32."""
    n = comptes.shape[0]
    i = np.arange(n, dtype=float)[:, None]
    vu = np.where(comptes > 0, i, -np.inf)
    dernier = np.vstack([np.full((1, comptes.shape[1]), -np.inf), np.maximum.accumulate(vu, axis=0)[:-1]])
    return np.log1p(np.minimum(i - dernier, PLAFOND).astype(np.float32))


def changements(heures_chg, g):
    """Depuis (≤ h) et jusqu'au (≥ h) changement le plus proche, heures entières, plafond 720, log1p en float64
    (la variante float32 ne redonne pas l'empreinte de controle_reference.json ; celle-ci, oui)."""
    c = np.unique(np.asarray(heures_chg, dtype=np.int64))
    depuis = np.full(len(g), float(PLAFOND))
    jusqua = np.full(len(g), float(PLAFOND))
    if len(c):
        k = np.searchsorted(c, g, side="right") - 1
        ok = k >= 0
        depuis[ok] = np.minimum((g[ok] - c[k[ok]]) / H, PLAFOND)
        k = np.searchsorted(c, g, side="left")
        ok = k < len(c)
        jusqua[ok] = np.minimum((c[k[ok]] - g[ok]) / H, PLAFOND)
    return np.log1p(depuis), np.log1p(jusqua)


def commandes(t, g, vocab):
    n = len(g)
    cmd = _commandes(t)
    temps = t.ut_ms.to_numpy(dtype=np.int64)
    i_heure = (temps // H * H - g[0]) // H
    prefixe = cmd.str[:4].to_numpy()
    sv = cmd.map(separer)
    stem, variante = sv.str[0].to_numpy(), sv.str[1].to_numpy()
    pref, stems = vocab["prefixes"], vocab["stems"]
    blocs = []
    cp = _comptes(i_heure, prefixe, pref, n)
    blocs.append(pd.DataFrame(cp, index=g, columns=[f"dm_p_{p}" for p in pref]))
    blocs.append(pd.DataFrame({"dm_total": _compter(temps, g)}, index=g))
    ident = cmd.str.rsplit(".", n=1).str[-1].to_numpy()
    act = {}
    for debut, fin in PAIRES:
        a = pd.DataFrame({"id": ident[prefixe == debut], "s": temps[prefixe == debut]})
        b = pd.DataFrame({"id": ident[prefixe == fin], "e": temps[prefixe == fin]})
        j = a.merge(b, on="id", how="inner")
        j = j[j.e > j.s]
        act[f"dm_act_{debut}"] = fraction(g, j.s.to_numpy(), j.e.to_numpy())
    blocs.append(pd.DataFrame(act, index=g))
    blocs.append(pd.DataFrame(recence(cp), index=g, columns=[f"dm_pr_{p}" for p in pref]))
    cs = _comptes(i_heure, stem, stems, n)
    blocs.append(pd.DataFrame(recence(cs), index=g, columns=[f"dm_st_{s}" for s in stems]))
    blocs.append(pd.DataFrame(cs, index=g, columns=[f"dm_sc_{s}" for s in stems]))
    cumul = np.vstack([np.zeros((1, len(stems)), dtype=np.float32), np.cumsum(cs.astype(np.float32), axis=0)])
    t_ = np.arange(n)
    sn1 = cumul[np.minimum(t_ + 2, n)] - cumul[np.minimum(t_ + 1, n)]
    sn6 = cumul[np.minimum(t_ + 7, n)] - cumul[np.minimum(t_ + 1, n)]
    blocs.append(pd.DataFrame(sn1, index=g, columns=[f"dm_sn1_{s}" for s in stems]))
    blocs.append(pd.DataFrame(sn6, index=g, columns=[f"dm_sn6_{s}" for s in stems]))
    sw = {}
    for s in sorted({k[0] for k in vocab["switches"]}):
        m = stem == s
        ordre = np.argsort(temps[m], kind="stable")
        lettres, instants = variante[m][ordre], temps[m][ordre]
        chg = instants[1:][lettres[1:] != lettres[:-1]] // H * H
        sw[f"dm_sw_{s}_chg"], sw[f"dm_sw_{s}_next"] = changements(chg, g)
    blocs.append(pd.DataFrame(sw, index=g))
    return pd.concat(blocs, axis=1)


# ---------------------------------------------------------------- §9 dérivées, §10 sélection

def derivees(x):
    sol = x["lt_solarconstantmars"] / 600
    shade = np.clip(1 - x["ev_umbra"] - 0.5 * x["ev_penumbra"], 0, 1)
    fx = {}
    for a in ANGLES:
        fx[f"fx_{a}_plus"] = sol * np.maximum(x[f"saaf_cos_{a}"], 0) * shade
        fx[f"fx_{a}_minus"] = sol * np.maximum(-x[f"saaf_cos_{a}"], 0) * shade
    fx["fx_total"] = sol * shade
    umbra = x["ev_umbra"].to_numpy()
    i = np.arange(len(x), dtype=float)
    dernier = np.maximum.accumulate(np.where(umbra > 0, i, -float(PLAFOND)))
    total = pd.Series(fx["fx_total"], index=x.index)
    hist = {"hist_h_since_umbra": np.log1p(np.minimum(i - dernier, PLAFOND)),
            "hist_umbra_7h": x["ev_umbra"].rolling(7, min_periods=1).sum(),
            "hist_umbra_24h": x["ev_umbra"].rolling(24, min_periods=1).sum(),
            "hist_umbra_7d": x["ev_umbra"].rolling(168, min_periods=1).mean(),
            "hist_fx_6h": total.rolling(6, min_periods=1).mean(),
            "hist_fx_24h": total.rolling(24, min_periods=1).mean(),
            "hist_comms_24h": x["ftl_comms"].rolling(24, min_periods=1).mean()}
    ll = {f"ll_{s}_{k:+d}": x[s].shift(-k) for s in DECALES if s in x.columns for k in (-3, -2, -1, 1, 2, 3)}
    return [pd.DataFrame(fx, index=x.index), pd.DataFrame(hist, index=x.index), pd.DataFrame(ll, index=x.index)]


def construire(tables, g, vocab):
    """X float32 d'une période continue (grille g), vocabulaire appris sur l'entraînement."""
    g = np.asarray(g, dtype=np.int64)
    x = pd.concat([saaf(tables["saaf"], g), ltdata(tables["ltdata"], g), evtf(tables["evtf"], g),
                   pointage(tables["ftl"], g, vocab["ftl_types"]), commandes(tables["dmop"], g, vocab)], axis=1)
    x = pd.concat([x] + derivees(x), axis=1)
    x = x.loc[:, [not re.search(EXCLUES, c) for c in x.columns]]
    x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return x.astype(np.float32)


def periode(dossier, vocab, coupure=None):
    """(grille, X, circuits, y) d'une période d'entraînement (grille de power.csv)."""
    g, circuits, y = cibles(Path(dossier) / "power.csv")
    tables = lire(dossier)
    if coupure is not None:  # année entière puis tranche (variante de bord de la chrono)
        garder = g < coupure
        return g[garder], construire(tables, g, vocab)[garder], circuits, y[garder]
    return g, construire(tables, g, vocab), circuits, y


def test(dossier, vocab):
    """(grille complète, X, heures à prédire, y_init 33 circuits, n_init) d'un dossier de test."""
    dossier = Path(dossier)
    a_predire = pd.read_csv(dossier / "heures_a_predire.csv").ut_ms.to_numpy(dtype=np.int64)
    gi, circuits, y_init = cibles(dossier / "power_init.csv")
    g = grille(int(gi[0]), int(a_predire[-1]))
    return g, construire(lire(dossier), g, vocab), a_predire, circuits, y_init, len(gi)


# ---------------------------------------------------------------- §11 entraînement

def bornes(n):
    return np.linspace(0, n, 6).astype(int)


def masque(y33, circuits, collectif=True):
    """Lignes apprises : les 33 cibles connues (collectif, recette) ou la seule cible NPWD2551 (propre, livré)."""
    return np.isfinite(y33).all(axis=1) if collectif else np.isfinite(y33[:, circuits.index(CIBLE)])


def _ajuster_sac(x, y, lignes, params):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):  # un fil par sac, comme gbm_sacs._ajuster
        return HistGradientBoostingRegressor(**params).fit(x[lignes], y[lignes])


def ajuster(x, y, garder, params, n_jobs=5):
    """5 HGB : le sac i apprend hors de [borne_i, borne_i+1) (sans embargo), puis sur les seules lignes garder."""
    from joblib import Parallel, delayed
    n = len(x)
    b = bornes(n)
    t = np.arange(n)
    sacs = [((t < b[i]) | (t >= b[i + 1])) & garder for i in range(5)]
    return Parallel(n_jobs=n_jobs)(delayed(_ajuster_sac)(x, y, s, params) for s in sacs)


def moyenne(modeles, x):
    return np.mean([m.predict(x) for m in modeles], axis=0)


def predire_sacs(modeles, x):
    """§11 : moyenne des 5 sacs, puis bornage à 0 (aucun recalage)."""
    return np.clip(moyenne(modeles, x), 0, None)


# ---------------------------------------------------------------- modèle installable (docs/journaux/JOURNAL_STRUCTURE.md)

RACINE = Path(__file__).resolve().parent.parent


def _periode(nom):
    """annee1 -> data/annee1 ; sinon un chemin (ex. validation_glissant/c62/entrainement)."""
    return RACINE / "data" / nom if (RACINE / "data" / nom).is_dir() else Path(nom)


def entrainer(annees, dossier, n_jobs=5, **_):
    """Recette B stricte : vocabulaire des seules périodes d'entraînement, 5 sacs PARAMS_B, masque collectif,
    sans recalage ; paquet modele_b.pkl.xz (modèles, vocabulaire, colonnes)."""
    import lzma
    import pickle
    tables = [lire(_periode(a)) for a in annees]
    vocab = apprendre(tables)
    parts = [periode(_periode(a), vocab) for a in annees]
    x = pd.concat([p[1] for p in parts])
    y33 = np.concatenate([p[3] for p in parts])
    circuits = parts[0][2]
    garder = masque(y33, circuits)
    modeles = ajuster(x.to_numpy(np.float32), y33[:, circuits.index(CIBLE)], garder, PARAMS_B, n_jobs)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    with lzma.open(dossier / "modele_b.pkl.xz", "wb") as f:
        pickle.dump({"modeles": modeles, "vocab": vocab, "colonnes": list(x.columns)}, f)
    (dossier / "info.json").write_text(pd.Series({
        "annees": list(annees), "lignes": len(x), "colonnes": x.shape[1], "masque_lignes": int(garder.sum()),
        "iterations": [int(m.n_iter_) for m in modeles]}).to_json(indent=1) + "\n")


def predire(test_dir, out, dossier, **_):
    """NPWD2551 sur les heures à prédire (DataFrame indexé par ut_ms), bornée à 0, sans recalage."""
    import lzma
    import pickle
    with lzma.open(Path(dossier) / "modele_b.pkl.xz", "rb") as f:
        paquet = pickle.load(f)
    gt, xt, a_predire, _, _, _ = test(test_dir, paquet["vocab"])
    if list(xt.columns) != paquet["colonnes"]:
        raise ValueError("recette B : colonnes différentes de celles de l'entraînement")
    t = pd.DataFrame({CIBLE: predire_sacs(paquet["modeles"], xt.to_numpy(np.float32))},
                     index=pd.Index(gt, name="ut_ms")).reindex(a_predire)
    if out:
        t.to_csv(out)
    return t

