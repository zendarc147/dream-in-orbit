"""Recette complète « S » des variables (docs/journaux/JOURNAL_RECETTE_COMPLETE.md), distincte de prepare.py.

Tout est calculé à partir des tables brutes (dmop, ftl, evtf, saaf, ltdata) sur une grille horaire : une ligne
décrit [h, h + 1 h), h = début de l'heure en ms UTC. Aucune consommation n'est lue. Le vocabulaire (familles,
préfixes, familles à changements de variante, types FTL) est appris par apprendre() sur les seules périodes
d'entraînement, séparément dans chaque période puis réuni, et réutilisé tel quel à la validation et à la prédiction.

Conventions :
- SAAF : cos(angle) de chaque mesure avant agrégation ; mesures groupées par heure plancher ; moyennes arithmétiques
  (non pondérées par la durée) des 4 angles et des 4 cosinus ; écart-type de sa par pandas std() (ddof=1), mis à 0
  quand l'heure a des mesures mais un écart-type indéfini (une seule mesure) ; heures sans mesure interpolées
  linéairement sur les timestamps, bords remplis par la valeur disponible la plus proche (np.interp).
- LTDATA : valeurs journalières interpolées linéairement au début de l'heure, bords = valeur la plus proche.
- Familles : commandes.famille_variante ; codes contenant un point exclus du vocabulaire ; famille retenue si ≥ 50
  occurrences dans une période d'entraînement ; union des périodes.
- Récences (familles, préfixes) : occurrences quantifiées à leur heure ; lue au début de l'heure h, avant les
  occurrences de h : h − heure de la dernière occurrence d'une heure < h ; plafond 720 h ; log1p ; jamais vue :
  log1p(720).
- Changements de variante : familles ayant, dans une période d'entraînement, au moins deux variantes (codes exacts)
  émises chacune ≥ 50 fois ; union des périodes. Changement = commande de la famille dont le code diffère de la
  précédente (lettre inconnue comprise ; la première occurrence n'en est pas un), quantifié à l'heure. Depuis =
  h − dernière heure de changement ≤ h ; jusqu'à = prochaine heure de changement ≥ h − h (0 et 0 dans une heure de
  changement) ; plafond 720 h ; log1p ; aucun changement : log1p(720). L'identité de la variante n'est pas fournie.
- Paires début/fin (PENS/PENE…) appariées par numéro (min des débuts, max des fins par numéro), gardées si
  0 < durée ≤ 7 jours ; intervalles fusionnés (union) avant la couverture horaire.
- Événements evtf : paires START/END (contexte.intervalles : ouverture répétée ignorée, fermeture orpheline ignorée,
  paire > 6 h écartée) ; fractions horaires par union. Contacts : seuil d'élévation _10 seulement (contexte.contacts),
  les seuils différents ne sont pas fusionnés.
- FTL : types appris sur l'entraînement (+ ftl_autre) en fractions horaires par union, communications (flagcomms),
  nombre de débuts de pointage dans l'heure.
- Géométrie : contexte.geometrie (milieu de l'heure, péricentres exacts, une période médiane ajoutée aux bords).
- Altitude : passages à N km, 330 km aux péricentres, 10 100 km aux apocentres ; timestamps dupliqués = moyenne des
  log ; log de l'altitude interpolé linéairement (bords = valeur la plus proche) aux positions 1/8, 3/8, 5/8, 7/8 de
  l'heure ; moyenne des quatre log (sans exponentielle).
- Thermique : shade = clip(1 − umbra − 0.5 pénombre, 0, 1), solar = constante solaire / 600, flux ± par axe,
  flux_incident = solar × shade ; historiques (fenêtres passées incluant l'heure, min_periods=1) sur l'umbra seule,
  flux_incident et communications ; décalages t±1..3 de 10 séries, bords = valeur la plus proche.
Les fenêtres et décalages sont calculés par période continue (jamais à travers deux périodes).

Options (ablations) : saaf « pondere » (moyennes pondérées par la durée, contexte.soleil) ; ltdata « milieu » ;
histoires « r » (umbra + 0.5 pénombre et somme des flux des faces, formules du boosting R) ; altitude False (ni
altitude ni ses décalages) ; recence « exacte » (instants exacts en ms au lieu des heures quantifiées) ;
distance_terre True (garde distance Terre–Mars et angle Soleil–Mars–Terre dans le GBM).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import commandes, contexte, prepare

H = 3_600_000
RACINE = Path(__file__).resolve().parent.parent
ANGLES = ("sa", "sx", "sy", "sz")
MIN_VOCAB = 50
PLAFOND_H = 720
DUREE_MAX_PAIRE = 7 * 24 * H
PAIRES = [("PENS", "PENE"), ("MOCS", "MOCE"), ("PPNS", "PPNE"), ("PDNS", "PDNE"), ("UPBS", "UPBE"), ("UDBS", "UDBE")]
EVENEMENTS = ("MAR_UMBRA", "MAR_PENUMBRA", "OCC_MARS", "OCC_MARS_200KM", "OCC_PHOBOS", "OCC_DEIMOS", "PHO_UMBRA",
              "PHO_PENUMBRA", "DEI_UMBRA", "DEI_PENUMBRA")
DECALAGES = (-3, -2, -1, 1, 2, 3)
LTDATA_EXCLUES = ("ltdata_earthmars_km", "ltdata_sunmarsearthangle_deg")
OPTIONS_DEFAUT = {"saaf": "arith", "ltdata": "debut", "histoires": "spec", "altitude": True, "recence": "quantifiee",
                  "distance_terre": False}


def dossier_periode(nom):
    """annee1 -> data/annee1 ; sinon un chemin (ex. validation_chrono/entrainement)."""
    d = RACINE / "data" / nom
    return d if d.is_dir() else Path(nom)


def charger_periode(nom, circuits):
    """(heures, tables brutes, y avec NaN) d'une période d'entraînement : même grille que prepare.py."""
    dossier = dossier_periode(nom)
    y = prepare.cibles_horaires(dossier / "power.csv", circuits)
    return y.index.to_numpy(), prepare.lire_contextes(dossier), y.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------- vocabulaire

def _sans_point(dmop):
    return dmop.subsystem[~dmop.subsystem.str.contains(".", regex=False)]


def apprendre(periodes):
    """periodes : liste de dicts de tables brutes d'entraînement. Seuils appliqués par période, puis union."""
    familles, prefixes, changements, ftl = set(), set(), set(), set()
    for t in periodes:
        codes = _sans_point(t["dmop"])
        c = codes.map(commandes.famille_variante).value_counts()
        familles |= set(c[c >= MIN_VOCAB].index)
        c = t["dmop"].subsystem.str[:4].value_counts()
        prefixes |= set(c[c >= MIN_VOCAB].index)
        c = codes.value_counts()
        frequents = pd.Series(c[c >= MIN_VOCAB].index)
        par_famille = frequents.groupby(frequents.map(commandes.famille_variante)).size()
        changements |= set(par_famille[par_famille >= 2].index)
        ftl |= set(t["ftl"].type.unique())
    return {"familles": sorted(familles), "prefixes": sorted(prefixes), "changements": sorted(changements),
            "ftl": sorted(ftl)}


# ---------------------------------------------------------------- contexte

def _interp(heures, valeurs):
    """Remplir les NaN par interpolation linéaire sur les timestamps, bords = valeur la plus proche."""
    ok = np.isfinite(valeurs)
    if ok.all() or not ok.any():
        return np.nan_to_num(valeurs)
    return np.interp(heures, heures[ok], valeurs[ok])


def saaf(table, heures, mode="arith"):
    heures = np.asarray(heures, dtype=np.int64)
    n = len(heures)
    if mode == "pondere":
        s = contexte.soleil(table, heures).rename(columns={"saaf_sa_ecart": "saaf_sa_std"})
        observe = np.isfinite(s["saaf_sa_mean"].to_numpy())
    else:
        i = (table.ut_ms.to_numpy(dtype=np.int64) - heures[0]) // H
        dans = (i >= 0) & (i < n)
        m = table.loc[dans, list(ANGLES)].astype(np.float64)
        for a in ANGLES:
            m[f"{a}_cos"] = np.cos(np.deg2rad(m[a]))
        g = m.groupby(i[dans])
        moy = g.mean().reindex(range(n))
        s = pd.DataFrame({**{f"saaf_{a}_mean": moy[a].to_numpy() for a in ANGLES},
                          **{f"saaf_{a}_cos": moy[f"{a}_cos"].to_numpy() for a in ANGLES},
                          "saaf_sa_std": g["sa"].std().reindex(range(n)).to_numpy()}, index=heures)
        observe = g.size().reindex(range(n)).fillna(0).to_numpy() > 0
    s.loc[observe & s["saaf_sa_std"].isna().to_numpy(), "saaf_sa_std"] = 0.0
    return pd.DataFrame({c: _interp(heures.astype(np.float64), s[c].to_numpy(dtype=np.float64)) for c in s.columns},
                        index=heures)


def ltdata(table, heures, instant="debut"):
    t = np.asarray(heures, dtype=np.float64) + (H / 2 if instant == "milieu" else 0)
    jours = table.ut_ms.to_numpy(dtype=np.float64)
    return pd.DataFrame({f"ltdata_{c}": np.interp(t, jours, table[c].to_numpy(dtype=np.float64))
                         for c in table.columns if c != "ut_ms"}, index=heures)


def evenements(evtf, heures):
    c = contexte.codes(evtf)
    sortie = {}
    for nom in EVENEMENTS:
        lignes = c.str.fullmatch(f"{nom}_(?:START|END)")
        d, f = contexte.intervalles(evtf.ut_ms[lignes].to_numpy(), c[lignes].str.rsplit("_", n=1).str[1].to_numpy(),
                                    "START", "END", contexte.DUREE_MAX_OMBRE_MS)
        sortie[f"ev_{nom.lower()}"] = prepare.durees_union(heures, d, f) / 60
    return pd.concat([pd.DataFrame(sortie, index=heures), contexte.contacts(evtf, heures)], axis=1)


def pointage(ftl, heures, types):
    sortie = {f"ftl_{t}": prepare.durees_union(heures, ftl.utb_ms[ftl.type == t], ftl.ute_ms[ftl.type == t]) / 60
              for t in types}
    autres = ftl[~ftl.type.isin(types)]
    sortie["ftl_autre"] = prepare.durees_union(heures, autres.utb_ms, autres.ute_ms) / 60
    comms = ftl[ftl.flagcomms.eq(True)]
    sortie["ftl_comms"] = prepare.durees_union(heures, comms.utb_ms, comms.ute_ms) / 60
    i = (ftl.utb_ms.to_numpy(dtype=np.int64) - heures[0]) // H
    dans = (i >= 0) & (i < len(heures))
    sortie["ftl_debuts"] = np.bincount(i[dans], minlength=len(heures)).astype(np.float64)
    return pd.DataFrame(sortie, index=heures)


def log_altitude(evtf, heures):
    c = contexte.codes(evtf)
    alt = c.str.extract(r"^(\d+)_KM_(?:ASCEND|DESCEND)$", expand=False).astype(float)
    alt[c == "PERICENTRE_PASSAGE"] = contexte.ALTITUDE_PERICENTRE_KM
    alt[c == "APOCENTRE_PASSAGE"] = contexte.ALTITUDE_APOCENTRE_KM
    ok = alt.notna().to_numpy()
    points = pd.Series(np.log(alt[ok].to_numpy()), index=evtf.ut_ms[ok].to_numpy()).groupby(level=0).mean()
    sous = heures[:, None] + (np.arange(4) + 0.5) * H / 4  # 1/8, 3/8, 5/8, 7/8
    log = np.interp(sous.ravel(), points.index.to_numpy(dtype=np.float64), points.to_numpy()).reshape(sous.shape)
    return log.mean(axis=1)


# ---------------------------------------------------------------- commandes

def _plafond_log(h):
    return np.log1p(np.minimum(h, PLAFOND_H))


def recence(temps, heures, convention="quantifiee"):
    """Heures depuis la dernière occurrence strictement avant le début de l'heure (temps triés)."""
    t = np.sort(np.asarray(temps, dtype=np.int64))
    if convention == "quantifiee":
        t = t // H * H
    k = np.searchsorted(t, heures, side="left") - 1
    sortie = np.full(len(heures), np.inf)
    sortie[k >= 0] = (heures[k >= 0] - t[k[k >= 0]]) / H
    return _plafond_log(sortie)


def distances_changement(instants, heures, convention="quantifiee"):
    """(depuis, jusqu'à) le changement le plus proche ; quantifiés : 0 et 0 dans une heure de changement."""
    t = np.sort(np.asarray(instants, dtype=np.int64))
    if convention == "quantifiee":
        t = np.unique(t // H * H)
        k = np.searchsorted(t, heures, side="right") - 1
    else:  # instants exacts lus au début de l'heure (convention du boosting R)
        k = np.searchsorted(t, heures, side="left") - 1
    depuis = np.full(len(heures), np.inf)
    depuis[k >= 0] = (heures[k >= 0] - t[k[k >= 0]]) / H
    j = np.searchsorted(t, heures, side="left")
    jusqua = np.full(len(heures), np.inf)
    jusqua[j < len(t)] = (t[j[j < len(t)]] - heures[j < len(t)]) / H
    return _plafond_log(depuis), _plafond_log(jusqua)


def periode_paire(dmop, heures, debut, fin):
    numero = dmop.subsystem.str.extract(r"^([A-Z]{4})\.(\d+)$")
    d = dmop.ut_ms[numero[0] == debut].groupby(numero.loc[numero[0] == debut, 1]).min()
    f = dmop.ut_ms[numero[0] == fin].groupby(numero.loc[numero[0] == fin, 1]).max()
    p = pd.concat([d, f], axis=1, join="inner").to_numpy(dtype=np.int64).reshape(-1, 2)
    p = p[(p[:, 1] > p[:, 0]) & (p[:, 1] - p[:, 0] <= DUREE_MAX_PAIRE)]
    return prepare.durees_union(heures, p[:, 0], p[:, 1]) / 60


def commandes_horaires(dmop, heures, vocab, convention="quantifiee"):
    n = len(heures)
    temps = dmop.ut_ms.to_numpy(dtype=np.int64)
    familles = dmop.subsystem.map(commandes.famille_variante).to_numpy()
    prefixes = dmop.subsystem.str[:4].to_numpy()
    i = (temps - heures[0]) // H
    dans = (i >= 0) & (i < n)
    sortie = {}
    comptes_fam = {}
    for f in vocab["familles"]:
        m = familles == f
        comptes_fam[f] = np.bincount(i[m & dans], minlength=n).astype(np.float64)
        sortie[f"cmd_fam_{f}_n"] = comptes_fam[f]
    for p in vocab["prefixes"]:
        sortie[f"cmd_pre_{p}_n"] = np.bincount(i[(prefixes == p) & dans], minlength=n).astype(np.float64)
    sortie["cmd_total_n"] = np.bincount(i[dans], minlength=n).astype(np.float64)
    for f in vocab["familles"]:
        sortie[f"cmd_fam_{f}_recence"] = recence(temps[familles == f], heures, convention)
    for p in vocab["prefixes"]:
        sortie[f"cmd_pre_{p}_recence"] = recence(temps[prefixes == p], heures, convention)
    for f in vocab["familles"]:
        cumul = np.r_[0.0, np.cumsum(comptes_fam[f])]
        t = np.arange(n)
        sortie[f"cmd_fam_{f}_futur1h"] = cumul[np.minimum(t + 2, n)] - cumul[np.minimum(t + 1, n)]
        sortie[f"cmd_fam_{f}_futur6h"] = cumul[np.minimum(t + 7, n)] - cumul[np.minimum(t + 1, n)]
    codes = dmop.subsystem.to_numpy()
    for f in vocab["changements"]:
        m = familles == f
        c, t = codes[m], temps[m]
        instants = t[1:][c[1:] != c[:-1]]
        sortie[f"cmd_chg_{f}_depuis"], sortie[f"cmd_chg_{f}_jusqua"] = distances_changement(instants, heures, convention)
    for d, f in PAIRES:
        sortie[f"cmd_paire_{d}_{f}"] = periode_paire(dmop, heures, d, f)
    return pd.DataFrame(sortie, index=heures)


# ---------------------------------------------------------------- thermique

def _fenetre(serie, w, moyenne):
    r = pd.Series(serie).rolling(w, min_periods=1)
    return (r.mean() if moyenne else r.sum()).to_numpy()


def thermique(t, histoires="spec"):
    """t : table avec ev_mar_umbra, ev_mar_penumbra, saaf_*_cos, ltdata_solarconstantmars, ftl_comms."""
    u, p = t["ev_mar_umbra"].to_numpy(), t["ev_mar_penumbra"].to_numpy()
    shade = np.clip(1 - u - 0.5 * p, 0, 1)
    solar = t["ltdata_solarconstantmars"].to_numpy() / 600
    sortie = {}
    for a in ANGLES:
        c = t[f"saaf_{a}_cos"].to_numpy()
        sortie[f"flux_{a}_plus"] = solar * np.maximum(c, 0) * shade
        sortie[f"flux_{a}_moins"] = solar * np.maximum(-c, 0) * shade
    sortie["flux_incident"] = solar * shade
    if histoires == "r":  # formules du boosting R : ombre = umbra + 0.5 pénombre, somme des flux des faces
        ombre = np.clip(u + 0.5 * p, 0, 1)
        flux = sum(sortie[f"flux_{a}_{s}"] for a in ANGLES for s in ("plus", "moins"))
    else:
        ombre, flux = u, sortie["flux_incident"]
    n = len(u)
    k = np.arange(n, dtype=float)
    derniere = np.maximum.accumulate(np.where(ombre > 0, k, -np.inf))
    sortie.update({
        "th_umbra_somme7h": _fenetre(ombre, 7, False), "th_umbra_somme24h": _fenetre(ombre, 24, False),
        "th_umbra_moy168h": _fenetre(ombre, 168, True), "th_umbra_depuis": _plafond_log(k - derniere),
        "th_incident_moy6h": _fenetre(flux, 6, True), "th_incident_moy24h": _fenetre(flux, 24, True),
        "th_comms_moy24h": _fenetre(t["ftl_comms"].to_numpy(), 24, True)})
    return pd.DataFrame(sortie, index=t.index)


def decalages(t, series):
    n = len(t)
    return pd.DataFrame({f"{s}_dec{d:+d}h": t[s].to_numpy()[np.clip(np.arange(n) + d, 0, n - 1)]
                         for s in series for d in DECALAGES}, index=t.index)


# ---------------------------------------------------------------- assemblage

def construire(tables, heures, vocab, options=None):
    """Toutes les variables d'une période continue (heures : grille complète, tables : tables brutes)."""
    o = {**OPTIONS_DEFAUT, **(options or {})}
    heures = np.asarray(heures, dtype=np.int64)
    blocs = [saaf(tables["saaf"], heures, o["saaf"]), ltdata(tables["ltdata"], heures, o["ltdata"]),
             evenements(tables["evtf"], heures), pointage(tables["ftl"], heures, vocab["ftl"]),
             contexte.geometrie(tables["evtf"], heures).drop(columns="geo_altitude_km")]
    t = pd.concat(blocs, axis=1)
    if o["altitude"]:
        t["geo_log_altitude"] = log_altitude(tables["evtf"], heures)
    t = pd.concat([t, commandes_horaires(tables["dmop"], heures, vocab, o["recence"]), thermique(t, o["histoires"])],
                  axis=1)
    series = ["ev_mar_umbra", "ev_mar_penumbra", *[f"saaf_{a}_cos" for a in ANGLES], "ftl_comms"]
    for mode in ("EARTH", "SLEW"):
        if f"ftl_{mode}" not in t:
            t[f"ftl_{mode}"] = 0.0
        series.append(f"ftl_{mode}")
    if o["altitude"]:
        series.append("geo_log_altitude")
    t = pd.concat([t, decalages(t, series)], axis=1)
    if not np.isfinite(t.to_numpy(dtype=np.float64)).all():
        raise ValueError("Variables non finies : " + ", ".join(t.columns[~np.isfinite(t.to_numpy(float)).all(axis=0)]))
    return t


GRU_EXCLUS = re.compile(r"cmd_(pre_.*|total_n|fam_.*_(recence|futur\d+h)|chg_.*|paire_.*)|flux_.*|th_.*|.*_dec[+-]\dh")


def colonnes_gru(table):
    """SAAF (4 angles, 4 cos, std de sa), LTDATA sans distance Terre ni angle SMT, événements, pointage, géométrie,
    altitude, comptes horaires par famille. Ni lettres, ni récences, ni changements, ni comptes futurs, ni flux,
    ni historiques, ni décalages."""
    return [c for c in table.columns if c not in LTDATA_EXCLUES and not GRU_EXCLUS.fullmatch(c)]


def colonnes_gbm(table, options=None):
    o = {**OPTIONS_DEFAUT, **(options or {})}
    return [c for c in table.columns if o["distance_terre"] or c not in LTDATA_EXCLUES]
