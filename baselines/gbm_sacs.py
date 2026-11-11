"""Gradient boosting par circuit avec bagging temporel : composant classique d'un mélange.

Un HistGradientBoostingRegressor par circuit et par sac. Le sac k apprend sans un bloc
contigu (un cinquième de la période, avec une marge de 7 jours de chaque côté) ; la
prédiction est la moyenne des sacs. Les prédictions de chaque sac sur son bloc écarté
(hors sac) sont gardées pour un empilement éventuel.

Variables, toutes tirées du contexte (jamais d'une consommation) :
- v1 : les 65 variables de prepare.py ;
- V1 : v1 + variables v2r (fenêtres passées et futures du planning, variables.py) ;
- V2 : V1 + flux solaire par face, part hors ombre et leurs moyennes passées (inertie
  thermique), comptes horaires par famille de commande dmop sans lettre de variante.
Ajouts optionnels (ablations de la phase 2, docs/JOURNAL_ENSEMBLE.md) :
- planning : par famille de commande (lettres de variante fusionnées, préfixes « XXXX.n »
  regroupés), heures depuis la dernière commande, comptes dans l'heure suivante et dans les
  six suivantes ; part active de chaque période début/fin appariable (ex. MOCS.n … MOCE.n).
  Vocabulaire appris sur l'entraînement seul ;
- historique : moyennes passées 6/24/72/168 h du flux solaire, de l'ombre, des modes de
  pointage ftl et des communications ;
- ema : moyennes exponentielles (constantes de temps 3/12/48/168 h) des mêmes séries ;
- orbite : harmoniques 2 à 4 de la phase orbitale et moyennes exponentielles longues
  (336 et 720 h) du flux éclairé par face et de l'ombre (inertie thermique lente) ;
- regimes : par famille du planning, comptes dans les fenêtres centrées de ±24 h et ±168 h
  (le planning est connu d'avance : marqueur d'un mode, ex. ATTTF410 pendant le palier 0.1
  de NPWD2531) ;
- sans_horloge : retire les distances et angles ltdata, lents et quasi monotones sur une
  année, qui servent d'horloge indirecte.
Options des spécialistes (docs/JOURNAL_SPECIALISTES.md) : circuits (n'apprendre que
certaines colonnes) et lissage_cible (moyenne glissante centrée de la cible, à l'entraînement
seulement : débruite le cycle des thermostats, la prédiction reste horaire).
Les variables quasi constantes sur l'entraînement sont écartées (même règle que monde.py).
power_init ne sert qu'au recalage du niveau, à la prédiction (boosting.recaler).
"""
import lzma
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from baselines.boosting import recaler
from dio import commandes, contexte as contexte_brut, donnees, monde, prepare, variables

H = donnees.H
HORLOGE = ["ltdata_sunmars_km", "ltdata_earthmars_km", "ltdata_sunmarsearthangle_deg", "ltdata_solarconstantmars"]
FENETRES_HISTOIRE = (6, 24, 72, 168)
CONSTANTES_EMA = (3, 12, 48, 168)
CONSTANTES_LENTES = (336, 720)
DEMI_FENETRES_REGIME = (24, 168)
HARMONIQUES_ORBITE = (2, 3, 4)
PLAFOND_RECENCE_H = 168
# Comptes horaires des commandes exactes de prepare.py (lettre de variante comprise, ex. dmop_AAAAF20C1_count).
LETTRES = [r"dmop_(?!other_count|total_count)\w+_count"]
# Noms utilisables dans retirer (évite d'écrire les expressions dans le JSON des options).
RETRAITS = {"lettres": LETTRES, "distance_terre": ["ltdata_earthmars_km", "ltdata_sunmarsearthangle_deg"],
            "ema": [r".*_ema\d+h"]}
AJOUTS_CONTEXTE = {"geometrie", "altitude", "ombres", "contacts"}
DECALAGES = (-3, -2, -1, 1, 2, 3)
ORBITE_HORAIRE = ["orbite_pericentre_depuis_h", "orbite_pericentre_jusqua_h", "orbite_phase_cos", "orbite_phase_sin"]
PLAFOND_LONG_H = 720
DUREE_MAX_PERIODE = 7 * 24 * H
PAIRES_LISTE = [("PENS", "PENE"), ("MOCS", "MOCE"), ("PPNS", "PPNE"), ("PDNS", "PDNE"), ("UPBS", "UPBE"),
                ("UDBS", "UDBE")]

REGLAGES = {
    # A : réglage retenu de la baseline (E1) ; B : le plus gros réglage encore correct d'E1.
    "A": {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 500,
          "l2_regularization": 1.0},
    "B": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
          "l2_regularization": 1.0},
    # C : plus lent et plus régularisé (candidat de la phase 2).
    "C": {"max_iter": 600, "learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 100,
          "l2_regularization": 3.0},
    # D : C avec l'arrêt anticipé (10 % de validation interne, 10 itérations sans gain) ; chaque circuit
    # garde entre 10 et 600 arbres (docs/JOURNAL_RECETTE.md, E18).
    "D": {"max_iter": 600, "learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 100,
          "l2_regularization": 3.0, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
          "tol": 1e-7},
}
# ET : arbres extrêmement aléatoires (comparaison de diversité, sans sacs : déjà aléatoires).
EXTRA_TREES = {"n_estimators": 100, "min_samples_leaf": 50, "max_features": 0.3}

def famille_etendue(code, variante=False):
    """Famille d'une commande : lettres de variante fusionnées ; MOCS.0000003738 -> MOCS.
    variante : commandes.famille_variante au lieu de commandes.famille (chiffres après la lettre gardés)."""
    m = re.match(r"^([A-Z]{4})\.\d+$", code)
    return m.group(1) if m else (commandes.famille_variante if variante else commandes.famille)(code)


def apprendre_planning(tables, variante=False, recence="fin", prefixes=False, changements=False, paires="apprises"):
    """Familles vues au moins MIN_FAMILLE fois et paires début/fin (XXXS / XXXE) appariables.
    variante : familles famille_etendue(variante=True) vues au moins MIN_FAMILLE_VARIANTE fois dans une
    des périodes, union des périodes. Options (docs/JOURNAL_RECETTE.md) :
    - recence « debut » : heures depuis la dernière commande au début de l'heure, avant les commandes de
      l'heure, plafond 720 h, log1p (« fin » : à la fin de l'heure, plafond 168 h) ;
    - prefixes : comptes horaires et récence par préfixe de 4 caractères (≥ 50 fois dans une période) ;
    - changements : familles ayant au moins deux variantes vues MIN_FAMILLE fois, pour les heures
      depuis et jusqu'au changement de variante ;
    - paires « liste » : périodes début/fin de PAIRES_LISTE au lieu des paires apprises."""
    options = {"recence": recence} if recence != "fin" else {}
    if prefixes:
        comptes = [t.subsystem.str[:4].value_counts() for t in tables]
        options["prefixes"] = sorted(set().union(*(c[c >= variables.MIN_FAMILLE_VARIANTE].index for c in comptes)))
    if changements:
        codes = pd.concat(tables).subsystem.value_counts()
        frequents = pd.Series(codes[codes >= variables.MIN_FAMILLE].index)
        par_famille = frequents.groupby(frequents.map(lambda c: famille_etendue(c, variante))).size()
        options["changements"] = sorted(par_famille[par_famille >= 2].index)
    if variante:
        comptes = [t.subsystem.map(lambda c: famille_etendue(c, True)).value_counts() for t in tables]
        familles = sorted(set().union(*(c[c >= variables.MIN_FAMILLE_VARIANTE].index for c in comptes)))
    else:
        comptes = pd.concat(tables).subsystem.map(famille_etendue).value_counts()
        familles = sorted(comptes[comptes >= variables.MIN_FAMILLE].index)
    paires = (PAIRES_LISTE if paires == "liste" else
              [(f, f[:3] + "E") for f in familles if re.fullmatch(r"[A-Z]{3}S", f) and f[:3] + "E" in familles])
    return {"familles": familles, "paires": paires, **({"variante": True} if variante else {}), **options}


def heures_depuis(temps, heures, convention="fin"):
    """Heures depuis le dernier instant de temps (trié). « fin » : à la fin de l'heure, plafond 168 h ;
    « debut » : au début de l'heure, avant ses commandes, log1p(min(·, 720)). Jamais vu : plafond."""
    debut = convention == "debut"
    reference = heures if debut else heures + H
    position = np.searchsorted(temps, reference, side="left") - 1
    sortie = np.full(len(heures), np.inf)
    vue = position >= 0
    sortie[vue] = (reference[vue] - temps[position[vue]]) / H
    return np.log1p(np.minimum(sortie, PLAFOND_LONG_H)) if debut else np.minimum(sortie, PLAFOND_RECENCE_H)


def changements_variante(dmop, heures, famille, variante):
    """log1p(min(heures, 720)) depuis le dernier et jusqu'au prochain changement de variante de la famille,
    au début de l'heure. Un changement : une commande dont le code diffère de la précédente de la famille
    (A, A, C, C, B : A→C et C→B) ; une lettre jamais vue à l'entraînement compte aussi. Pas d'identité."""
    lignes = dmop[dmop.subsystem.map(lambda c: famille_etendue(c, variante)) == famille]
    codes, temps = lignes.subsystem.to_numpy(), lignes.ut_ms.to_numpy(dtype=np.int64)
    instants = temps[1:][codes[1:] != codes[:-1]]
    depuis = heures_depuis(instants, heures, "debut")
    suivant = np.searchsorted(instants, heures, side="left")
    jusqua = np.full(len(heures), np.inf)
    connu = suivant < len(instants)
    jusqua[connu] = (instants[suivant[connu]] - heures[connu]) / H
    return depuis, np.log1p(np.minimum(jusqua, PLAFOND_LONG_H))


def familles_planning(dmop, planning):
    """Table dmop dont les codes sont remplacés par leur famille du planning."""
    variante = planning.get("variante", False)
    return dmop.assign(subsystem=dmop.subsystem.map(lambda c: famille_etendue(c, variante)))


def periodes_actives(dmop, heures, debut, fin):
    """Part de chaque heure couverte par un intervalle debut.n … fin.n (même numéro)."""
    numero = dmop.subsystem.str.extract(r"^([A-Z]{4})\.(\d+)$")
    debuts = dmop[numero[0] == debut].assign(n=numero[1]).groupby("n").ut_ms.min()
    fins = dmop[numero[0] == fin].assign(n=numero[1]).groupby("n").ut_ms.max()
    paires = pd.concat([debuts, fins], axis=1, join="inner").to_numpy(dtype=np.int64)
    sortie = np.zeros(len(heures))
    if len(paires):
        paires = paires[(paires[:, 1] > paires[:, 0]) & (paires[:, 1] - paires[:, 0] <= DUREE_MAX_PERIODE)]
        i, _, duree = prepare.decouper_intervalles(np.asarray(heures, dtype=np.int64), paires[:, 0], paires[:, 1])
        np.add.at(sortie, i, duree / H)
    return np.minimum(sortie, 1.0)


def variables_planning(dmop, heures, planning):
    heures = np.asarray(heures, dtype=np.int64)
    table = familles_planning(dmop, planning)
    familles = planning["familles"]
    comptes = commandes.impulsions(table, heures, familles)
    cumul = np.vstack([np.zeros((1, len(familles))), np.cumsum(comptes, axis=0)])
    t, n = np.arange(len(heures)), len(heures)
    sortie = {}
    convention = planning.get("recence", "fin")
    for j, f in enumerate(familles):
        temps = np.sort(table.ut_ms[table.subsystem == f].to_numpy(dtype=np.int64))
        sortie[f"plan_{f}_depuis_h"] = heures_depuis(temps, heures, convention)  # famille jamais vue : plafond
        sortie[f"plan_{f}_futur1h"] = cumul[np.minimum(t + 2, n), j] - cumul[np.minimum(t + 1, n), j]
        sortie[f"plan_{f}_futur6h"] = cumul[np.minimum(t + 7, n), j] - cumul[np.minimum(t + 1, n), j]
    for debut, fin in planning["paires"]:
        sortie[f"plan_{debut}_{fin}_actif"] = periodes_actives(dmop, heures, debut, fin)
    prefixes = planning.get("prefixes", [])
    if prefixes:
        quatre = dmop.assign(subsystem=dmop.subsystem.str[:4])
        comptes = commandes.impulsions(quatre, heures, prefixes)
        for j, p in enumerate(prefixes):
            sortie[f"plan_pre_{p}_count"] = comptes[:, j]
            temps = np.sort(quatre.ut_ms[quatre.subsystem == p].to_numpy(dtype=np.int64))
            sortie[f"plan_pre_{p}_depuis_h"] = heures_depuis(temps, heures, convention)
    for f in planning.get("changements", []):
        sortie[f"plan_{f}_depuis_chgt_h"], sortie[f"plan_{f}_jusqua_chgt_h"] = changements_variante(
            dmop, heures, f, planning.get("variante", False))
    return pd.DataFrame(sortie, index=heures)


def variables_regimes(dmop, heures, planning):
    """Comptes de chaque famille du planning dans [t - w, t + w] (heures entières)."""
    heures = np.asarray(heures, dtype=np.int64)
    table = familles_planning(dmop, planning)
    comptes = commandes.impulsions(table, heures, planning["familles"])
    cumul = np.vstack([np.zeros((1, comptes.shape[1])), np.cumsum(comptes, axis=0)])
    t, n = np.arange(len(heures)), len(heures)
    sortie = {}
    for w in DEMI_FENETRES_REGIME:
        fenetre = cumul[np.minimum(t + w + 1, n)] - cumul[np.maximum(t - w, 0)]
        sortie.update({f"plan_{f}_pm{w}h": fenetre[:, j] for j, f in enumerate(planning["familles"])})
    return pd.DataFrame(sortie, index=heures)


def series_contexte(x):
    """Flux solaire éclairé par face, ombre, modes de pointage ftl et communications."""
    physique = variables.physiques(x)
    series = {c: physique[c].to_numpy() for c in physique.columns if c == "ombre" or c.endswith("_eclaire")}
    for c in x.columns:
        if c.startswith("ftl_") and c not in ("ftl_coverage_minutes", "ftl_overlap_minutes"):
            series[c] = x[c].to_numpy(dtype=np.float64)
    return series


def decalages(x, table):
    """Valeurs à t−3, t−2, t−1, t+1, t+2, t+3 h (bords répétés) de l'ombre, de la pénombre (fractions
    d'heure), des quatre cosinus solaires, des communications et des modes EARTH et SLEW."""
    series = {"ombre_umbra": x["evtf_MAR_UMBRA_minutes"] / 60, "ombre_penumbra": x["evtf_MAR_PENUMBRA_minutes"] / 60,
              **{f"saaf_{a}_cos": table[f"saaf_{a}_cos"] for a in contexte_brut.ANGLES},
              "ftl_comms_minutes": x["ftl_comms_minutes"], "ftl_EARTH_minutes": x["ftl_EARTH_minutes"],
              "ftl_SLEW_minutes": x["ftl_SLEW_minutes"]}
    n = len(x)
    sortie = {f"{nom}_dec{d:+d}h": serie.to_numpy(dtype=np.float64)[np.clip(np.arange(n) + d, 0, n - 1)]
              for nom, serie in series.items() for d in DECALAGES}
    return pd.DataFrame(sortie, index=x.index)


def face_sa_moins(x, table):
    """Flux de la face sa− : constante solaire × max(−cos sa, 0), brut et éclairé (même ombre que physiques)."""
    soleil = x["ltdata_solarconstantmars"].to_numpy(dtype=np.float64)
    ombre = np.clip((x["evtf_MAR_UMBRA_minutes"] + 0.5 * x["evtf_MAR_PENUMBRA_minutes"]).to_numpy() / 60, 0, 1)
    flux = soleil * np.maximum(-table["saaf_sa_cos"].to_numpy(dtype=np.float64), 0)
    return pd.DataFrame({"flux_sa_moins": flux, "flux_sa_moins_eclaire": flux * (1 - ombre)}, index=x.index)


def histoires_ciblees(x):
    """Ombre (umbra + 0.5 pénombre, fraction d'heure) cumulée sur 7 et 24 h, moyenne sur 168 h, heures
    depuis la dernière heure d'ombre (log1p, plafond 720 h) ; flux éclairé total moyen sur 6 et 24 h ;
    communications moyennes sur 24 h. Fenêtres passées, heure courante comprise."""
    physique = variables.physiques(x)
    ombre = physique["ombre"].to_numpy()
    flux = physique[[c for c in physique.columns if c.startswith("flux_") and c.endswith("_eclaire")]].sum(axis=1)
    comms = x["ftl_comms_minutes"].to_numpy(dtype=np.float64)
    t = np.arange(len(x), dtype=float)
    derniere = np.maximum.accumulate(np.where(ombre > 0, t, -np.inf))
    sortie = {"ombre_cumul7h": variables.sommes_glissantes(ombre, 7)[0] * np.minimum(t + 1, 7),
              "ombre_cumul24h": variables.sommes_glissantes(ombre, 24)[0] * np.minimum(t + 1, 24),
              "ombre_moy168h": variables.sommes_glissantes(ombre, 168)[0],
              "ombre_depuis_h": np.log1p(np.minimum(t - derniere, PLAFOND_LONG_H)),
              "flux_total_moy6h": variables.sommes_glissantes(flux.to_numpy(), 6)[0],
              "flux_total_moy24h": variables.sommes_glissantes(flux.to_numpy(), 24)[0],
              "ftl_comms_moy24h": variables.sommes_glissantes(comms, 24)[0]}
    return pd.DataFrame(sortie, index=x.index)


def historique(x, existantes):
    """Moyennes passées des séries de contexte, sans les colonnes déjà présentes (même nom)."""
    sortie = {}
    for nom, serie in series_contexte(x).items():
        for w in FENETRES_HISTOIRE:
            if f"{nom}_passe{w}h" not in existantes:
                sortie[f"{nom}_passe{w}h"] = variables.sommes_glissantes(serie, w)[0]
    return pd.DataFrame(sortie, index=x.index)


def moyennes_exponentielles(x, constantes=CONSTANTES_EMA, filtre=None):
    """y_t = a x_t + (1 - a) y_(t-1), a = 1 - exp(-1/tau), y_0 = x_0 : causale."""
    sortie = {}
    for nom, serie in series_contexte(x).items():
        if filtre and not filtre(nom):
            continue
        for tau in constantes:
            a = 1 - np.exp(-1 / tau)
            sortie[f"{nom}_ema{tau}h"] = lfilter([a], [1, a - 1], serie, zi=[(1 - a) * serie[0]])[0]
    return pd.DataFrame(sortie, index=x.index)


def orbite(x, table):
    """Harmoniques de la phase orbitale (péricentre = 0) et inertie thermique lente par face."""
    phase = np.arctan2(table["orbite_phase_sin"].to_numpy(), table["orbite_phase_cos"].to_numpy())
    sortie = {}
    for k in HARMONIQUES_ORBITE:
        sortie[f"orbite_phase_cos{k}"], sortie[f"orbite_phase_sin{k}"] = np.cos(k * phase), np.sin(k * phase)
    lentes = moyennes_exponentielles(x, CONSTANTES_LENTES, lambda nom: nom == "ombre" or nom.endswith("_eclaire"))
    return pd.concat([pd.DataFrame(sortie, index=x.index), lentes], axis=1)


def preparer_contexte(x, contextes, contexte):
    """X de prepare.py modifié par les options de contexte, avant le calcul des variables :
    - ltdata « lineaire » : valeurs journalières interpolées au milieu de l'heure (au lieu du jour répété) ;
    - soleil « mesures » : saaf_*_cos = moyenne horaire des cosinus des mesures (au lieu du cosinus de
      l'angle moyen), saaf_sa_ecart ; heure sans mesure : cosinus de l'angle moyen rempli par prepare.py ;
    - soleil « interpole » : idem, mais angles, cosinus et écart-type des heures sans mesure interpolés
      dans le temps (bords : valeur la plus proche) au lieu des valeurs de remplissage de prepare.py."""
    x = x.copy()
    if contexte.get("ltdata") == "lineaire":
        lt = contexte_brut.ltdata_lineaire(contextes["ltdata"], x.index)
        x[lt.columns] = lt.to_numpy()
    mode = contexte.get("soleil")
    if mode:
        s = contexte_brut.soleil(contextes["saaf"], x.index)
        if mode == "interpole":
            s = contexte_brut.interpoler(s)
            colonnes = list(s.columns)
        else:
            for a in contexte_brut.ANGLES:
                s[f"saaf_{a}_cos"] = s[f"saaf_{a}_cos"].fillna(pd.Series(
                    np.cos(np.deg2rad(x[f"saaf_{a}_mean"].to_numpy())), index=s.index))
            s["saaf_sa_ecart"] = s["saaf_sa_ecart"].fillna(0.0)
            colonnes = [c for c in s.columns if not c.endswith("_mean")]
        x[colonnes] = s[colonnes].to_numpy()
    return x


def construire(x, dmop, jeu, familles=(), ajouts=(), planning=None, sans_horloge=False, variante=False,
               retirer=(), contextes=None, contexte=None):
    """Variables d'une période continue (x : X horaire de prepare.py, dmop : table brute).
    retirer : expressions régulières des colonnes à écarter (ablations de retrait) ; contextes : tables
    brutes de la période (saaf, ltdata, evtf), lues par les options contexte (preparer_contexte) et les
    ajouts geometrie (péricentres exacts, remplacent ORBITE_HORAIRE), altitude, ombres et contacts."""
    if jeu == "v1":
        return x.copy()
    if jeu not in ("V1", "V2"):
        raise ValueError(f"Jeu de variables inconnu : {jeu}")
    if contexte:
        x = preparer_contexte(x, contextes, contexte)
    inconnus = (set(ajouts) - {"planning", "historique", "ema", "orbite", "regimes", "decalages", "histoires",
                               "face_sa_moins"}
                - AJOUTS_CONTEXTE)
    if inconnus:
        raise ValueError(f"Ajouts inconnus : {sorted(inconnus)}")
    table = variables.ajouter_v2r(x)
    if jeu == "V2":
        table = pd.concat([table, variables.physiques(x),
                           variables.comptes_familles(dmop, x.index.to_numpy(), familles, variante)], axis=1)
    blocs = [table]
    if "planning" in ajouts:
        blocs.append(variables_planning(dmop, x.index.to_numpy(), planning).set_axis(x.index))
    if "historique" in ajouts:
        blocs.append(historique(x, set(table.columns)))
    if "ema" in ajouts:
        blocs.append(moyennes_exponentielles(x))
    if "orbite" in ajouts:
        blocs.append(orbite(x, table))
    if "regimes" in ajouts:
        blocs.append(variables_regimes(dmop, x.index.to_numpy(), planning).set_axis(x.index))
    if {"geometrie", "altitude"} & set(ajouts):
        geo = contexte_brut.geometrie(contextes["evtf"], x.index.to_numpy()).set_axis(x.index)
        if "geometrie" in ajouts:
            blocs[0] = blocs[0].drop(columns=ORBITE_HORAIRE)
            blocs.append(geo.drop(columns="geo_altitude_km"))
        if "altitude" in ajouts:
            blocs.append(geo[["geo_altitude_km"]])
    if "face_sa_moins" in ajouts:  # E16 : les flux V2 n'ont que la face sa+ (sx−, sy−, sz− existent)
        blocs.append(face_sa_moins(x, blocs[0]))
    if "decalages" in ajouts:
        blocs.append(decalages(x, blocs[0]))
    if "histoires" in ajouts:
        blocs.append(histoires_ciblees(x))
    if "ombres" in ajouts:
        blocs.append(contexte_brut.ombres(contextes["evtf"], x.index.to_numpy()).set_axis(x.index))
    if "contacts" in ajouts:
        blocs.append(contexte_brut.contacts(contextes["evtf"], x.index.to_numpy()).set_axis(x.index))
    table = pd.concat(blocs, axis=1)
    if sans_horloge:
        table = table.drop(columns=HORLOGE)
    if retirer:
        motifs = [m for r in retirer for m in RETRAITS.get(r, [r])]
        table = table.loc[:, ~table.columns.str.fullmatch("|".join(motifs))]
    return table


def sacs_temporels(n, sacs, embargo):
    """(indices d'entraînement, bloc écarté) de chaque sac ; un seul sac = toute la période."""
    t = np.arange(n)
    if sacs == 1:
        return [(t, t[:0])]
    bornes = np.linspace(0, n, sacs + 1).astype(int)
    return [(t[(t < a - embargo) | (t >= b + embargo)], t[a:b]) for a, b in zip(bornes[:-1], bornes[1:])]


def _ajuster(f, y, lignes, config, graine, perte="squared_error"):
    """Un circuit, un sac ; un seul thread par processus (les données sont petites)."""
    y = y[lignes]
    connus = np.isfinite(y)  # cibles absentes retirées, jamais remplacées
    with threadpool_limits(1):
        if config == "ET":
            modele = ExtraTreesRegressor(**EXTRA_TREES, random_state=graine, n_jobs=1)
        else:
            modele = HistGradientBoostingRegressor(**{"early_stopping": False, **REGLAGES[config]}, loss=perte,
                                                   random_state=graine)
        return modele.fit(f[lignes][connus], y[connus])


def entrainer(annees, dossier, config="A", jeu="V1", sacs=5, embargo_h=168, filtre=True, graine=0,
              ajouts=(), sans_horloge=False, compresser=False, circuits=None, lissage_cible=0,
              perte="squared_error", familles_variante=False, retirer=(), plan=None, contexte=None):
    """compresser : gbm_sacs.pkl.xz (lzma, environ 8 fois plus petit, 2 min de plus à l'écriture).
    circuits : colonnes apprises (toutes par défaut) ; lissage_cible : fenêtre en heures de la moyenne
    centrée appliquée aux cibles de chaque année (0 : cibles horaires brutes) ; perte : « absolute_error »
    apprend la médiane conditionnelle (un mode rare et inexpliqué ne tire plus le niveau de base) ;
    familles_variante : familles de commandes commandes.famille_variante (≥ 50 fois par période, union) ;
    retirer : expressions régulières des colonnes écartées (ex. LETTRES) ; plan : options de
    apprendre_planning (recence, prefixes, changements, paires) ; contexte : options de preparer_contexte."""
    schema = donnees.charger_schema()
    noms = list(circuits or schema["target_columns"])
    indices = [schema["target_columns"].index(c) for c in noms]
    tables = {annee: donnees.lire_dmop(annee) for annee in annees}
    familles = variables.apprendre_familles(list(tables.values()), familles_variante) if jeu == "V2" else []
    planning = (apprendre_planning(list(tables.values()), familles_variante, **(plan or {}))
                if {"planning", "regimes"} & set(ajouts) else None)
    heures, blocs, ys = [], [], []
    for annee in annees:
        h, x, y = donnees.charger_annee(annee, schema)
        heures.append(h)
        contextes = donnees.lire_contextes(annee) if contexte or AJOUTS_CONTEXTE & set(ajouts) else None
        blocs.append(construire(x, tables[annee], jeu, familles, ajouts, planning, sans_horloge, familles_variante,
                                retirer, contextes, contexte))
        y = pd.DataFrame(y[:, indices])
        if lissage_cible:
            y = y.rolling(lissage_cible, center=True, min_periods=1).mean().where(y.notna())
        ys.append(y.to_numpy())
    heures, table, y = np.concatenate(heures), pd.concat(blocs), np.concatenate(ys)
    colonnes, f = list(table.columns), table.to_numpy(dtype=np.float64)
    if filtre:
        garder = monde.calculer_stats(f, y)["garder"]
        colonnes, f = [c for c, g in zip(colonnes, garder) if g], f[:, garder]
    decoupe = sacs_temporels(len(f), sacs, embargo_h)
    n_circuits = y.shape[1]
    processus = int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1))
    ajustes = Parallel(n_jobs=processus)(
        delayed(_ajuster)(f, y[:, i], lignes, config, graine, perte)
        for lignes, _ in decoupe for i in range(n_circuits))
    modeles = [ajustes[k * n_circuits:(k + 1) * n_circuits] for k in range(len(decoupe))]
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    if sacs > 1:
        hors_sac = np.full(y.shape, np.nan)
        for sac, (_, bloc) in zip(modeles, decoupe):
            hors_sac[bloc] = np.stack([m.predict(f[bloc]) for m in sac], axis=1)
        pd.DataFrame(hors_sac, index=pd.Index(heures, name="ut_ms"),
                     columns=noms).to_csv(dossier / "hors_sac.csv")
    with (lzma.open(dossier / "gbm_sacs.pkl.xz", "wb") if compresser else open(dossier / "gbm_sacs.pkl", "wb")) as fichier:
        pickle.dump({"config": config, "jeu": jeu, "colonnes": colonnes, "familles": familles,
                     "ajouts": list(ajouts), "planning": planning, "sans_horloge": sans_horloge,
                     "circuits": noms, "lissage_cible": lissage_cible, "familles_variante": familles_variante, "retirer": list(retirer), "contexte": contexte, "perte": perte, "annees": list(annees), "modeles": modeles}, fichier)
    print(f"{len(modeles)} sac(s) × {n_circuits} boostings ({config}, {jeu}, {len(colonnes)} variables)"
          f" entraînés sur {', '.join(annees)}")


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    chemin = Path(dossier) / "gbm_sacs.pkl"
    with (open(chemin, "rb") if chemin.exists() else lzma.open(chemin.with_suffix(".pkl.xz"), "rb")) as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    table = construire(test["X"], test["dmop"], paquet["jeu"], paquet["familles"], paquet.get("ajouts", ()),
                       paquet.get("planning"), paquet.get("sans_horloge", False), paquet.get("familles_variante", False),
                       paquet.get("retirer", ()), test["tables"], paquet.get("contexte"))
    f = table[paquet["colonnes"]].to_numpy(dtype=np.float64)
    prediction = np.mean([[m.predict(f) for m in sac] for sac in paquet["modeles"]], axis=0).T
    indices = [donnees.charger_schema()["target_columns"].index(c) for c in paquet["circuits"]]
    prediction = recaler(prediction, test["y_init"][:, indices], test["n_init"], recalage, demi_vie)
    return donnees.ecrire_predictions(test["heures"], np.clip(prediction, 0, None), test["heures_a_predire"],
                                      paquet["circuits"], out)
