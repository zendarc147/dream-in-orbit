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
- sans_horloge : retire les distances et angles ltdata, lents et quasi monotones sur une
  année, qui servent d'horloge indirecte.
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
from dio import commandes, donnees, monde, prepare, variables

H = donnees.H
HORLOGE = ["ltdata_sunmars_km", "ltdata_earthmars_km", "ltdata_sunmarsearthangle_deg", "ltdata_solarconstantmars"]
FENETRES_HISTOIRE = (6, 24, 72, 168)
CONSTANTES_EMA = (3, 12, 48, 168)
PLAFOND_RECENCE_H = 168
DUREE_MAX_PERIODE = 7 * 24 * H

REGLAGES = {
    # A : réglage retenu de la baseline (E1) ; B : le plus gros réglage encore correct d'E1.
    "A": {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 500,
          "l2_regularization": 1.0},
    "B": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
          "l2_regularization": 1.0},
    # C : plus lent et plus régularisé (candidat de la phase 2).
    "C": {"max_iter": 600, "learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 100,
          "l2_regularization": 3.0},
}
# ET : arbres extrêmement aléatoires (comparaison de diversité, sans sacs : déjà aléatoires).
EXTRA_TREES = {"n_estimators": 100, "min_samples_leaf": 50, "max_features": 0.3}

def famille_etendue(code):
    """Famille d'une commande : lettres de variante fusionnées ; MOCS.0000003738 -> MOCS."""
    m = re.match(r"^([A-Z]{4})\.\d+$", code)
    return m.group(1) if m else commandes.famille(code)


def apprendre_planning(tables):
    """Familles vues au moins MIN_FAMILLE fois et paires début/fin (XXXS / XXXE) appariables."""
    comptes = pd.concat(tables).subsystem.map(famille_etendue).value_counts()
    familles = sorted(comptes[comptes >= variables.MIN_FAMILLE].index)
    paires = [(f, f[:3] + "E") for f in familles
              if re.fullmatch(r"[A-Z]{3}S", f) and f[:3] + "E" in familles]
    return {"familles": familles, "paires": paires}


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
    table = dmop.assign(subsystem=dmop.subsystem.map(famille_etendue))
    familles = planning["familles"]
    comptes = commandes.impulsions(table, heures, familles)
    cumul = np.vstack([np.zeros((1, len(familles))), np.cumsum(comptes, axis=0)])
    t, n = np.arange(len(heures)), len(heures)
    sortie = {}
    for j, f in enumerate(familles):
        temps = np.sort(table.ut_ms[table.subsystem == f].to_numpy(dtype=np.int64))
        position = np.searchsorted(temps, heures + H, side="left") - 1
        recence = np.full(n, np.inf)  # famille jamais vue avant l'heure (ou absente du test)
        vue = position >= 0
        recence[vue] = (heures[vue] + H - temps[position[vue]]) / H
        sortie[f"plan_{f}_depuis_h"] = np.minimum(recence, PLAFOND_RECENCE_H)
        sortie[f"plan_{f}_futur1h"] = cumul[np.minimum(t + 2, n), j] - cumul[np.minimum(t + 1, n), j]
        sortie[f"plan_{f}_futur6h"] = cumul[np.minimum(t + 7, n), j] - cumul[np.minimum(t + 1, n), j]
    for debut, fin in planning["paires"]:
        sortie[f"plan_{debut}_{fin}_actif"] = periodes_actives(dmop, heures, debut, fin)
    return pd.DataFrame(sortie, index=heures)


def series_contexte(x):
    """Flux solaire éclairé par face, ombre, modes de pointage ftl et communications."""
    physique = variables.physiques(x)
    series = {c: physique[c].to_numpy() for c in physique.columns if c == "ombre" or c.endswith("_eclaire")}
    for c in x.columns:
        if c.startswith("ftl_") and c not in ("ftl_coverage_minutes", "ftl_overlap_minutes"):
            series[c] = x[c].to_numpy(dtype=np.float64)
    return series


def historique(x, existantes):
    """Moyennes passées des séries de contexte, sans les colonnes déjà présentes (même nom)."""
    sortie = {}
    for nom, serie in series_contexte(x).items():
        for w in FENETRES_HISTOIRE:
            if f"{nom}_passe{w}h" not in existantes:
                sortie[f"{nom}_passe{w}h"] = variables.sommes_glissantes(serie, w)[0]
    return pd.DataFrame(sortie, index=x.index)


def moyennes_exponentielles(x):
    """y_t = a x_t + (1 - a) y_(t-1), a = 1 - exp(-1/tau), y_0 = x_0 : causale."""
    sortie = {}
    for nom, serie in series_contexte(x).items():
        for tau in CONSTANTES_EMA:
            a = 1 - np.exp(-1 / tau)
            sortie[f"{nom}_ema{tau}h"] = lfilter([a], [1, a - 1], serie, zi=[(1 - a) * serie[0]])[0]
    return pd.DataFrame(sortie, index=x.index)


def construire(x, dmop, jeu, familles=(), ajouts=(), planning=None, sans_horloge=False):
    """Variables d'une période continue (x : X horaire de prepare.py, dmop : table brute)."""
    if jeu == "v1":
        return x.copy()
    if jeu not in ("V1", "V2"):
        raise ValueError(f"Jeu de variables inconnu : {jeu}")
    inconnus = set(ajouts) - {"planning", "historique", "ema"}
    if inconnus:
        raise ValueError(f"Ajouts inconnus : {sorted(inconnus)}")
    table = variables.ajouter_v2r(x)
    if jeu == "V2":
        table = pd.concat([table, variables.physiques(x),
                           variables.comptes_familles(dmop, x.index.to_numpy(), familles)], axis=1)
    blocs = [table]
    if "planning" in ajouts:
        blocs.append(variables_planning(dmop, x.index.to_numpy(), planning).set_axis(x.index))
    if "historique" in ajouts:
        blocs.append(historique(x, set(table.columns)))
    if "ema" in ajouts:
        blocs.append(moyennes_exponentielles(x))
    table = pd.concat(blocs, axis=1)
    if sans_horloge:
        table = table.drop(columns=HORLOGE)
    return table


def sacs_temporels(n, sacs, embargo):
    """(indices d'entraînement, bloc écarté) de chaque sac ; un seul sac = toute la période."""
    t = np.arange(n)
    if sacs == 1:
        return [(t, t[:0])]
    bornes = np.linspace(0, n, sacs + 1).astype(int)
    return [(t[(t < a - embargo) | (t >= b + embargo)], t[a:b]) for a, b in zip(bornes[:-1], bornes[1:])]


def _ajuster(f, y, lignes, config, graine):
    """Un circuit, un sac ; un seul thread par processus (les données sont petites)."""
    y = y[lignes]
    connus = np.isfinite(y)  # cibles absentes retirées, jamais remplacées
    with threadpool_limits(1):
        if config == "ET":
            modele = ExtraTreesRegressor(**EXTRA_TREES, random_state=graine, n_jobs=1)
        else:
            modele = HistGradientBoostingRegressor(**REGLAGES[config], early_stopping=False, random_state=graine)
        return modele.fit(f[lignes][connus], y[connus])


def entrainer(annees, dossier, config="A", jeu="V1", sacs=5, embargo_h=168, filtre=True, graine=0,
              ajouts=(), sans_horloge=False, compresser=False):
    """compresser : gbm_sacs.pkl.xz (lzma, environ 8 fois plus petit, 2 min de plus à l'écriture)."""
    schema = donnees.charger_schema()
    tables = {annee: donnees.lire_dmop(annee) for annee in annees}
    familles = variables.apprendre_familles(list(tables.values())) if jeu == "V2" else []
    planning = apprendre_planning(list(tables.values())) if "planning" in ajouts else None
    heures, blocs, ys = [], [], []
    for annee in annees:
        h, x, y = donnees.charger_annee(annee, schema)
        heures.append(h)
        blocs.append(construire(x, tables[annee], jeu, familles, ajouts, planning, sans_horloge))
        ys.append(y)
    heures, table, y = np.concatenate(heures), pd.concat(blocs), np.concatenate(ys)
    colonnes, f = list(table.columns), table.to_numpy(dtype=np.float64)
    if filtre:
        garder = monde.calculer_stats(f, y)["garder"]
        colonnes, f = [c for c, g in zip(colonnes, garder) if g], f[:, garder]
    decoupe = sacs_temporels(len(f), sacs, embargo_h)
    n_circuits = y.shape[1]
    processus = int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1))
    ajustes = Parallel(n_jobs=processus)(
        delayed(_ajuster)(f, y[:, i], lignes, config, graine)
        for lignes, _ in decoupe for i in range(n_circuits))
    modeles = [ajustes[k * n_circuits:(k + 1) * n_circuits] for k in range(len(decoupe))]
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    if sacs > 1:
        hors_sac = np.full(y.shape, np.nan)
        for sac, (_, bloc) in zip(modeles, decoupe):
            hors_sac[bloc] = np.stack([m.predict(f[bloc]) for m in sac], axis=1)
        pd.DataFrame(hors_sac, index=pd.Index(heures, name="ut_ms"),
                     columns=schema["target_columns"]).to_csv(dossier / "hors_sac.csv")
    with (lzma.open(dossier / "gbm_sacs.pkl.xz", "wb") if compresser else open(dossier / "gbm_sacs.pkl", "wb")) as fichier:
        pickle.dump({"config": config, "jeu": jeu, "colonnes": colonnes, "familles": familles,
                     "ajouts": list(ajouts), "planning": planning, "sans_horloge": sans_horloge,
                     "circuits": schema["target_columns"], "annees": list(annees), "modeles": modeles}, fichier)
    print(f"{len(modeles)} sac(s) × {n_circuits} boostings ({config}, {jeu}, {len(colonnes)} variables)"
          f" entraînés sur {', '.join(annees)}")


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    chemin = Path(dossier) / "gbm_sacs.pkl"
    with (open(chemin, "rb") if chemin.exists() else lzma.open(chemin.with_suffix(".pkl.xz"), "rb")) as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    table = construire(test["X"], test["dmop"], paquet["jeu"], paquet["familles"], paquet.get("ajouts", ()),
                       paquet.get("planning"), paquet.get("sans_horloge", False))
    f = table[paquet["colonnes"]].to_numpy(dtype=np.float64)
    prediction = np.mean([[m.predict(f) for m in sac] for sac in paquet["modeles"]], axis=0).T
    prediction = recaler(prediction, test["y_init"], test["n_init"], recalage, demi_vie)
    return donnees.ecrire_predictions(test["heures"], np.clip(prediction, 0, None), test["heures_a_predire"],
                                      paquet["circuits"], out)
