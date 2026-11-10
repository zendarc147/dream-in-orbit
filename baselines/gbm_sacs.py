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
Les variables quasi constantes sur l'entraînement sont écartées (même règle que monde.py).
power_init ne sert qu'au recalage du niveau, à la prédiction (boosting.recaler).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from baselines.boosting import recaler
from dio import donnees, monde, variables

REGLAGES = {
    # A : réglage retenu de la baseline (E1) ; B : le plus gros réglage encore correct d'E1.
    "A": {"max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 500,
          "l2_regularization": 1.0},
    "B": {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
          "l2_regularization": 1.0},
}
# ET : arbres extrêmement aléatoires (comparaison de diversité, sans sacs : déjà aléatoires).
EXTRA_TREES = {"n_estimators": 100, "min_samples_leaf": 50, "max_features": 0.3}

def construire(x, dmop, jeu, familles=()):
    """Variables d'une période continue (x : X horaire de prepare.py, dmop : table brute)."""
    if jeu == "v1":
        return x.copy()
    if jeu not in ("V1", "V2"):
        raise ValueError(f"Jeu de variables inconnu : {jeu}")
    table = variables.ajouter_v2r(x)
    if jeu == "V2":
        table = pd.concat([table, variables.physiques(x),
                           variables.comptes_familles(dmop, x.index.to_numpy(), familles)], axis=1)
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


def entrainer(annees, dossier, config="A", jeu="V1", sacs=5, embargo_h=168, filtre=True, graine=0):
    schema = donnees.charger_schema()
    tables = {annee: donnees.lire_dmop(annee) for annee in annees}
    familles = variables.apprendre_familles(list(tables.values())) if jeu == "V2" else []
    heures, blocs, ys = [], [], []
    for annee in annees:
        h, x, y = donnees.charger_annee(annee, schema)
        heures.append(h)
        blocs.append(construire(x, tables[annee], jeu, familles))
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
    with open(dossier / "gbm_sacs.pkl", "wb") as fichier:
        pickle.dump({"config": config, "jeu": jeu, "colonnes": colonnes, "familles": familles,
                     "circuits": schema["target_columns"], "annees": list(annees), "modeles": modeles}, fichier)
    print(f"{len(modeles)} sac(s) × {n_circuits} boostings ({config}, {jeu}, {len(colonnes)} variables)"
          f" entraînés sur {', '.join(annees)}")


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    with open(Path(dossier) / "gbm_sacs.pkl", "rb") as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    table = construire(test["X"], test["dmop"], paquet["jeu"], paquet["familles"])
    f = table[paquet["colonnes"]].to_numpy(dtype=np.float64)
    prediction = np.mean([[m.predict(f) for m in sac] for sac in paquet["modeles"]], axis=0).T
    prediction = recaler(prediction, test["y_init"], test["n_init"], recalage, demi_vie)
    return donnees.ecrire_predictions(test["heures"], np.clip(prediction, 0, None), test["heures_a_predire"],
                                      paquet["circuits"], out)
