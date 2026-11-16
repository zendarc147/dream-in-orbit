"""Gradient boosting par circuit avec bagging temporel : la part boosting du modèle final.

Un HistGradientBoostingRegressor par circuit et par sac. Le sac k apprend sans un bloc contigu (un
cinquième de la période) ; la prédiction est la moyenne des sacs, recalée sur power_init (boosting.recaler).
Variables « S » de dio/caracteristiques.py, toutes tirées du contexte et du planning (jamais d'une
consommation), vocabulaire appris sur les seules périodes d'entraînement. Réglage « D » : taux 0.02, jusqu'à
600 arbres de 15 feuilles, arrêt anticipé sur 10 % de validation interne (docs/journaux/JOURNAL_RECETTE.md).
"""
import json
import lzma
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from baselines.boosting import recaler
from dio import caracteristiques, donnees

REGLAGES = {
    "D": {"max_iter": 600, "learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 100,
          "l2_regularization": 3.0, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
          "tol": 1e-7},
}


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
        modele = HistGradientBoostingRegressor(**{"early_stopping": False, **REGLAGES[config]},
                                               loss="squared_error", random_state=graine)
        return modele.fit(f[lignes][connus], y[connus])


def entrainer(annees, dossier, config="D", sacs=5, embargo_h=0, compresser=False, recette=None, graine=0):
    """compresser : gbm_sacs.pkl.xz (lzma, environ 8 fois plus petit, 2 min de plus à l'écriture).
    recette : options de caracteristiques.construire (None : variables S par défaut)."""
    schema = donnees.charger_schema()
    noms = list(schema["target_columns"])
    periodes = [caracteristiques.charger_periode(a, noms) for a in annees]
    vocab = caracteristiques.apprendre([t for _, t, _ in periodes])
    blocs = [caracteristiques.construire(t, h, vocab, recette) for h, t, _ in periodes]
    table = pd.concat(blocs)[caracteristiques.colonnes_gbm(blocs[0], recette)]
    heures = np.concatenate([h for h, _, _ in periodes])
    y = np.concatenate([yy for _, _, yy in periodes])
    colonnes, f = list(table.columns), table.to_numpy(dtype=np.float64)
    decoupe = sacs_temporels(len(f), sacs, embargo_h)
    n_circuits = y.shape[1]
    ajustes = Parallel(n_jobs=int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1)))(
        delayed(_ajuster)(f, y[:, i], lignes, config, graine)
        for lignes, _ in decoupe for i in range(n_circuits))
    modeles = [ajustes[k * n_circuits:(k + 1) * n_circuits] for k in range(len(decoupe))]
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "colonnes.json").write_text(json.dumps(colonnes, indent=0) + "\n")
    # Nombre effectif d'arbres par circuit et par sac : max_iter n'est qu'un plafond avec l'arrêt anticipé.
    iterations = {c: [int(sac[i].n_iter_) for sac in modeles] for i, c in enumerate(noms)}
    (dossier / "iterations.json").write_text(json.dumps(iterations) + "\n")
    paquet = {"jeu": "S", "vocab": vocab, "recette": recette, "annees": list(annees), "config": config,
              "colonnes": colonnes, "circuits": noms, "modeles": modeles}
    with (lzma.open(dossier / "gbm_sacs.pkl.xz", "wb") if compresser else open(dossier / "gbm_sacs.pkl", "wb")) as fichier:
        pickle.dump(paquet, fichier)
    print(f"{len(modeles)} sac(s) × {n_circuits} boostings ({config}, {len(colonnes)} variables)"
          f" entraînés sur {', '.join(annees)}")


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    chemin = Path(dossier) / "gbm_sacs.pkl"
    with (open(chemin, "rb") if chemin.exists() else lzma.open(chemin.with_suffix(".pkl.xz"), "rb")) as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    table = caracteristiques.construire(test["tables"], test["heures"], paquet["vocab"], paquet.get("recette"))
    f = table[paquet["colonnes"]].to_numpy(dtype=np.float64)
    prediction = np.mean([[m.predict(f) for m in sac] for sac in paquet["modeles"]], axis=0).T
    indices = [donnees.charger_schema()["target_columns"].index(c) for c in paquet["circuits"]]
    prediction = np.clip(recaler(prediction, test["y_init"][:, indices], test["n_init"], recalage, demi_vie), 0, None)
    return donnees.ecrire_predictions(test["heures"], prediction, test["heures_a_predire"],
                                      paquet["circuits"], out)
