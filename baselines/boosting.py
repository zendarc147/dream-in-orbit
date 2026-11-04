"""Baseline gradient boosting : un modèle par circuit, heure par heure, sans mémoire.

Recalage optionnel : on ajoute à chaque circuit une fraction de l'écart moyen
observé entre la vérité et la prédiction sur les 7 jours de power_init.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import donnees  # noqa: E402


def entrainer(annees, dossier, graine=0, max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
              min_samples_leaf=500):
    schema = donnees.charger_schema()
    xs, ys = [], []
    for annee in annees:
        _, x, y = donnees.charger_annee(annee, schema)
        xs.append(x.to_numpy(dtype=np.float64))
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    modeles = []
    for i, circuit in enumerate(schema["target_columns"]):
        connus = np.isfinite(y[:, i])
        modele = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                               max_leaf_nodes=max_leaf_nodes,
                                               min_samples_leaf=min_samples_leaf, l2_regularization=1.0,
                                               random_state=graine)
        modeles.append(modele.fit(x[connus], y[connus, i]))
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    with open(dossier / "boosting.pkl", "wb") as f:
        pickle.dump({"circuits": schema["target_columns"], "modeles": modeles, "annees": list(annees)}, f)
    print(f"{len(modeles)} boostings entraînés sur {', '.join(annees)}")


def recaler(prediction, y_init, n_init, poids, demi_vie=None):
    """Décaler chaque circuit d'une fraction de l'écart moyen sur les heures initiales.

    Avec une demi-vie (en heures), le décalage s'estompe après les 7 jours initiaux.
    """
    ecart = y_init[:n_init] - prediction[:n_init]
    connus = np.isfinite(ecart)
    nombre = connus.sum(axis=0)
    decalage = np.where(nombre > 0, np.where(connus, ecart, 0).sum(axis=0) / np.maximum(nombre, 1), 0.0)
    attenuation = np.ones(len(prediction))
    if demi_vie:
        attenuation[n_init:] = 0.5 ** (np.arange(1, len(prediction) - n_init + 1) / demi_vie)
    return prediction + poids * attenuation[:, None] * decalage


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    with open(Path(dossier) / "boosting.pkl", "rb") as f:
        paquet = pickle.load(f)
    test = donnees.lire_dossier_test(test_dir)
    x = test["X"].to_numpy(dtype=np.float64)
    prediction = np.stack([m.predict(x) for m in paquet["modeles"]], axis=1)
    prediction = recaler(prediction, test["y_init"], test["n_init"], recalage, demi_vie)
    prediction = np.clip(prediction, 0, None)
    return donnees.ecrire_predictions(test["heures"], prediction, test["heures_a_predire"],
                                      paquet["circuits"], out)
