"""Modèle en deux régimes avec mémoire du régime, pour un circuit à chauffage tout-ou-rien (NPWD2551).

docs/JOURNAL_2551.md, programme 3. La moyenne journalière de NPWD2551 est bimodale (régime bas < 0.1, haut
0.25–0.8). Trois boostings sur les mêmes variables (baselines/gbm_sacs.preparer) :
- un classifieur de P(jour en régime haut | contexte de l'heure), étiquette = moyenne du jour > SEUIL_ON ;
- deux régressions, E[y | régime haut, x] et E[y | régime bas, x], apprises sur les heures de chaque régime.
Mémoire du régime : les probabilités moyennes par jour passent dans une chaîne de Markov à deux états
(lisser) ; chaque jour, l'état se renouvelle avec la probabilité `renouvellement` (tiré selon la part de
régime haut de l'entraînement), sinon il persiste. renouvellement = 1 redonne les probabilités du classifieur.
Le lissage avant-arrière utilise tout le contexte du test, jamais une consommation ; seuls les jours de
power_init fixent l'état (observé). Prédiction : π m_haut + (1 − π) m_bas, bornée à 0, sans recalage.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from baselines import gbm_sacs
from dio import donnees

H = donnees.H
JOUR = 24 * H
SEUIL_ON = 0.15
MIN_HEURES_JOUR = 6
CONFIANCE_INIT = 1 - 1e-6  # régime des jours de power_init : observé
P_MIN = 1e-3


def etiquettes(heures, y):
    """Par heure : 1 si la moyenne des heures connues de son jour dépasse SEUIL_ON, 0 sinon, NaN si le jour
    a moins de MIN_HEURES_JOUR heures connues."""
    jours = np.asarray(heures, dtype=np.int64) // JOUR
    _, inverse = np.unique(jours, return_inverse=True)
    connus = np.isfinite(y)
    somme = np.bincount(inverse, np.where(connus, y, 0.0))
    nombre = np.bincount(inverse, connus.astype(float))
    moyenne = np.where(nombre >= MIN_HEURES_JOUR, somme / np.maximum(nombre, 1), np.nan)[inverse]
    return np.where(np.isfinite(moyenne), (moyenne > SEUIL_ON).astype(float), np.nan)


def ajuster(f, y, heures, reglage="D3", graine=0):
    etiquette = etiquettes(heures, y)
    params = {"early_stopping": False, **gbm_sacs.REGLAGES[reglage]}
    lab = np.isfinite(etiquette)
    clf = HistGradientBoostingClassifier(**params, random_state=graine).fit(f[lab], etiquette[lab])
    connus = np.isfinite(y)
    haut = HistGradientBoostingRegressor(**params, random_state=graine).fit(
        f[connus & (etiquette == 1)], y[connus & (etiquette == 1)])
    bas = HistGradientBoostingRegressor(**params, random_state=graine).fit(
        f[connus & (etiquette == 0)], y[connus & (etiquette == 0)])
    return {"clf": clf, "haut": haut, "bas": bas, "part_haut": float(np.mean(etiquette[lab]))}


def composantes(modele, f):
    return modele["clf"].predict_proba(f)[:, 1], modele["haut"].predict(f), modele["bas"].predict(f)


def lisser(p, heures, renouvellement, part_haut, observes=None):
    """Probabilité a posteriori du régime haut, par heure (constante dans un jour). observes : {jour: 0/1}
    des jours dont le régime est connu (power_init), fixés avec CONFIANCE_INIT."""
    jours = np.asarray(heures, dtype=np.int64) // JOUR
    uniques, inverse = np.unique(jours, return_inverse=True)
    pj = np.clip(np.bincount(inverse, p) / np.bincount(inverse), P_MIN, 1 - P_MIN)
    # Vraisemblances (à une constante près) : rapport des cotes du classifieur à celles de l'a priori.
    v = np.stack([np.full(len(pj), 1.0), (pj / (1 - pj)) / (part_haut / (1 - part_haut))], axis=1)
    for j, etat in (observes or {}).items():
        k = np.searchsorted(uniques, j)
        if k < len(uniques) and uniques[k] == j:
            v[k] = [1 - CONFIANCE_INIT, CONFIANCE_INIT] if etat else [CONFIANCE_INIT, 1 - CONFIANCE_INIT]
    a_priori = np.array([1 - part_haut, part_haut])
    a = (1 - renouvellement) * np.eye(2) + renouvellement * a_priori[None, :]
    n = len(pj)
    avant, arriere = np.zeros((n, 2)), np.ones((n, 2))
    courant = a_priori * v[0]
    avant[0] = courant / courant.sum()
    for t in range(1, n):
        courant = (avant[t - 1] @ a) * v[t]
        avant[t] = courant / courant.sum()
    for t in range(n - 2, -1, -1):
        courant = a @ (v[t + 1] * arriere[t + 1])
        arriere[t] = courant / courant.sum()
    post = avant * arriere
    return (post[:, 1] / post.sum(axis=1))[inverse]


def observes_init(heures, y, n_init):
    """{jour: régime} des jours entièrement couverts par les n_init heures initiales (consommation fournie)."""
    heures = np.asarray(heures, dtype=np.int64)[:n_init]
    etiquette = etiquettes(heures, y[:n_init])
    jours = heures // JOUR
    sortie = {}
    for j in np.unique(jours):
        m = jours == j
        if m.sum() == 24 and np.isfinite(etiquette[m][0]):
            sortie[int(j)] = int(etiquette[m][0])
    return sortie


def melanger(pi, haut, bas):
    return np.clip(pi * haut + (1 - pi) * bas, 0, None)


def entrainer(annees, dossier, circuit="NPWD2551", renouvellement=0.1, reglage="D3", options=None):
    """options : celles de gbm_sacs.preparer (variables)."""
    schema = donnees.charger_schema()
    heures, table, y, meta = gbm_sacs.preparer(annees, [schema["target_columns"].index(circuit)],
                                               schema=schema, **(options or {}))
    modele = ajuster(table.to_numpy(dtype=np.float64), y[:, 0], heures, reglage)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    with open(dossier / "regime.pkl", "wb") as fichier:
        pickle.dump({**meta, "colonnes": list(table.columns), "circuit": circuit,
                     "renouvellement": renouvellement, "modele": modele}, fichier)


def predire(test_dir, out, dossier):
    with open(Path(dossier) / "regime.pkl", "rb") as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    f = gbm_sacs.construire_test(test, paquet)[paquet["colonnes"]].to_numpy(dtype=np.float64)
    i = donnees.charger_schema()["target_columns"].index(paquet["circuit"])
    p, haut, bas = composantes(paquet["modele"], f)
    pi = lisser(p, test["heures"], paquet["renouvellement"], paquet["modele"]["part_haut"],
                observes_init(test["heures"], test["y_init"][:, i], test["n_init"]))
    return donnees.ecrire_predictions(test["heures"], melanger(pi, haut, bas)[:, None], test["heures_a_predire"],
                                      [paquet["circuit"]], out)
