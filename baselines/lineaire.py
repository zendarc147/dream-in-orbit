"""Ridge sur variables physiques et thermiques : composant classique lisse d'un mélange.

Variables (contexte seul) : flux solaire par face, ombre et leurs moyennes passées
(gbm_sacs.physiques, base FIR en boîtes de 6, 24 et 72 h), minutes de communication et
d'éclipse, distances et constante solaire. Option `retards` : retards explicites de 1 à 48 h
et moyenne passée sur 7 jours des flux éclairés et de l'ombre (réponse impulsionnelle finie
plus fine), sans jamais réinjecter de sortie. Entrées bornées à la plage d'entraînement puis
centrées-réduites. Un seul alpha pour tous les circuits, choisi par blocs temporels
(5 blocs contigus, marge de 7 jours) sur l'année d'entraînement seule ; les prédictions de
chaque bloc écarté sont gardées (hors_sac.csv). Heures où un courant manque : écartées.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from baselines import gbm_sacs
from baselines.boosting import recaler
from dio import commandes, donnees, variables

CONTEXTE = ["ftl_comms_minutes", "evtf_MAR_UMBRA_minutes", "evtf_MAR_PENUMBRA_minutes", "evtf_OCC_MARS_minutes",
            "ltdata_sunmars_km", "ltdata_earthmars_km", "ltdata_eclipseduration_min", "ltdata_solarconstantmars"]
ALPHAS = np.logspace(-1, 5, 13)
RETARDS = (1, 2, 3, 6, 12, 24, 48)


def construire(x, retards=False):
    table = pd.concat([x[CONTEXTE], gbm_sacs.physiques(x)], axis=1)
    if retards:
        lentes = ["ombre"] + [c for c in table.columns if c.endswith("_eclaire")]
        nouvelles = {}
        for c in lentes:
            v = table[c].to_numpy(dtype=np.float64)
            for r in RETARDS:
                nouvelles[f"{c}_retard{r}h"] = commandes.decaler(v[:, None], -r)[:, 0]
            nouvelles[f"{c}_passe168h"] = variables.sommes_glissantes(v, 168)[0]
        table = pd.concat([table, pd.DataFrame(nouvelles, index=table.index)], axis=1)
    return table


def normaliser(f, stats):
    return (np.clip(f, stats["min"], stats["max"]) - stats["moyenne"]) / stats["ecart"]


def entrainer(annees, dossier, retards=False, sacs=5, embargo_h=168):
    schema = donnees.charger_schema()
    heures, tables, ys = [], [], []
    for annee in annees:
        h, x, y = donnees.charger_annee(annee, schema)
        heures.append(h)
        tables.append(construire(x, retards))
        ys.append(y)
    heures, table, y = np.concatenate(heures), pd.concat(tables), np.concatenate(ys)
    f = table.to_numpy(dtype=np.float64)
    ecart = f.std(axis=0)
    stats = {"min": f.min(axis=0), "max": f.max(axis=0), "moyenne": f.mean(axis=0),
             "ecart": np.where(ecart > 1e-9, ecart, 1.0)}
    f = normaliser(f, stats)
    complet = np.isfinite(y).all(axis=1)
    decoupe = gbm_sacs.sacs_temporels(len(f), sacs, embargo_h)
    scores = []
    for alpha in ALPHAS:
        erreurs = []
        for lignes, bloc in decoupe:
            lignes, bloc = lignes[complet[lignes]], bloc[complet[bloc]]
            modele = Ridge(alpha=alpha).fit(f[lignes], y[lignes])
            erreurs.append((modele.predict(f[bloc]) - y[bloc]) ** 2)
        scores.append(float(np.concatenate(erreurs).mean()))
    alpha = float(ALPHAS[int(np.argmin(scores))])
    hors_sac = np.full(y.shape, np.nan)
    for lignes, bloc in decoupe:
        hors_sac[bloc] = Ridge(alpha=alpha).fit(f[lignes[complet[lignes]]], y[lignes[complet[lignes]]]).predict(f[bloc])
    modele = Ridge(alpha=alpha).fit(f[complet], y[complet])
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hors_sac, index=pd.Index(heures, name="ut_ms"),
                 columns=schema["target_columns"]).to_csv(dossier / "hors_sac.csv")
    with open(dossier / "lineaire.pkl", "wb") as fichier:
        pickle.dump({"retards": retards, "colonnes": list(table.columns), "stats": stats, "modele": modele,
                     "alpha": alpha, "scores_alpha": dict(zip(ALPHAS.tolist(), scores)),
                     "circuits": schema["target_columns"], "annees": list(annees)}, fichier)
    print(f"Ridge ({len(table.columns)} variables, alpha {alpha:g}, RMSE hors bloc {min(scores) ** 0.5:.4f})"
          f" entraîné sur {', '.join(annees)}")


def predire(test_dir, out, dossier, recalage=1.0, demi_vie=1440):
    with open(Path(dossier) / "lineaire.pkl", "rb") as fichier:
        paquet = pickle.load(fichier)
    test = donnees.lire_dossier_test(test_dir)
    table = construire(test["X"], paquet["retards"])[paquet["colonnes"]]
    prediction = paquet["modele"].predict(normaliser(table.to_numpy(dtype=np.float64), paquet["stats"]))
    prediction = recaler(prediction, test["y_init"], test["n_init"], recalage, demi_vie)
    return donnees.ecrire_predictions(test["heures"], np.clip(prediction, 0, None), test["heures_a_predire"],
                                      paquet["circuits"], out)
