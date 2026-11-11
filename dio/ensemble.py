"""Ensemble GRU + boosting : (1 - w) × world models GRU (mixte.py) + w × boosting à sacs (gbm_sacs.py).

Le world model reste le modèle principal ; le boosting, sans mémoire, corrige surtout la bande
rapide de quelques circuits (docs/JOURNAL_ENSEMBLE.md). Un seul poids w pour les 33 circuits,
fixé avant l'entraînement final. Retour au GRU seul à la prédiction :
    python predict.py ... --options '{"poids_gbm": 0}'
Dossier : membre*/ et membres.json (GRU), gbm/gbm_sacs.pkl (boosting), ensemble.json (poids).

Spécialistes de deux circuits (docs/JOURNAL_SPECIALISTES.md), dans specialiste/ si présent :
- NPWD2532 = 0.6 × boosting + 0.4 × world model avec variables physiques (specialiste/physique/) ;
- NPWD2531 : hors des marqueurs de régime du planning, niveau médian recalé sur power_init au lieu
  de l'ensemble (baselines/marqueurs.py, specialiste/marqueurs.json).
Les 31 autres colonnes ne changent pas. Retour à l'ensemble seul : --options '{"specialistes": false}'.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs, marqueurs
from . import donnees, mixte

POIDS_GBM = 0.3
GBM_DEFAUT = {"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"], "compresser": True}
# World model du spécialiste de NPWD2532 : configuration de l'essai H1a (docs/RESULTATS.md).
PHYSIQUE = {"physique": True, "familles": False, "facteur_niveau_prediction": 1.0}
SPECIALISTES = {"NPWD2532": {"gbm": 0.6, "physique": 0.4}, "NPWD2531": {"porte": "marqueurs"}}


def entrainer(annees, dossier, poids_gbm=POIDS_GBM, gbm=None, gru_depuis=None, specialistes=True,
              physique_depuis=None, **options):
    """gru_depuis : dossier d'un gru_mixte déjà entraîné sur les mêmes années avec les mêmes options,
    copié au lieu d'être réentraîné (sur CPU, le GRU coûte des heures, le boosting quelques minutes).
    physique_depuis : idem pour le world model physique du spécialiste de NPWD2532."""
    dossier = Path(dossier)
    if gru_depuis and Path(gru_depuis).resolve() != dossier.resolve():
        dossier.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(gru_depuis) / "membres.json", dossier / "membres.json")
        for membre in sorted(Path(gru_depuis).glob("membre[0-9]*")):
            shutil.copytree(membre, dossier / membre.name, dirs_exist_ok=True)
    elif not gru_depuis:
        mixte.entrainer(annees, dossier, **options)
    gbm_sacs.entrainer(annees, dossier / "gbm", **{**GBM_DEFAUT, **(gbm or {})})
    (dossier / "gbm" / "hors_sac.csv").unlink(missing_ok=True)  # diagnostic d'empilement, inutile ici
    (dossier / "ensemble.json").write_text(json.dumps({"poids_gbm": poids_gbm}, indent=2) + "\n")
    if specialistes:
        entrainer_specialistes(annees, dossier, physique_depuis, **options)


def entrainer_specialistes(annees, dossier, physique_depuis=None, **options):
    specialiste = Path(dossier) / "specialiste"
    if physique_depuis and Path(physique_depuis).resolve() == (specialiste / "physique").resolve():
        pass  # déjà en place (réentraînement du seul boosting dans poids/)
    elif physique_depuis:
        shutil.copytree(physique_depuis, specialiste / "physique", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.csv", "*.log", "modele.json"))
    else:
        mixte.entrainer(annees, specialiste / "physique", **{**options, **PHYSIQUE})
    marqueurs.entrainer(annees, specialiste, [c for c, s in SPECIALISTES.items() if s.get("porte")])
    (specialiste / "specialistes.json").write_text(json.dumps(SPECIALISTES, indent=2) + "\n")


def predire(test_dir, out, dossier, poids_gbm=None, sortie_incertitude=None, recalage=1.0, demi_vie=1440,
            specialistes=True):
    dossier = Path(dossier)
    if poids_gbm is None:
        poids_gbm = json.loads((dossier / "ensemble.json").read_text())["poids_gbm"]
    gru = mixte.predire(test_dir, out, dossier, sortie_incertitude)
    if not poids_gbm:
        return gru
    with tempfile.TemporaryDirectory() as temp:
        gbm = gbm_sacs.predire(test_dir, Path(temp) / "gbm.csv", dossier / "gbm", recalage, demi_vie)
    if not gru.index.equals(gbm.index) or list(gru.columns) != list(gbm.columns):
        raise ValueError("Heures ou circuits différents entre GRU et boosting")
    melange = (1 - poids_gbm) * gru + poids_gbm * gbm
    specialiste = dossier / "specialiste"
    if specialistes and (specialiste / "specialistes.json").is_file():
        melange = appliquer_specialistes(test_dir, specialiste, melange, gbm)
    return donnees.ecrire_predictions(melange.index.to_numpy(), np.clip(melange.to_numpy(), 0, None),
                                      melange.index.to_numpy(), list(melange.columns), out)


def appliquer_specialistes(test_dir, specialiste, melange, gbm):
    """Remplacer les seules colonnes des spécialistes ; les autres restent identiques au bit près."""
    regles = json.loads((specialiste / "specialistes.json").read_text())
    melange = melange.copy()
    physique = None
    with tempfile.TemporaryDirectory() as temp:
        for c, regle in regles.items():
            if "physique" in regle:
                if physique is None:
                    physique = mixte.predire(test_dir, Path(temp) / "physique.csv", specialiste / "physique")
                melange[c] = regle["gbm"] * gbm[c] + regle["physique"] * physique[c]
    portes = [c for c, regle in regles.items() if regle.get("porte") == "marqueurs"]
    if portes:
        test = donnees.lire_dossier_test(test_dir)
        for c, (hors, garder) in marqueurs.porte(test, specialiste).items():
            hors = pd.Series(hors, index=test["heures"]).reindex(melange.index)
            garder = pd.Series(garder, index=test["heures"]).reindex(melange.index)
            melange[c] = melange[c].where(garder, hors)
    return melange
