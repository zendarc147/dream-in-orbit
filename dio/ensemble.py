"""Ensemble GRU + boosting : (1 - w) × world models GRU (mixte.py) + w × boosting à sacs (gbm_sacs.py).

Le world model reste le modèle principal ; le boosting, sans mémoire, corrige surtout la bande
rapide de quelques circuits (docs/JOURNAL_ENSEMBLE.md). Un seul poids w pour les 33 circuits,
fixé avant l'entraînement final. Retour au GRU seul à la prédiction :
    python predict.py ... --options '{"poids_gbm": 0}'
Dossier : membre*/ et membres.json (GRU), gbm/gbm_sacs.pkl (boosting), ensemble.json (poids).
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from baselines import gbm_sacs
from . import donnees, mixte

POIDS_GBM = 0.3
GBM_DEFAUT = {"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"], "compresser": True}


def entrainer(annees, dossier, poids_gbm=POIDS_GBM, gbm=None, gru_depuis=None, **options):
    """gru_depuis : dossier d'un gru_mixte déjà entraîné sur les mêmes années avec les mêmes options,
    copié au lieu d'être réentraîné (sur CPU, le GRU coûte des heures, le boosting quelques minutes)."""
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


def predire(test_dir, out, dossier, poids_gbm=None, sortie_incertitude=None, recalage=1.0, demi_vie=1440):
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
    return donnees.ecrire_predictions(melange.index.to_numpy(), np.clip(melange.to_numpy(), 0, None),
                                      melange.index.to_numpy(), list(melange.columns), out)
