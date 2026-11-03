"""Ensemble mixte de world models : plusieurs configurations, prédictions moyennées.

Chaque membre est un ensemble de world models GRU (monde.py) avec son propre jeu de
variables (ex. v1 avec bruit d'entrée, et v2r : fenêtres d'éclipses et de communications,
phase orbitale, cos/sin saaf). Les membres se trompent différemment, surtout sur NPWD2551 :
leur moyenne est plus juste que chacun. L'incertitude est l'écart-type sur tous les
modèles de tous les membres. Tout reste un réseau de neurones.
"""
import json
from pathlib import Path

import numpy as np
import torch

import donnees
import monde

MEMBRES_DEFAUT = [{"variables": "v1", "bruit": 0.3}, {"variables": "v2r", "bruit": 0.0}]


def entrainer(annees, dossier, membres=None, **options):
    dossier = Path(dossier)
    membres = MEMBRES_DEFAUT if membres is None else membres
    for i, membre in enumerate(membres):
        monde.entrainer(annees, dossier / f"membre{i}", **{**options, **membre})
    (dossier / "membres.json").write_text(json.dumps(membres, indent=2) + "\n")


def predire(test_dir, out, dossier, sortie_incertitude=None):
    dossier = Path(dossier)
    membres = json.loads((dossier / "membres.json").read_text())
    test = donnees.lire_dossier_test(test_dir)
    moyennes, carres, poids = 0.0, 0.0, 0
    for i in range(len(membres)):
        sous_dossier = dossier / f"membre{i}"
        m, e = monde.simuler(sous_dossier, test["X"], test["y_init"], test["n_init"], test["dmop"])
        # Chaque membre pèse autant ; variance totale = moyenne des (variance + carré de la moyenne).
        moyennes, carres, poids = moyennes + m, carres + e ** 2 + m ** 2, poids + 1
        circuits = torch.load(sous_dossier / "monde.pt", weights_only=True)["circuits"]
    moyenne = moyennes / poids
    ecart = np.sqrt(np.maximum(carres / poids - moyenne ** 2, 0))
    table = donnees.ecrire_predictions(test["heures"], np.clip(moyenne, 0, None), test["heures_a_predire"],
                                       circuits, out)
    if sortie_incertitude:
        donnees.ecrire_predictions(test["heures"], ecart, test["heures_a_predire"], circuits, sortie_incertitude)
    return table
