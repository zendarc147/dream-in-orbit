"""Registre des modèles : chacun expose entrainer(annees, dossier, ...) et predire(test_dir, out, dossier, ...)."""
import importlib

MODELES = {
    "boosting": "baselines.boosting",
    "gru": "monde",
    "gru_mixte": "mixte",
}
FINAL = "gru_mixte"


def module(nom):
    return importlib.import_module(MODELES[nom])
