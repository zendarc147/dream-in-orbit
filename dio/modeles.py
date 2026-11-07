"""Registre des modèles : chacun expose entrainer(annees, dossier, ...) et predire(test_dir, out, dossier, ...)."""
import importlib

MODELES = {
    "boosting": "baselines.boosting",
    "gru": "dio.monde",
    "gru_mixte": "dio.mixte",
}
FINAL = "gru_mixte"


def module(nom):
    return importlib.import_module(MODELES[nom])
