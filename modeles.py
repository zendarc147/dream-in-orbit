"""Registre des modèles : chacun expose entrainer(annees, dossier, ...) et predire(test_dir, out, dossier, ...)."""
import importlib

MODELES = {
    "boosting": "baselines.boosting",
    "gru": "monde",
}
FINAL = "gru"


def module(nom):
    return importlib.import_module(MODELES[nom])
