"""Registre des modèles : chacun expose entrainer(annees, dossier, ...) et predire(test_dir, out, dossier, ...)."""
import importlib

MODELES = {
    "boosting": "baselines.boosting",
    "gbm_sacs": "baselines.gbm_sacs",
    "lineaire": "baselines.lineaire",
    "gru": "dio.monde",
    "gru_mixte": "dio.mixte",
    "gru_gbm": "dio.ensemble",
    "gru_ar": "dio.autoregressif",
}
FINAL = "gru_mixte"


def module(nom):
    return importlib.import_module(MODELES[nom])
