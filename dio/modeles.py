"""Registre des modèles : chacun expose entrainer(annees, dossier, ...) et predire(test_dir, out, dossier, ...).

"gru_gbm" est le modèle final ; les autres sont ses composants ou la baseline boosting, entraînables seuls
pour la validation et les ablations (outils/valider.py, outils/ablations.py).
"""
import importlib

MODELES = {
    "gru_gbm": "dio.ensemble",       # modèle final (structure S2)
    "gru_recette": "dio.recette",     # GRU S borné ou GRU à boucle, selon les options
    "gru": "dio.monde",               # world model GRU à encodeur (ablations, spécialiste NPWD2532)
    "gru_mixte": "dio.mixte",         # moyenne de deux world models GRU (v1 et v2r)
    "gbm_sacs": "baselines.gbm_sacs",  # boosting à sacs temporels
    "recette_b": "baselines.recette_b",  # boosting propre à NPWD2551
    "boosting": "baselines.boosting",  # baseline : gradient boosting + recalage sur power_init
}
FINAL = "gru_gbm"


def module(nom):
    return importlib.import_module(MODELES[nom])
