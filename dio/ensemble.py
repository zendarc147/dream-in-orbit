"""Modèle final (structure S2) : réseaux de neurones + boosting, avec trois spécialistes.

Part réseau (world model) = 0.5 × GRU S borné (recette/, recalé sur power_init)
                          + 0.5 × GRU à boucle (recette2/, relit sa propre sortie, sans recalage) ;
part boosting = boosting à sacs temporels sur les variables S (gbm/, recalé sur power_init) ;
prédiction = 0.5 × part réseau + 0.5 × part boosting.

Spécialistes (specialiste/, docs/journaux/JOURNAL_SPECIALISTES.md et JOURNAL_2551.md) :
- NPWD2551 : la part boosting vient de la recette B, un boosting propre à ce circuit (b_NPWD2551/) ;
- NPWD2532 : 0.6 × boosting + 0.4 × world model GRU avec variables physiques (physique/) ;
- NPWD2531 : hors des marqueurs de régime du planning, niveau médian recalé sur power_init.
Les 30 autres colonnes ne dépendent que du mélange.

Options de prédiction (predict.py --options) : {"poids_gbm": 0} donne la part réseau seule,
{"specialistes": false} le mélange sans spécialistes, {"sortie_incertitude": "ecart.csv"} écrit l'écart-type
des 20 GRU de la part réseau. Choix et mesures : docs/journaux/JOURNAL_GRU_BOUCLE.md.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs, marqueurs, recette_b
from . import donnees, mixte, recette

POIDS_GBM = 0.5   # part du boosting dans le mélange
PART_BOUCLE = 0.5  # part du GRU à boucle dans la part réseau
GRAINES = list(range(10))
GRU_S = {"variables": "S", "reinjection": False, "normalisation": "minmax", "graines": GRAINES}
GRU_BOUCLE = {"variables": "S", "reinjection": True, "premier_segment_verite": True, "normalisation": "standard",
              "graines": GRAINES}
GBM = {"config": "D", "sacs": 5, "embargo_h": 0, "compresser": True}
# World model du spécialiste de NPWD2532 (essai H1a de docs/RESULTATS.md), deux membres de mixte.py.
PHYSIQUE = {"physique": True, "familles": False, "facteur_niveau_prediction": 1.0}
SPECIALISTES = {"NPWD2532": {"gbm": 0.6, "physique": 0.4}, "NPWD2531": {"porte": "marqueurs"},
                "NPWD2551": {"recette_b": 1.0}}
RESEAUX = ("recette/recette.pt", "recette2/recette.pt", "specialiste/physique")


def entrainer(annees, dossier, reseaux_depuis=None):
    """reseaux_depuis : dossier de poids déjà entraînés sur les mêmes années ; ses réseaux (les deux GRU et le
    world model physique) sont copiés au lieu d'être réentraînés, seuls les boostings sont refaits
    (quelques minutes sur CPU au lieu de plusieurs heures)."""
    dossier = Path(dossier)
    if reseaux_depuis:
        copier_reseaux(Path(reseaux_depuis), dossier)
    else:
        recette.entrainer(annees, dossier / "recette", **GRU_S)
        recette.entrainer(annees, dossier / "recette2", **GRU_BOUCLE)
        mixte.entrainer(annees, dossier / "specialiste" / "physique", **PHYSIQUE)
    gbm_sacs.entrainer(annees, dossier / "gbm", **GBM)
    specialiste = dossier / "specialiste"
    recette_b.entrainer(annees, specialiste / "b_NPWD2551")
    marqueurs.entrainer(annees, specialiste, [c for c, s in SPECIALISTES.items() if s.get("porte")])
    (specialiste / "specialistes.json").write_text(json.dumps(SPECIALISTES, indent=2) + "\n")
    (dossier / "ensemble.json").write_text(json.dumps({"poids_gbm": POIDS_GBM, "recette2": PART_BOUCLE},
                                                      indent=2) + "\n")


def copier_reseaux(source, dossier):
    if source.resolve() == dossier.resolve():
        return
    for chemin in RESEAUX:
        cible = dossier / chemin
        cible.parent.mkdir(parents=True, exist_ok=True)
        if (source / chemin).is_dir():
            shutil.copytree(source / chemin, cible, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.csv", "*.log", "modele.json"))
        else:
            shutil.copy(source / chemin, cible)


def predire(test_dir, out, dossier, poids_gbm=None, sortie_incertitude=None, specialistes=True, parts=False):
    """parts : écrire aussi les composants (dossier <out>.parts/) ; la prédiction ne change pas."""
    dossier = Path(dossier)
    reglages = json.loads((dossier / "ensemble.json").read_text())
    poids_gbm = reglages["poids_gbm"] if poids_gbm is None else poids_gbm
    part2 = reglages["recette2"]
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        gru_s, membres_s = recette.predire(test_dir, temp / "s.csv", dossier / "recette", recalage=1.0,
                                           membres=True)
        boucle, membres_b = recette.predire(test_dir, temp / "b.csv", dossier / "recette2", recalage=0.0,
                                            membres=True)
        gru = (1 - part2) * gru_s + part2 * boucle
        if sortie_incertitude:
            ecart = np.concatenate([membres_s, membres_b]).std(axis=0)
            donnees.ecrire_predictions(gru.index.to_numpy(), ecart, gru.index.to_numpy(), list(gru.columns),
                                       sortie_incertitude)
        composants = {"gru_s": gru_s, "gru_boucle": boucle, "gru": gru}
        if not poids_gbm:
            return ecrire(gru, out, parts, composants)
        gbm = gbm_sacs.predire(test_dir, temp / "gbm.csv", dossier / "gbm", 1.0, 1440)
    specialiste = dossier / "specialiste"
    specialistes = specialistes and (specialiste / "specialistes.json").is_file()
    if specialistes:
        gbm = gbm_specialistes(test_dir, specialiste, gbm)
    melange = (1 - poids_gbm) * gru + poids_gbm * gbm
    composants.update(gbm=gbm, melange=melange)
    if specialistes:
        melange = appliquer_specialistes(test_dir, specialiste, melange, gbm)
    return ecrire(melange, out, parts, composants)


def ecrire(table, out, parts, composants):
    if parts:
        sortie = Path(str(out) + ".parts")
        sortie.mkdir(parents=True, exist_ok=True)
        for nom, t in composants.items():
            t.to_csv(sortie / f"{nom}.csv")
    return donnees.ecrire_predictions(table.index.to_numpy(), np.clip(table.to_numpy(), 0, None),
                                      table.index.to_numpy(), list(table.columns), out)


def gbm_specialistes(test_dir, specialiste, gbm):
    """Remplacer la part boosting des circuits qui ont leur recette B (b_<circuit>/)."""
    regles = json.loads((specialiste / "specialistes.json").read_text())
    gbm = gbm.copy()
    for c, regle in regles.items():
        if regle.get("recette_b"):
            b = recette_b.predire(test_dir, None, specialiste / f"b_{c}")[c].reindex(gbm.index)
            if b.isna().any():
                raise ValueError(f"recette B : heures manquantes pour {c}")
            gbm[c] = b
    return gbm


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
