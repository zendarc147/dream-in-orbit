"""Ensemble GRU + boosting : (1 - w) × world models GRU (mixte.py) + w × boosting à sacs (gbm_sacs.py).

Le world model reste le modèle principal ; le boosting, sans mémoire, corrige surtout la bande
rapide de quelques circuits (docs/JOURNAL_ENSEMBLE.md). Un seul poids w pour les 33 circuits,
fixé avant l'entraînement final. Retour au GRU seul à la prédiction :
    python predict.py ... --options '{"poids_gbm": 0}'
Dossier : membre*/ et membres.json (GRU), gbm/gbm_sacs.pkl (boosting), ensemble.json (poids).

Spécialistes de deux circuits (docs/JOURNAL_SPECIALISTES.md), dans specialiste/ si présent :
- NPWD2532 = 0.6 × boosting + 0.4 × world model avec variables physiques (specialiste/physique/) ;
- NPWD2531 : hors des marqueurs de régime du planning, niveau médian recalé sur power_init au lieu
  de l'ensemble (baselines/marqueurs.py, specialiste/marqueurs.json) ;
- NPWD2551 : la part boosting vient d'un boosting appris sur ce seul circuit, plus long (D3L) et avec
  les variables d'orbite (specialiste/gbm_NPWD2551/, docs/JOURNAL_2551.md).
Les 30 autres colonnes ne changent pas. Retour à l'ensemble seul : --options '{"specialistes": false}'.

GRU recette (dio/recette.py, sans réinjection, recalé sur power_init), dans recette/ si part_recette > 0 :
la part GRU devient (1 - part_recette) × GRU mixte + part_recette × GRU recette (docs/JOURNAL_RECETTE.md).
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs, marqueurs
from . import donnees, mixte, recette

POIDS_GBM = 0.5  # 0.3 jusqu'au 12 nov. ; 0.5 avec le boosting recette (docs/JOURNAL_RECETTE.md)
PART_RECETTE = 0.3  # part du GRU recette dans la part GRU
# Recette R (docs/JOURNAL_RECETTE.md, E22b) : options de baselines/gbm_sacs.entrainer.
GBM_DEFAUT = {"config": "D", "jeu": "V2", "compresser": True, "familles_variante": True, "plan": {"changements": True, "recence": "debut", "prefixes": True, "paires": "liste"}, "contexte": {"soleil": "interpole", "ltdata": "lineaire"}, "ajouts": ["planning", "ema", "geometrie", "altitude", "ombres", "contacts", "decalages", "histoires", "face_sa_moins"], "embargo_h": 0, "filtre": False, "retirer": ["lettres", "distance_terre", "ema"]}
# World model du spécialiste de NPWD2532 : configuration de l'essai H1a (docs/RESULTATS.md).
PHYSIQUE = {"physique": True, "familles": False, "facteur_niveau_prediction": 1.0}
RECETTE = {"reinjection": False}
SPECIALISTES = {"NPWD2532": {"gbm": 0.6, "physique": 0.4}, "NPWD2531": {"porte": "marqueurs"},
                "NPWD2551": {"gbm_specialiste": {"config": "D3L", "ajouts": GBM_DEFAUT["ajouts"] + ["orbite"]}}}


def entrainer(annees, dossier, poids_gbm=POIDS_GBM, gbm=None, gru_depuis=None, specialistes=True,
              physique_depuis=None, part_recette=PART_RECETTE, recette_depuis=None, gbm_depuis=None, **options):
    """gru_depuis : dossier d'un gru_mixte déjà entraîné sur les mêmes années avec les mêmes options,
    copié au lieu d'être réentraîné (sur CPU, le GRU coûte des heures, le boosting quelques minutes).
    physique_depuis : idem pour le world model physique du spécialiste de NPWD2532 ; recette_depuis : idem
    pour le GRU recette (entraîné sinon avec RECETTE si part_recette > 0) ; gbm_depuis : idem pour les
    boostings propres à un circuit (gbm_<circuit>/)."""
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
    if part_recette:
        if recette_depuis:
            (dossier / "recette").mkdir(parents=True, exist_ok=True)
            if Path(recette_depuis).resolve() != (dossier / "recette").resolve():
                shutil.copy(Path(recette_depuis) / "recette.pt", dossier / "recette" / "recette.pt")
        else:
            recette.entrainer(annees, dossier / "recette", **RECETTE)
    (dossier / "ensemble.json").write_text(json.dumps({"poids_gbm": poids_gbm, "part_recette": part_recette},
                                                      indent=2) + "\n")
    if specialistes:
        entrainer_specialistes(annees, dossier, physique_depuis, gbm_depuis, **options)


def entrainer_specialistes(annees, dossier, physique_depuis=None, gbm_depuis=None, **options):
    """gbm_depuis : dossier contenant des gbm_<circuit>/ déjà entraînés sur les mêmes années, copiés."""
    specialiste = Path(dossier) / "specialiste"
    for c, regle in SPECIALISTES.items():
        if "gbm_specialiste" not in regle:
            continue
        if gbm_depuis:
            shutil.copytree(Path(gbm_depuis) / f"gbm_{c}", specialiste / f"gbm_{c}", dirs_exist_ok=True)
        else:
            gbm_sacs.entrainer(annees, specialiste / f"gbm_{c}",
                               **{**GBM_DEFAUT, **regle["gbm_specialiste"], "circuits": [c]})
            (specialiste / f"gbm_{c}" / "hors_sac.csv").unlink(missing_ok=True)
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
            specialistes=True, part_recette=None):
    dossier = Path(dossier)
    reglages = json.loads((dossier / "ensemble.json").read_text())
    if poids_gbm is None:
        poids_gbm = reglages["poids_gbm"]
    if part_recette is None:
        part_recette = reglages.get("part_recette", 0.0)
    gru = mixte.predire(test_dir, out, dossier, sortie_incertitude)
    if part_recette:
        with tempfile.TemporaryDirectory() as temp:
            autre = recette.predire(test_dir, Path(temp) / "recette.csv", dossier / "recette", recalage=1.0)
        gru = (1 - part_recette) * gru + part_recette * autre
    if not poids_gbm:
        return donnees.ecrire_predictions(gru.index.to_numpy(), np.clip(gru.to_numpy(), 0, None), gru.index.to_numpy(),
                                          list(gru.columns), out)
    with tempfile.TemporaryDirectory() as temp:
        gbm = gbm_sacs.predire(test_dir, Path(temp) / "gbm.csv", dossier / "gbm", recalage, demi_vie)
    if not gru.index.equals(gbm.index) or list(gru.columns) != list(gbm.columns):
        raise ValueError("Heures ou circuits différents entre GRU et boosting")
    specialiste = dossier / "specialiste"
    specialistes = specialistes and (specialiste / "specialistes.json").is_file()
    if specialistes:
        gbm = gbm_specialistes(test_dir, specialiste, gbm, recalage, demi_vie)
    melange = (1 - poids_gbm) * gru + poids_gbm * gbm
    if specialistes:
        melange = appliquer_specialistes(test_dir, specialiste, melange, gbm)
    return donnees.ecrire_predictions(melange.index.to_numpy(), np.clip(melange.to_numpy(), 0, None),
                                      melange.index.to_numpy(), list(melange.columns), out)


def gbm_specialistes(test_dir, specialiste, gbm, recalage, demi_vie):
    """Remplacer la part boosting des circuits qui ont leur propre boosting (gbm_<circuit>/)."""
    regles = json.loads((specialiste / "specialistes.json").read_text())
    gbm = gbm.copy()
    with tempfile.TemporaryDirectory() as temp:
        for c, regle in regles.items():
            if "gbm_specialiste" in regle:
                gbm[c] = gbm_sacs.predire(test_dir, Path(temp) / f"{c}.csv", specialiste / f"gbm_{c}",
                                          recalage, demi_vie)[c]
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
