"""Chargement commun des données horaires : années préparées et dossiers de test.

Toutes les entrées X viennent de prepare.py (schéma appris sur annee1, contextes
seulement). Les courants restent à l'échelle brute, avec NaN là où ils sont inconnus.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import prepare

H = 3_600_000
RACINE = Path(__file__).resolve().parent.parent
PREPARE = RACINE / "data" / "prepared"


def charger_schema(chemin=None):
    return json.loads(Path(chemin or PREPARE / "schema.json").read_text())


def charger_annee(annee, schema=None):
    """Retourner (heures, X DataFrame, Y ndarray avec NaN) pour une période préparée."""
    schema = schema or charger_schema()
    dossier = PREPARE / annee
    brut = Path(annee)
    if not (RACINE / "data" / annee).is_dir() and (brut / "power.csv").is_file():
        # Période hors du kit (validation chronologique) : préparée à la volée avec le schéma d'annee1.
        y = prepare.cibles_horaires(brut / "power.csv", schema["target_columns"])
        x = prepare.remplir_entrees(prepare.construire_entrees(prepare.lire_contextes(brut), y.index, schema), schema)
        return x.index.to_numpy(), x, y.to_numpy(dtype=np.float64)
    if not (dossier / "X.csv").is_file():
        # Préparer à la demande à partir du kit brut (power.csv + contextes).
        prepare.preparer_periodes(RACINE / "data", PREPARE)
    x = pd.read_csv(dossier / "X.csv", index_col="ut_ms")
    y = pd.read_csv(dossier / "y.csv", index_col="ut_ms")
    if x.columns.tolist() != schema["feature_columns"] or y.columns.tolist() != schema["target_columns"]:
        raise ValueError(f"{annee} : colonnes différentes du schéma")
    if not x.index.equals(y.index):
        raise ValueError(f"{annee} : heures X et y non alignées")
    return x.index.to_numpy(), x, y.to_numpy(dtype=np.float64)


def lire_dossier_test(test_dir, schema=None):
    """Préparer un dossier de test sur une grille continue : 7 jours initiaux + heures à prédire.

    Retourne un dict avec heures (grille complète), X (DataFrame rempli),
    y_init (ndarray, NaN hors des 7 jours), n_init (nombre d'heures initiales)
    et heures_a_predire.
    """
    schema = schema or charger_schema()
    dossier = Path(test_dir)
    a_predire = pd.read_csv(dossier / "heures_a_predire.csv")
    if a_predire.columns.tolist() != ["ut_ms"]:
        raise ValueError("heures_a_predire : seule la colonne ut_ms est attendue")
    a_predire = prepare.verifier_heures(a_predire.ut_ms)
    y_init = prepare.cibles_horaires(dossier / "power_init.csv", schema["target_columns"])
    if y_init.index[-1] >= a_predire[0]:
        raise ValueError("power_init empiète sur les heures à prédire")
    heures = prepare.verifier_heures(range(int(y_init.index[0]), int(a_predire[-1]) + H, H))
    tables = prepare.lire_contextes(dossier)
    x = prepare.construire_entrees(tables, heures, schema)
    x = prepare.remplir_entrees(x, schema)
    y = np.full((len(heures), len(schema["target_columns"])), np.nan)
    y[: len(y_init)] = y_init.to_numpy(dtype=np.float64)
    return {
        "heures": heures.to_numpy(),
        "X": x,
        "y_init": y,
        "n_init": len(y_init),
        "heures_a_predire": a_predire.to_numpy(),
        "dmop": tables["dmop"],
        "tables": tables,
    }


def lire_dmop(annee):
    """Table dmop brute d'une période du kit (pour l'état des sous-systèmes)."""
    table = pd.read_csv(_dossier(annee) / "dmop.csv")
    if table.columns.tolist() != prepare.COLONNES["dmop"]:
        raise ValueError(f"{annee}/dmop.csv : colonnes inattendues")
    return table


def lire_contextes(annee):
    """Les cinq contextes bruts d'une période du kit (dmop, ftl, evtf, saaf, ltdata)."""
    return prepare.lire_contextes(_dossier(annee))


def _dossier(annee):
    return RACINE / "data" / annee if (RACINE / "data" / annee).is_dir() else Path(annee)


def ecrire_predictions(heures, valeurs, heures_a_predire, circuits, out):
    """Écrire une ligne par heure demandée, sans valeur manquante."""
    table = pd.DataFrame(valeurs, index=pd.Index(heures, name="ut_ms"), columns=circuits)
    table = table.reindex(heures_a_predire)
    if table.isna().any().any() or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Prédictions manquantes ou non finies")
    table.to_csv(out)
    print(f"{len(table)} heures prédites dans {out}")
    return table
