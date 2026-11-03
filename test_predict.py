"""Tests du world model : variables v2 et bout en bout de predict.py avec les poids commités."""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import variables

RACINE = Path(__file__).resolve().parent
VALIDATION = RACINE / "validation"


class VariablesTest(unittest.TestCase):
    def test_fenetres_passees_et_futures_tronquees(self):
        passe, futur = variables.sommes_glissantes(np.array([1., 0., 0., 4.]), 3)
        np.testing.assert_allclose(passe, [1, 0.5, 1 / 3, 4 / 3])
        np.testing.assert_allclose(futur, [1 / 3, 4 / 3, 2, 4])

    def test_heures_depuis_et_jusqua_plafonnees(self):
        depuis, jusqua = variables.heures_depuis_jusqua(np.array([0, 1, 0, 0, 1, 0], bool))
        np.testing.assert_allclose(depuis, [variables.PLAFOND_H, 0, 1, 2, 0, 1])
        np.testing.assert_allclose(jusqua, [1, 0, 2, 1, 0, variables.PLAFOND_H])

    def test_ablation_retire_tout_le_groupe(self):
        x = pd.read_csv(RACINE / "data/prepared/annee1/X.csv", index_col="ut_ms", nrows=300)
        _, noms = variables.construire(x, "v2", ["eclipses"])
        self.assertFalse([n for n in noms if "UMBRA" in n or "OCC_MARS" in n or "eclipse" in n])
        _, tous = variables.construire(x, "v2")
        self.assertGreater(len(tous), len(noms))


@unittest.skipUnless((RACINE / "poids/modele.json").is_file() and (VALIDATION / "test").is_dir(),
                     "poids/ ou validation/ absent")
class BoutEnBoutTest(unittest.TestCase):
    def test_predict_format_officiel_et_score(self):
        with tempfile.TemporaryDirectory() as dossier:
            pred = Path(dossier) / "pred.csv"
            subprocess.run([sys.executable, "predict.py", "--test_dir", VALIDATION / "test", "--out", pred],
                           cwd=RACINE, check=True, capture_output=True)
            table = pd.read_csv(pred)
            heures = pd.read_csv(VALIDATION / "test/heures_a_predire.csv")
            verite = pd.read_csv(VALIDATION / "verite_horaire.csv", nrows=1)
            self.assertEqual(table.columns.tolist(), verite.columns.tolist())
            self.assertTrue(table.ut_ms.equals(heures.ut_ms))
            self.assertFalse(table.isna().any().any())
            self.assertTrue((table.drop(columns="ut_ms") >= 0).all().all())
            sortie = subprocess.run([sys.executable, "evaluate.py", "--pred", pred,
                                     "--verite", VALIDATION / "verite_horaire.csv",
                                     "--heures", VALIDATION / "test/heures_a_predire.csv"],
                                    cwd=RACINE, check=True, capture_output=True, text=True).stdout
            rmse = float(re.search(r"RMSE = ([0-9.]+)", sortie).group(1))
            # Les poids finaux ont vu annee2 : le score doit au moins battre la constante.
            self.assertLess(rmse, 0.1058)


if __name__ == "__main__":
    unittest.main()
