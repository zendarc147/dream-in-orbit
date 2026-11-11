"""Tests de l'ensemble GRU + boosting (gru_gbm) avec les poids installés, ou ceux de DIO_POIDS.

Format officiel, heures, 33 colonnes, valeurs finies, reproductibilité, retour au GRU seul,
seule consommation lue = power_init, cibles manquantes dans power_init.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dio import donnees, ensemble

RACINE = Path(__file__).resolve().parent.parent
POIDS = Path(os.environ.get("DIO_POIDS", RACINE / "poids"))
TEST = RACINE / "validation" / "test"
EST_ENSEMBLE = (POIDS / "modele.json").is_file() and json.loads((POIDS / "modele.json").read_text())["modele"] == "gru_gbm"


def predire(test_dir, dossier, **options):
    sortie = Path(dossier) / f"pred_{len(list(Path(dossier).glob('pred_*')))}.csv"
    subprocess.run([sys.executable, "predict.py", "--test_dir", test_dir, "--out", sortie, "--poids", POIDS,
                    "--options", json.dumps(options)], cwd=RACINE, check=True, capture_output=True)
    return pd.read_csv(sortie)


@unittest.skipUnless(EST_ENSEMBLE and TEST.is_dir(), "poids gru_gbm ou validation/ absents")
class EnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.ensemble = predire(TEST, cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_format_heures_colonnes_valeurs_finies(self):
        heures = pd.read_csv(TEST / "heures_a_predire.csv")
        circuits = donnees.charger_schema()["target_columns"]
        self.assertEqual(self.ensemble.columns.tolist(), ["ut_ms"] + circuits)
        self.assertEqual(len(circuits), 33)
        self.assertTrue(self.ensemble.ut_ms.equals(heures.ut_ms))
        valeurs = self.ensemble[circuits].to_numpy()
        self.assertTrue(np.isfinite(valeurs).all() and (valeurs >= 0).all())

    def test_reproductible_et_melange_des_deux_composants(self):
        np.testing.assert_array_equal(predire(TEST, self.temp.name).to_numpy(), self.ensemble.to_numpy())
        gru = predire(TEST, self.temp.name, poids_gbm=0).set_index("ut_ms")
        gbm = predire(TEST, self.temp.name, poids_gbm=1, specialistes=False).set_index("ut_ms")
        w = json.loads((POIDS / "ensemble.json").read_text())["poids_gbm"]
        seul = predire(TEST, self.temp.name, specialistes=False).set_index("ut_ms")
        np.testing.assert_allclose(seul, (1 - w) * gru + w * gbm, atol=1e-9)
        self.assertGreater(np.abs(gru - gbm).to_numpy().max(), 1e-3)

    @unittest.skipUnless((POIDS / "specialiste" / "specialistes.json").is_file(), "pas de spécialistes")
    def test_specialistes_ne_changent_que_leurs_colonnes(self):
        seul = predire(TEST, self.temp.name, specialistes=False).set_index("ut_ms")
        avec = self.ensemble.set_index("ut_ms")
        regles = json.loads((POIDS / "specialiste" / "specialistes.json").read_text())
        autres = [c for c in avec.columns if c not in regles]
        self.assertEqual(len(autres), 31)
        np.testing.assert_array_equal(avec[autres].to_numpy(), seul[autres].to_numpy())
        for c in regles:
            self.assertGreater(np.abs(avec[c] - seul[c]).max(), 1e-3)

    def test_consommation_future_refusee_et_cibles_manquantes(self):
        with tempfile.TemporaryDirectory() as temp:
            copie = Path(temp) / "test"
            shutil.copytree(TEST, copie)
            power = pd.read_csv(TEST / "power_init.csv")
            premiere = pd.read_csv(TEST / "heures_a_predire.csv").ut_ms.iloc[0]
            futur = power.tail(1).assign(ut_ms=premiere + 60_000)
            pd.concat([power, futur]).to_csv(copie / "power_init.csv", index=False)
            with self.assertRaisesRegex(ValueError, "empiète"):
                ensemble.predire(copie, Path(temp) / "p.csv", POIDS)
            # Un circuit sans aucune mesure et un jour entier manquant dans power_init.
            trous = power.copy()
            trous["NPWD2551"] = np.nan
            trous.loc[trous.ut_ms < trous.ut_ms.iloc[0] + 86_400_000, trous.columns[1:]] = np.nan
            trous.to_csv(copie / "power_init.csv", index=False)
            table = ensemble.predire(copie, Path(temp) / "p.csv", POIDS)
            self.assertTrue(np.isfinite(table.to_numpy()).all())
            self.assertEqual(len(table), len(self.ensemble))


if __name__ == "__main__":
    unittest.main()
