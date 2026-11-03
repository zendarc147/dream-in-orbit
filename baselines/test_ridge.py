"""Vérifier l'alignement et les cibles absentes du modèle Ridge."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from baselines.predict_ridge import predire
from baselines.train_ridge import entrainer


SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "data/prepared/schema.json").read_text())


class RidgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.racine = Path(self.temp.name)
        prepare = self.racine / "data/prepared/annee1"
        prepare.mkdir(parents=True)
        (prepare.parent / "schema.json").write_text(json.dumps(SCHEMA))
        heures = np.arange(6, dtype=np.int64) * 3_600_000
        valeurs = np.zeros((6, 65))
        valeurs[:, 0] = np.arange(6)
        valeurs[:, 1] = 10  # Colonne constante : StandardScaler doit la gérer.
        self.x = pd.DataFrame(valeurs, columns=SCHEMA["feature_columns"])
        self.x.insert(0, "ut_ms", heures)
        self.x.to_csv(prepare / "X.csv", index=False)
        self.y = pd.DataFrame({c: 1 + 2 * np.arange(6, dtype=float)
                               for c in SCHEMA["target_columns"]})
        self.y.loc[2, SCHEMA["target_columns"][0]] = np.nan
        self.y.insert(0, "ut_ms", heures)
        self.y.to_csv(prepare / "y.csv", index=False)

    def test_cible_absente_ignoree_et_prediction_rechargee(self):
        modele_path = entrainer(self.racine)
        sortie = self.racine / "pred.csv"
        resultat = predire(self.racine / "data/prepared/annee1/X.csv", sortie, modele_path)
        pd.testing.assert_frame_equal(resultat, pd.read_csv(sortie), check_exact=False, atol=1e-14)
        variable = SCHEMA["target_columns"][0]
        variables = [c for c in SCHEMA["feature_columns"] if c != "saaf_coverage_fraction"]
        echelle = StandardScaler().fit_transform(self.x[variables])
        connus = self.y[variable].notna().to_numpy()
        attendu = Ridge(alpha=1.0).fit(echelle[connus], self.y.loc[connus, variable]).predict(echelle)
        np.testing.assert_allclose(resultat[variable], attendu, atol=1e-12)
        self.assertEqual(resultat.columns.tolist(), ["ut_ms"] + SCHEMA["target_columns"])
        self.assertTrue(np.isfinite(resultat[SCHEMA["target_columns"]].to_numpy()).all())

    def test_heures_desalignees_refusees(self):
        self.y.loc[0, "ut_ms"] += 3_600_000
        self.y.to_csv(self.racine / "data/prepared/annee1/y.csv", index=False)
        with self.assertRaisesRegex(ValueError, "alignées"):
            entrainer(self.racine)

    def test_couverture_saaf_hors_plage_sans_effet(self):
        self.x["saaf_coverage_fraction"] = 1 - np.arange(6) * 1e-6
        x_path = self.racine / "data/prepared/annee1/X.csv"
        self.x.to_csv(x_path, index=False)
        modele_path = entrainer(self.racine)
        sortie = self.racine / "pred.csv"
        reference = predire(x_path, sortie, modele_path)
        # Une absence de télémétrie ne doit plus provoquer d'extrapolation via sa couverture.
        self.x["saaf_coverage_fraction"] = 0.0
        self.x.to_csv(x_path, index=False)
        resultat = predire(x_path, sortie, modele_path)
        np.testing.assert_allclose(resultat, reference, atol=1e-12, rtol=0)
        modele = json.loads(modele_path.read_text())
        self.assertNotIn("saaf_coverage_fraction", modele["feature_columns"])
        self.assertEqual(modele["input_feature_columns"], SCHEMA["feature_columns"])

    def test_couverture_absente_du_csv_refusee(self):
        modele_path = entrainer(self.racine)
        x_path = self.racine / "x_incomplet.csv"
        self.x.drop(columns="saaf_coverage_fraction").to_csv(x_path, index=False)
        with self.assertRaisesRegex(ValueError, "Colonnes X"):
            predire(x_path, self.racine / "pred.csv", modele_path)

    def test_colonnes_reordonnees_refusees(self):
        modele_path = entrainer(self.racine)
        x_path = self.racine / "x_inverse.csv"
        ordre = self.x.columns.tolist()
        ordre[1], ordre[2] = ordre[2], ordre[1]
        self.x[ordre].to_csv(x_path, index=False)
        with self.assertRaisesRegex(ValueError, "Colonnes X"):
            predire(x_path, self.racine / "pred.csv", modele_path)


if __name__ == "__main__":
    unittest.main()
