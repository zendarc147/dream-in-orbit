"""Garanties de métrique et d'isolation des courants pour les départs décalés."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from outils import diagnostic_temporel as diagnostic


class TestDiagnosticTemporel(unittest.TestCase):
    def test_contributions_et_nan(self):
        y = pd.DataFrame({"a": [1., np.nan, 3.], "b": [2., 4., 6.]})
        p = pd.DataFrame({"a": [2., 100., 3.], "b": [2., 5., 6.]})
        resultat = diagnostic.mesures(y.to_numpy(), p.to_numpy())
        self.assertEqual(resultat["n"], 5)
        self.assertAlmostEqual(resultat["rmse"] ** 2, 2 / 5)
        self.assertAlmostEqual(diagnostic.circuits(y, p).contribution_mse.sum(), 2 / 5)

    def test_reinitialisation_ne_transmet_que_sept_jours(self):
        heures = np.arange(1000) * diagnostic.donnees.H
        x = pd.DataFrame({"contexte": np.ones(1000)}, index=heures)
        y = np.arange(1000, dtype=float)[:, None]
        y[30] = np.nan
        appels = []

        def simuler(dossier, contexte, courants, n_init, dmop):
            self.assertEqual(n_init, 168)
            debut = int(contexte.index[0] // diagnostic.donnees.H)
            np.testing.assert_equal(courants[:168], y[debut:debut+168])
            self.assertTrue(np.isnan(courants[168:]).all())
            appels.append(debut)
            return np.zeros_like(courants), np.zeros_like(courants)

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "directe").mkdir()
            (run / "directe/membres.json").write_text(json.dumps([{}, {}]))
            with patch.object(diagnostic.donnees, "charger_annee", return_value=(heures, x, y)), \
                 patch.object(diagnostic.donnees, "lire_dmop", return_value=pd.DataFrame()), \
                 patch.object(diagnostic.torch, "load", return_value={"annees": ["annee1"]}), \
                 patch.object(diagnostic.monde, "simuler", side_effect=simuler):
                diagnostic.reinitialisations(run, "directe", run, jours_depart=(0, 10))
            self.assertEqual(appels, [0, 0, 240, 240])
            with patch.object(diagnostic.donnees, "charger_annee", return_value=(heures, x, y)), \
                 patch.object(diagnostic.donnees, "lire_dmop", return_value=pd.DataFrame()), \
                 patch.object(diagnostic.torch, "load", return_value={"annees": ["annee2"]}):
                with self.assertRaisesRegex(ValueError, "indépendants"):
                    diagnostic.reinitialisations(run, "directe", run)


if __name__ == "__main__":
    unittest.main()
