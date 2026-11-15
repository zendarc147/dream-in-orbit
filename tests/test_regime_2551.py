"""Modèle en deux régimes de NPWD2551 et décomposition de l'erreur (docs/JOURNAL_2551.md, programme 3)."""
import unittest

import numpy as np

from baselines import regime

J = regime.JOUR


class TestRegime(unittest.TestCase):

    def test_etiquettes_journalieres(self):
        heures = np.arange(72) * regime.H
        y = np.r_[np.full(24, 0.5), np.full(24, 0.01), np.full(24, np.nan)]
        np.testing.assert_array_equal(regime.etiquettes(heures, y)[[0, 30, 60]], [1.0, 0.0, np.nan])

    def test_lissage(self):
        heures = np.arange(20 * 24) * regime.H
        p = np.repeat(np.r_[np.full(10, 0.9), [0.2], np.full(9, 0.9)], 24)
        brut = regime.lisser(p, heures, 1.0, 0.4)
        np.testing.assert_allclose(brut, np.clip(p, regime.P_MIN, 1 - regime.P_MIN))  # renouvellement 1 : brut
        lisse = regime.lisser(p, heures, 0.02, 0.4)
        self.assertGreater(lisse[10 * 24], 0.5)  # le jour isolé à 0.2 est absorbé par la persistance
        obs = regime.lisser(p, heures, 0.02, 0.4, {0: 0})
        self.assertLess(obs[0], 0.1)  # jour observé (power_init) fixé


class TestDecomposition(unittest.TestCase):

    def test_parts_somment_a_la_mse(self):
        import pandas as pd
        from outils import bilan_2551 as b
        h = np.arange(60 * 24) * b.H
        y = pd.Series(np.where((h // J) % 30 < 15, 0.6, 0.01) + 0.1 * np.sin(h / b.H), index=h)
        b._CACHE["y", "test"] = y
        p = np.roll(y.to_numpy(), 48) + 0.05
        d = b.decomposer(p, "test")
        self.assertAlmostEqual(sum(d["parts"].values()), d["mse"], places=12)
        self.assertGreater(d["parts"]["bascule"], 0)
        del b._CACHE["y", "test"]


if __name__ == "__main__":
    unittest.main()
