"""Porte de régime par marqueurs (baselines/marqueurs.py) sur une série synthétique."""
import unittest

import numpy as np

from baselines import marqueurs


class ListeTest(unittest.TestCase):
    def test_marqueur_explicatif_retenu_et_mode_rare_hors_niveau_de_base(self):
        n = 3000
        y = np.full(n, 0.001)
        palier = np.zeros(n, bool)
        palier[500:1000] = True
        y[palier] = np.where(np.arange(500) % 6 == 0, 0.6, 0.0)  # impulsions de moyenne 0.1
        y[2000:2400] = 0.6  # campagne que rien n'annonce
        presence = {"BON": palier.copy(), "BRUIT": np.random.default_rng(0).random(n) < 0.1}
        defaut, liste = marqueurs.apprendre_liste(y, presence)
        self.assertEqual(defaut, 0.001)  # médiane : la campagne ne relève pas le niveau de base
        self.assertEqual(liste[0][0], "BON")
        self.assertAlmostEqual(liste[0][1], y[palier].mean())
        np.testing.assert_allclose(marqueurs.niveaux(liste, defaut, presence, n)[[0, 600]], [0.001, y[palier].mean()])


if __name__ == "__main__":
    unittest.main()
