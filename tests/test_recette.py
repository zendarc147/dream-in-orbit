"""Tests du GRU recette (dio/recette.py) sur données synthétiques : forme, réinjection, calendrier."""
import unittest

import numpy as np
import torch

from dio import recette


class TestRecette(unittest.TestCase):
    def test_calendrier_de_reinjection(self):
        config = {**recette.CONFIG_DEFAUT, "epoques": 80, "epoques_verite": 15}
        p = [recette.probabilite_verite(e, config) for e in range(80)]
        self.assertEqual(p[:15], [1.0] * 15)
        self.assertEqual(p[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(p, p[1:])))

    def test_standardisation_sans_bornes_min_max(self):
        f = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
        stats = recette.statistiques([(f, np.zeros((3, 1)))])
        self.assertEqual(stats["x_ecart"][1], 1.0)  # écart-type nul remplacé par 1
        z = recette.standardiser(np.array([[1000.0, 5.0]]), stats)
        self.assertEqual(z[0, 0], recette.BORNE)  # borné à ±10 après standardisation seulement

    def test_entrainement_court_et_avancer(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(200, 4))
        y = np.abs(rng.normal(size=(200, 3)))
        y[10:20, 1] = np.nan
        for reinjection in (True, False):
            config = {**recette.CONFIG_DEFAUT, "taille": 8, "epoques": 2, "epoques_verite": 1, "flux": 2,
                      "segment_h": 24, "reinjection": reinjection}
            stats = recette.statistiques([(f, y)])
            modele = recette.entrainer_un([(f, y)], stats, config, 0)
            x = torch.as_tensor(recette.standardiser(f, stats)[None], dtype=torch.float32)
            with torch.no_grad():
                sortie, h, _ = modele.avancer(x, torch.zeros(1, 8), torch.zeros(1, 3))
            self.assertEqual(tuple(sortie.shape), (1, 200, 3))
            self.assertTrue(torch.isfinite(sortie).all())


if __name__ == "__main__":
    unittest.main()
