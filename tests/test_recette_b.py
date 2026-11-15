"""Tests de la recette B (baselines/recette_b.py) : conventions de la recette et contrôle des entrées."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs, recette_b as rb

H = rb.H
RACINE = Path(__file__).resolve().parent.parent
CONTROLES = RACINE / "controles" / "recette_b"


class TestConventions(unittest.TestCase):
    def test_separer_exemples_de_la_recette(self):
        self.assertEqual(rb.separer("AAAAF20C1"), ("AAAAF20_1", "C"))
        self.assertEqual(rb.separer("AOOOF01D0"), ("AOOOF01_0", "D"))
        self.assertEqual(rb.separer("AOOOF100C"), ("AOOOF100_", "C"))
        self.assertEqual(rb.separer("ATTT305C"), ("ATTT305_", "C"))
        self.assertEqual(rb.separer("MOCS.0000003738"), ("MOCS.0000003738", ""))

    def test_fraction_union_et_appariement(self):
        g = np.arange(3) * H
        # [0.5h, 1.5h) et [1h, 2h) fusionnés ; [2.5h, 2.5h) vide.
        np.testing.assert_allclose(rb.fraction(g, [H // 2, H, 5 * H // 2], [3 * H // 2, 2 * H, 5 * H // 2]),
                                   [0.5, 1.0, 0.0])
        # START, START (remplace), END, END orphelin.
        s, e = rb.apparier(np.array([0, 10, 20, 30]), np.array([1, 1, 0, 0], bool), np.array([0, 0, 1, 1], bool))
        np.testing.assert_array_equal(s, [10])
        np.testing.assert_array_equal(e, [20])

    def test_recence_lue_avant_mise_a_jour(self):
        # Commande à 12:20 (heure 12) : à 12:00 la récence n'est pas nulle (plafond), à 13:00 elle vaut 1.
        comptes = np.zeros((15, 1))
        comptes[12] = 1
        r = np.expm1(rb.recence(comptes)[:, 0].astype(float))
        self.assertAlmostEqual(r[12], 720, places=2)
        self.assertAlmostEqual(r[13], 1, places=5)
        self.assertAlmostEqual(r[14], 2, places=5)

    def test_phase_avant_et_apres_les_pericentres(self):
        evtf = pd.DataFrame({"ut_ms": [10 * H, 17 * H], "description": ["PERICENTRE_PASSAGE_1", "PERICENTRE_PASSAGE_2"]})
        o = rb.orbite(evtf, np.array([5, 12, 20]) * H)
        np.testing.assert_allclose(o["ev_since_peri_h"], [0, 2.5, 3.5])
        np.testing.assert_allclose(o["ev_to_peri_h"], [4.5, 4.5, 0])
        np.testing.assert_allclose(o["ev_orbit_period_h"], [7, 7, 7])

    def test_reglages(self):
        self.assertEqual(rb.D3L, gbm_sacs.REGLAGES["D3L"])
        self.assertEqual(rb.PARAMS_B, json.loads((CONTROLES / "parametres_reference.json").read_text()))
        n = 1003
        np.testing.assert_array_equal(rb.bornes(n), np.linspace(0, n, 6).astype(int))
        for i, (lignes, bloc) in enumerate(gbm_sacs.sacs_temporels(n, 5, 0)):
            b = rb.bornes(n)
            np.testing.assert_array_equal(bloc, np.arange(b[i], b[i + 1]))

    def test_ajustement_identique_au_livre(self):
        """Cellule d3l, masque propre : mêmes arbres que gbm_sacs._ajuster (même graine, mêmes lignes)."""
        hasard = np.random.default_rng(0)
        x = hasard.normal(size=(600, 4))
        y = x[:, 0] + hasard.normal(scale=0.1, size=600)
        y[::7] = np.nan
        reglage = {**rb.REGLAGES["d3l"], "max_iter": 20}
        gbm_sacs.REGLAGES["_test"] = {k: v for k, v in reglage.items() if k not in ("loss", "random_state")}
        try:
            modeles = rb.ajuster(x, y, np.isfinite(y), reglage, n_jobs=1)
            for (lignes, _), m in zip(gbm_sacs.sacs_temporels(600, 5, 0), modeles):
                ref = gbm_sacs._ajuster(x, y, lignes, "_test", 0)
                np.testing.assert_array_equal(ref.predict(x), m.predict(x))
        finally:
            del gbm_sacs.REGLAGES["_test"]


@unittest.skipUnless((RACINE / "data/annee1/power.csv").exists(), "données absentes")
class TestControle(unittest.TestCase):
    def test_entrees_annee1_vocabulaire_de_controle(self):
        vocab = json.loads((CONTROLES / "vocabulaire_reference.json").read_text())
        ref = json.loads((CONTROLES / "controle_reference.json").read_text())
        g, x, _, _ = rb.periode(RACINE / "data/annee1", vocab)
        self.assertEqual(list(x.columns), json.loads((CONTROLES / "colonnes_reference.json").read_text()))
        self.assertEqual((int(g[0]), int(g[-1]), x.shape), (ref["grid_first"], ref["grid_last"], tuple(ref["shape"])))
        self.assertEqual(x.dtypes.unique().tolist(), [np.float32])
        echantillon = pd.read_csv(CONTROLES / "controle_entrees_annee1.csv", index_col="ut_ms")
        np.testing.assert_allclose(x.loc[echantillon.index, echantillon.columns].to_numpy(np.float64),
                                   echantillon.to_numpy(np.float64), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
