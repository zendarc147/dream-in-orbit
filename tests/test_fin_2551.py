"""Programme 5 de NPWD2551 : cibles à la résolution brute et variables ajoutées au boosting."""
import re
import unittest

import numpy as np
import pandas as pd

from baselines import fin_2551, gbm_sacs

H = 3_600_000


class Cibles(unittest.TestCase):
    def test_decodage_des_lignes(self):
        niveaux = np.array([0.0005, 0.205, 0.41, 0.63, 0.835, 1.045, 1.675, 1.88])
        e = fin_2551.decoder(niveaux)
        self.assertEqual(e["A"].tolist(), [0, 1, 2, 0, 1, 0, 0, 1])
        self.assertEqual(e["B"].tolist(), [0, 0, 0, 1, 1, 0, 1, 1])
        self.assertEqual(e["C"].tolist(), [0, 0, 0, 0, 0, 1, 1, 1])

    def test_somme_des_lignes_et_grille_de_10_min(self):
        hasard = np.random.default_rng(0)
        t = np.arange(0, 3 * H, 30_000, dtype=np.int64)  # 20 mesures par case de 10 min
        v = hasard.choice([0.0, 0.205, 0.63, 1.045, 1.88], len(t))
        lignes, dix = fin_2551.cibles(t, v)
        horaire = pd.Series(v).groupby(t // H * H).mean()
        np.testing.assert_allclose(lignes.sum(axis=1).to_numpy(), horaire.to_numpy(), atol=1e-12)
        t10 = fin_2551.grille(horaire.index.to_numpy())
        self.assertEqual(len(t10), 18)
        np.testing.assert_allclose(fin_2551.vers_heures(dix.reindex(t10).to_numpy()), horaire.to_numpy(), atol=1e-12)

    def test_vers_heures_ignore_les_cases_inconnues(self):
        v = np.array([1, np.nan, 3, np.nan, np.nan, np.nan] + [np.nan] * 6, dtype=float)
        self.assertEqual(fin_2551.vers_heures(v)[0], 2.0)
        self.assertTrue(np.isnan(fin_2551.vers_heures(v)[1]))

    def test_minutes_depuis_et_jusqua(self):
        t = np.array([0, 600_000, 1_200_000], dtype=np.int64)
        depuis, jusqua = fin_2551._depuis_jusqua([300_000], t)
        self.assertEqual(depuis.tolist(), [fin_2551.PLAFOND_MIN, 5.0, 15.0])
        self.assertEqual(jusqua.tolist(), [5.0, fin_2551.PLAFOND_MIN, fin_2551.PLAFOND_MIN])


class Ajouts(unittest.TestCase):
    def test_saison(self):
        x = pd.DataFrame({"ltdata_eclipseduration_min": [0, 0, 5, 7, 3, 0, 0]}, index=np.arange(7) * H)
        s = gbm_sacs.saison(x)
        np.testing.assert_allclose(s.saison_jusqua_j * 24, [2, 1, 0, 0, 0, gbm_sacs.PLAFOND_SAISON_H] * 1
                                   + [gbm_sacs.PLAFOND_SAISON_H])
        np.testing.assert_allclose(s.saison_depuis_j.to_numpy()[2:] * 24, [0, 0, 0, 1, 2])
        np.testing.assert_allclose(s.saison_age_j.to_numpy()[2:5] * 24, [0, 1, 2])
        np.testing.assert_allclose(s.saison_reste_j.to_numpy()[2:5] * 24, [2, 1, 0])
        self.assertTrue(s.saison_age_j.iloc[[0, 1, 5, 6]].isna().all())

    def test_filtres_lents_survivent_au_retrait_ema(self):
        n = 50
        x = pd.DataFrame({"evtf_MAR_UMBRA_minutes": np.r_[np.zeros(25), np.full(25, 30.0)],
                          "evtf_MAR_PENUMBRA_minutes": 0.0, "ltdata_solarconstantmars": 500.0,
                          **{f"saaf_{a}_mean": 45.0 for a in ("sa", "sx", "sy", "sz")}}, index=np.arange(n) * H)
        t = gbm_sacs.filtres_lents(x, (6, 24), "th_")
        motif = "|".join(gbm_sacs.RETRAITS["ema"])
        self.assertTrue(len(t.columns) and not any(re.fullmatch(motif, c) for c in t.columns))
        ombre = t["th_ombre_tau6"].to_numpy()  # causal : nul avant l'ombre, croissant ensuite, sous le palier
        self.assertTrue((ombre[:25] == 0).all() and (np.diff(ombre[24:]) > 0).all() and ombre[-1] < 0.5)


if __name__ == "__main__":
    unittest.main()
