"""Conventions exactes de dio/caracteristiques.py (recette complète S)."""
import unittest

import numpy as np
import pandas as pd

from dio import caracteristiques as C
from dio import commandes

H = C.H


class Conventions(unittest.TestCase):
    def test_familles(self):
        attendu = {"AAAAF23G1": "AAAAF23_1", "AAAAF23A2": "AAAAF23_2", "AOOOF01D0": "AOOOF01_0",
                   "AOOOF01D1": "AOOOF01_1", "AACFE03A": "AACFE03_", "ASEQ4200": "ASEQ4200"}
        for code, famille in attendu.items():
            self.assertEqual(commandes.famille_variante(code), famille)

    def test_vocabulaire_par_periode_union_sans_point(self):
        p1 = {"dmop": pd.DataFrame({"ut_ms": range(100), "subsystem": ["AAAAF23G1"] * 50 + ["MOCS.0001"] * 50}),
              "ftl": pd.DataFrame({"type": ["EARTH"]})}
        p2 = {"dmop": pd.DataFrame({"ut_ms": range(100), "subsystem": ["AAAAF23A1"] * 50 + ["AAAAF23G1"] * 50}),
              "ftl": pd.DataFrame({"type": ["SLEW"]})}
        v = C.apprendre([p1, p2])
        self.assertEqual(v["familles"], ["AAAAF23_1"])          # MOCS.0001 exclu (point)
        self.assertEqual(v["changements"], ["AAAAF23_1"])       # deux variantes ≥ 50 dans p2 seulement
        self.assertEqual(v["prefixes"], ["AAAA", "MOCS"])
        self.assertEqual(v["ftl"], ["EARTH", "SLEW"])
        self.assertEqual(C.apprendre([p1])["changements"], [])

    def test_recence_quantifiee(self):
        heures = np.arange(5) * H
        r = C.recence([int(1.5 * H)], heures)                   # occurrence dans l'heure 1
        np.testing.assert_allclose(r, np.log1p([720, 720, 1, 2, 3]))

    def test_changements_zero_dans_l_heure(self):
        heures = np.arange(6) * H
        depuis, jusqua = C.distances_changement([int(2.5 * H)], heures)
        np.testing.assert_allclose(depuis, np.log1p([720, 720, 0, 1, 2, 3]))
        np.testing.assert_allclose(jusqua, np.log1p([2, 1, 0, 720, 720, 720]))

    def test_commandes_changements(self):
        dmop = pd.DataFrame({"ut_ms": [0, H, 3 * H + 10, 4 * H], "subsystem": ["AAAAF23G1", "AAAAF23G1", "AAAAF23Z1",
                                                                                 "AAAAF23Z1"]})
        v = {"familles": ["AAAAF23_1"], "prefixes": ["AAAA"], "changements": ["AAAAF23_1"]}
        t = C.commandes_horaires(dmop, np.arange(6) * H, v)
        np.testing.assert_allclose(t["cmd_chg_AAAAF23_1_depuis"], np.log1p([720, 720, 720, 0, 1, 2]))
        np.testing.assert_allclose(t["cmd_fam_AAAAF23_1_futur1h"], [1, 0, 1, 1, 0, 0])
        np.testing.assert_allclose(t["cmd_fam_AAAAF23_1_futur6h"], [3, 2, 2, 1, 0, 0])
        np.testing.assert_allclose(t["cmd_total_n"], [1, 1, 0, 1, 1, 0])

    def test_saaf_std_et_moyenne_arithmetique(self):
        heures = np.arange(4) * H
        saaf = pd.DataFrame({"ut_ms": [0, 10, H + 5, 3 * H], "sa": [0.0, 90.0, 30.0, 60.0], "sx": 0.0, "sy": 0.0,
                             "sz": 0.0})
        s = C.saaf(saaf, heures)
        np.testing.assert_allclose(s["saaf_sa_cos"][:2], [0.5, np.cos(np.deg2rad(30))])
        np.testing.assert_allclose(s["saaf_sa_std"], [np.std([0, 90], ddof=1), 0, 0, 0])
        np.testing.assert_allclose(s["saaf_sa_mean"], [45, 30, 45, 60])          # heure 2 interpolée

    def test_ltdata_debut_d_heure(self):
        lt = pd.DataFrame({"ut_ms": [0, 24 * H], "x": [0.0, 24.0]})
        np.testing.assert_allclose(C.ltdata(lt, np.arange(3) * H)["ltdata_x"], [0, 1, 2])

    def test_altitude_moyenne_des_log(self):
        evtf = pd.DataFrame({"ut_ms": [0, H], "description": ["PERICENTRE_PASSAGE", "APOCENTRE_PASSAGE"]})
        la = C.log_altitude(evtf, np.array([0]))
        attendu = np.mean(np.log(330) + (np.arange(4) + 0.5) / 4 * (np.log(10100) - np.log(330)))
        self.assertAlmostEqual(la[0], attendu)

    def test_historiques_umbra_seule(self):
        t = pd.DataFrame({"ev_mar_umbra": [0, 0.5, 0, 0], "ev_mar_penumbra": [1.0, 0, 0, 0],
                          **{f"saaf_{a}_cos": [1.0] * 4 for a in C.ANGLES}, "ltdata_solarconstantmars": [600.0] * 4,
                          "ftl_comms": [1.0, 0, 0, 0]})
        th = C.thermique(t)
        np.testing.assert_allclose(th["th_umbra_somme7h"], [0, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(th["th_umbra_depuis"], np.log1p([720, 0, 1, 2]))
        np.testing.assert_allclose(th["flux_incident"], [0.5, 0.5, 1, 1])
        np.testing.assert_allclose(th["th_incident_moy6h"], [0.5, 0.5, 2 / 3, 0.75])
        np.testing.assert_allclose(th["flux_sa_plus"], th["flux_incident"])

    def test_paires_union(self):
        dmop = pd.DataFrame({"ut_ms": [0, H // 2, H // 4, 3 * H // 4],
                             "subsystem": ["PENS.01", "PENE.01", "PENS.02", "PENE.02"]}).sort_values("ut_ms")
        np.testing.assert_allclose(C.periode_paire(dmop, np.array([0]), "PENS", "PENE"), [0.75])

    def test_colonnes_gru(self):
        noms = ["saaf_sa_cos", "ltdata_earthmars_km", "cmd_fam_X_n", "cmd_fam_X_recence", "cmd_fam_X_futur1h",
                "cmd_pre_AAAA_n", "cmd_chg_X_depuis", "flux_incident", "th_umbra_moy168h", "ftl_EARTH_dec+1h",
                "geo_log_altitude", "ftl_EARTH"]
        t = pd.DataFrame(columns=noms)
        self.assertEqual(C.colonnes_gru(t), ["saaf_sa_cos", "cmd_fam_X_n", "geo_log_altitude", "ftl_EARTH"])
        self.assertNotIn("ltdata_earthmars_km", C.colonnes_gbm(t))


if __name__ == "__main__":
    unittest.main()
