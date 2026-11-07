"""Tests des variables de commandes (commandes.py)."""
import unittest

import numpy as np
import pandas as pd

from dio import commandes

H = commandes.H


def table(*lignes):
    return pd.DataFrame({"ut_ms": [t for t, _ in lignes], "subsystem": [c for _, c in lignes]})


class TestCommandes(unittest.TestCase):
    def test_famille(self):
        self.assertEqual(commandes.famille("ATTTF321A"), "ATTTF321")
        self.assertEqual(commandes.famille("AOOOF22R1"), "AOOOF22")
        self.assertEqual(commandes.famille("ATTT305C"), "ATTT305")

    def test_impulsions_etats_decalage(self):
        heures = np.arange(6) * H
        t = table((1 * H + 10, "ATTTF321A"), (3 * H + 10, "ATTTF321D"))
        codes = ["ATTTF321A", "ATTTF321D"]
        np.testing.assert_array_equal(commandes.impulsions(t, heures, codes)[:, 0], [0, 1, 0, 0, 0, 0])
        # Allumé de l'heure 1 à l'heure 2, éteint ensuite par la commande D de la même famille.
        np.testing.assert_array_equal(commandes.etats(t, heures, codes)[:, 0], [0, 1, 1, 0, 0, 0])
        # Délai +1 : on lit l'heure suivante (commande vue une heure à l'avance).
        np.testing.assert_array_equal(commandes.decaler(np.arange(6)[:, None], 1)[:, 0], [1, 2, 3, 4, 5, 5])

    def test_apprendre_choisit_le_delai(self):
        heures = np.arange(200) * H
        moments = np.arange(10, 190, 20)
        t = table(*[(int(m) * H, "AOOOF01D0") for m in moments])
        y = np.zeros((200, 1))
        y[moments - 2, 0] = 1.0  # le circuit réagit deux heures avant la commande
        voc = commandes.apprendre([(t, heures, y)], ["NPWD2881"], {"prefixes": ["AOOO"], "types": ["impulsion"]})
        self.assertEqual(voc["codes"], [])  # moins de MIN_OCCURRENCES : ignorée
        t = table(*[(int(m) * H, "AOOOF01D0") for m in moments] * 3)
        voc = commandes.apprendre([(t, heures, y)], ["NPWD2881"], {"prefixes": ["AOOO"], "types": ["impulsion"]})
        self.assertEqual(voc["delais"]["impulsion"], [2])

    def test_atmb_et_masque(self):
        heures = np.arange(5) * H
        t = table((1 * H, "ATMB228K"), (3 * H, "ATMB057K"), (2 * H, "ATTT305C"))
        np.testing.assert_array_equal(commandes.consigne_atmb(t, heures), [0, 228, 228, 57, 57])
        np.testing.assert_array_equal(commandes.fenetre_masque(t, heures, ["ATTT305C"], 2), [0, 0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
