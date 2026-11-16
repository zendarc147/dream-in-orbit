"""Tests du boosting à sacs temporels : découpage, vocabulaire, absence de fuite de consommation."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs
from dio import caracteristiques, commandes, donnees, variables

H = donnees.H
RACINE = Path(__file__).resolve().parent.parent
VALIDATION = RACINE / "validation" / "test"


def table(*lignes):
    return pd.DataFrame({"ut_ms": [t for t, _ in lignes], "subsystem": [c for _, c in lignes]})


class TestSacs(unittest.TestCase):
    def test_blocs_contigus_et_embargo(self):
        decoupe = gbm_sacs.sacs_temporels(1000, 5, 30)
        blocs = np.concatenate([bloc for _, bloc in decoupe])
        np.testing.assert_array_equal(blocs, np.arange(1000))  # chaque heure écartée une fois
        for lignes, bloc in decoupe:
            self.assertTrue((np.diff(bloc) == 1).all())
            distance = np.abs(lignes[:, None] - bloc[None, :]).min()
            self.assertGreater(distance, 30)  # marge de 30 h de chaque côté du bloc
        lignes, bloc = gbm_sacs.sacs_temporels(10, 1, 30)[0]
        self.assertEqual((len(lignes), len(bloc)), (10, 0))

    def test_familles_apprises_sur_l_entrainement_seul(self):
        entrainement = table(*[(i * H, "ATTTF321A") for i in range(15)], *[(i * H, "ATTTF321D") for i in range(5)],
                             *[(i * H, "AOOOF01D0") for i in range(19)])
        familles = variables.apprendre_familles([entrainement])
        self.assertEqual(familles, ["ATTTF321"])  # lettres fusionnées ; AOOOF01 sous le seuil
        test = table((0, "ATTTF321P"), (0, "AXXX999A"), (2 * H + 5, "ATTTF321A"))
        comptes = variables.comptes_familles(test, np.arange(3) * H, familles)
        self.assertEqual(comptes.columns.tolist(), ["dmopfam_ATTTF321_count"])
        np.testing.assert_array_equal(comptes.iloc[:, 0], [1, 0, 1])


class TestFamillesVariante(unittest.TestCase):
    def test_exemples(self):
        attendus = {"AAAAF23G1": "AAAAF23_1", "AAAAF23A2": "AAAAF23_2", "AOOOF01D0": "AOOOF01_0",
                    "AOOOF01D1": "AOOOF01_1", "AACFE03A": "AACFE03_"}
        for code, famille in attendus.items():
            self.assertEqual(commandes.famille_variante(code), famille)
        # Seule la dernière lettre change : pas de suppression de toutes les lettres ni des suffixes.
        for code in ("MOCS.0000002776", "ASEQ4200", "AAAA"):
            self.assertEqual(commandes.famille_variante(code), code)
        self.assertEqual(commandes.famille_variante("ATTTF321A"), "ATTTF321_")

    def test_vocabulaire_par_periode_et_union(self):
        a = table(*[(i * H, "AOOOF01D0") for i in range(50)], *[(i * H, "AOOOF01D1") for i in range(49)])
        b = table(*[(i * H, "AOOOF01A1") for i in range(30)], *[(i * H, "AOOOF01B1") for i in range(20)])
        self.assertEqual(variables.apprendre_familles([a], variante=True), ["AOOOF01_0"])
        self.assertEqual(variables.apprendre_familles([a, b], variante=True), ["AOOOF01_0", "AOOOF01_1"])
        comptes = variables.comptes_familles(table((0, "AOOOF01C0"), (H, "AOOOF01D1")), np.arange(2) * H,
                                             ["AOOOF01_0"], variante=True)
        np.testing.assert_array_equal(comptes["dmopfam_AOOOF01_0_count"], [1, 0])


class TestDonneesReelles(unittest.TestCase):
    def test_physique_causale_et_options_du_world_model(self):
        """Les variables physiques à l'heure t ne lisent rien après t ; les options du GRU ajoutent
        exactement les colonnes physiques et familles."""
        x = pd.read_csv(RACINE / "data/prepared/annee1/X.csv", index_col="ut_ms").iloc[:500]
        futur = x.copy()
        futur.iloc[300:] = futur.iloc[300:].to_numpy()[::-1]
        a, b = variables.physiques(x), variables.physiques(futur)
        pd.testing.assert_frame_equal(a.iloc[:300], b.iloc[:300])
        dmop = donnees.lire_dmop("annee1")
        base, noms_base = variables.construire(x, "v1", (), dmop)
        f, noms = variables.construire(x, "v1", (), dmop, physique=True, familles=["ATTTF321"])
        self.assertEqual(noms, noms_base + list(a.columns) + ["dmopfam_ATTTF321_count"])
        np.testing.assert_array_equal(f[:, :len(noms_base)], base)

    @unittest.skipUnless(VALIDATION.is_dir(), "validation/ absent (make_validation.py)")
    def test_power_init_n_entre_pas_dans_les_variables(self):
        with tempfile.TemporaryDirectory() as temp:
            copie = Path(temp) / "test"
            shutil.copytree(VALIDATION, copie)
            power = pd.read_csv(copie / "power_init.csv")
            power.iloc[:, 1:] = power.iloc[:, 1:] * 3 + 1
            power.to_csv(copie / "power_init.csv", index=False)
            tables = []
            for dossier in (VALIDATION, copie):
                test = donnees.lire_dossier_test(dossier)
                vocab = caracteristiques.apprendre([test["tables"]])
                tables.append(caracteristiques.construire(test["tables"], test["heures"], vocab))
            pd.testing.assert_frame_equal(tables[0], tables[1])


if __name__ == "__main__":
    unittest.main()
