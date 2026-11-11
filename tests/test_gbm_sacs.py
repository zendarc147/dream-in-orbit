"""Tests du boosting à sacs temporels : découpage, vocabulaire, absence de fuite."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import gbm_sacs
from dio import commandes, donnees, variables

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
        planning = gbm_sacs.apprendre_planning([a, table(*[(i * H, f"MOCS.{i:010d}") for i in range(50)])], True)
        self.assertEqual(planning["familles"], ["AOOOF01_0", "MOCS"])


class TestEvenementsCommandes(unittest.TestCase):
    def test_changements_de_variante_sans_identite(self):
        # A, A, C, C, B aux heures 0, 1, 3, 4, 8 : changements A→C à 3 h et C→B à 8 h.
        dmop = table((0, "AAAAF20A1"), (H, "AAAAF20A1"), (3 * H, "AAAAF20C1"), (4 * H, "AAAAF20C1"),
                     (8 * H, "AAAAF20B1"), (5 * H, "AXXX301A"))
        depuis, jusqua = gbm_sacs.changements_variante(dmop.sort_values("ut_ms"), np.arange(10) * H, "AAAAF20_1", True)
        plafond = np.log1p(720)
        np.testing.assert_allclose(depuis, [plafond] * 4 + list(np.log1p([1, 2, 3, 4, 5, 1])))
        np.testing.assert_allclose(jusqua, list(np.log1p([3, 2, 1, 0, 4, 3, 2, 1, 0])) + [plafond])
        # Lettre inconnue à l'entraînement (Z) : un changement comme un autre.
        _, jusqua = gbm_sacs.changements_variante(table((0, "AAAAF20A1"), (2 * H, "AAAAF20Z1")), np.arange(2) * H,
                                                  "AAAAF20_1", True)
        np.testing.assert_allclose(jusqua, np.log1p([2, 1]))

    def test_recence_au_debut_de_l_heure(self):
        temps = np.array([H + H // 2])  # commande à 1 h 30
        np.testing.assert_allclose(gbm_sacs.heures_depuis(temps, np.arange(4) * H, "debut"),
                                   np.log1p([720, 720, 0.5, 1.5]))  # l'heure 1 ne voit pas sa propre commande
        np.testing.assert_allclose(gbm_sacs.heures_depuis(temps, np.arange(4) * H), [168, 0.5, 1.5, 2.5])

    def test_options_du_planning(self):
        dmop = table(*[(i * H, "AAAAF20A1") for i in range(30)], *[(i * H, "AAAAF20C1") for i in range(30, 60)],
                     *[(i * H, "ASEQ4200") for i in range(5)]).sort_values("ut_ms")
        planning = gbm_sacs.apprendre_planning([dmop], True, recence="debut", prefixes=True, changements=True,
                                               paires="liste")
        self.assertEqual((planning["familles"], planning["prefixes"], planning["changements"]),
                         (["AAAAF20_1"], ["AAAA"], ["AAAAF20_1"]))
        self.assertEqual(planning["paires"], gbm_sacs.PAIRES_LISTE)
        f = gbm_sacs.variables_planning(dmop, np.arange(70) * H, planning)
        np.testing.assert_array_equal(f["plan_pre_AAAA_count"][:60], 1)
        self.assertAlmostEqual(f["plan_AAAAF20_1_jusqua_chgt_h"][0], np.log1p(30))


class TestPlanning(unittest.TestCase):
    def test_vocabulaire_recence_futur_et_periodes(self):
        entrainement = table(*[(i * H, "ATTTF321A") for i in range(20)], *[(i * H, f"MOCS.{i:010d}") for i in range(20)],
                             *[(i * H, f"MOCE.{i:010d}") for i in range(20)], *[(i * H, "AXXX301A") for i in range(5)])
        planning = gbm_sacs.apprendre_planning([entrainement])
        self.assertEqual(planning, {"familles": ["ATTTF321", "MOCE", "MOCS"], "paires": [("MOCS", "MOCE")]})
        # Test : ATTTF321B (variante inconnue, même famille) à l'heure 2 ; période MOCS.7 de 1 h 30 à 3 h.
        test = table((2 * H + 10, "ATTTF321B"), (H + H // 2, "MOCS.0000000007"), (3 * H, "MOCE.0000000007"),
                     (0, "MOCE.0000000099"))
        f = gbm_sacs.variables_planning(test, np.arange(6) * H, planning)
        np.testing.assert_allclose(f["plan_ATTTF321_depuis_h"], [168, 168, (H - 10) / H, (2 * H - 10) / H,
                                                                 (3 * H - 10) / H, (4 * H - 10) / H])
        np.testing.assert_array_equal(f["plan_ATTTF321_futur1h"], [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(f["plan_ATTTF321_futur6h"], [1, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(f["plan_MOCS_MOCE_actif"], [0, 0.5, 1, 0, 0, 0])
        absente = gbm_sacs.variables_planning(table((0, "AXXX301A")), np.arange(3) * H, planning)
        np.testing.assert_array_equal(absente["plan_MOCS_depuis_h"], [168, 168, 168])

    def test_historique_et_moyennes_exponentielles_causales(self):
        x = pd.read_csv(RACINE / "data/prepared/annee1/X.csv", index_col="ut_ms").iloc[:600]
        futur = x.copy()
        futur.iloc[400:] = futur.iloc[400:].to_numpy()[::-1]
        for fonction in (lambda t: gbm_sacs.historique(t, set()), gbm_sacs.moyennes_exponentielles):
            a, b = fonction(x), fonction(futur)
            pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])
        constante = gbm_sacs.moyennes_exponentielles(x.iloc[:1].reindex(x.index[:50], method="ffill"))
        np.testing.assert_allclose(constante.to_numpy(), np.broadcast_to(constante.iloc[0], constante.shape))

    def test_sans_horloge_retire_seulement_ltdata_lentes(self):
        x = pd.read_csv(RACINE / "data/prepared/annee1/X.csv", index_col="ut_ms").iloc[:300]
        dmop = donnees.lire_dmop("annee1")
        a = gbm_sacs.construire(x, dmop, "V2", [])
        b = gbm_sacs.construire(x, dmop, "V2", [], sans_horloge=True)
        self.assertEqual(sorted(set(a.columns) - set(b.columns)), sorted(gbm_sacs.HORLOGE))
        c = gbm_sacs.construire(x, dmop, "V2", [], retirer=["lettres"])
        codes = donnees.charger_schema()["dmop_codes"]
        self.assertEqual(sorted(set(a.columns) - set(c.columns)), sorted(f"dmop_{k}_count" for k in codes))


class TestDonneesReelles(unittest.TestCase):
    def test_aucune_horloge_absolue(self):
        """Décaler toute la période dans le temps ne change aucune variable."""
        x = pd.read_csv(RACINE / "data/prepared/annee1/X.csv", index_col="ut_ms").iloc[:2000]
        dmop = donnees.lire_dmop("annee1")
        dmop = dmop[dmop.ut_ms < x.index[-1] + H]
        familles = variables.apprendre_familles([dmop])
        planning, ajouts = gbm_sacs.apprendre_planning([dmop]), ("planning", "historique", "ema")
        a = gbm_sacs.construire(x, dmop, "V2", familles, ajouts, planning)
        decalage = 1000 * H
        b = gbm_sacs.construire(x.set_axis(x.index + decalage), dmop.assign(ut_ms=dmop.ut_ms + decalage),
                                "V2", familles, ajouts, planning)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
        self.assertTrue(np.isfinite(a.to_numpy()).all())

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
    def test_lissage_cible_garde_les_trous_et_les_autres_colonnes(self):
        from dio import monde
        y = np.array([[0., 1.], [3., 1.], [np.nan, 1.], [6., 1.], [0., 1.]])
        lisse = monde.lisser(y, ["A", "B"], {"A": 3})
        np.testing.assert_allclose(lisse[:, 0], [1.5, 1.5, np.nan, 3., 3.])  # centré, NaN ignorés puis remis
        np.testing.assert_array_equal(lisse[:, 1], y[:, 1])
        self.assertEqual(y[1, 0], 3.)  # entrée intacte

    def test_power_init_n_entre_pas_dans_les_variables(self):
        with tempfile.TemporaryDirectory() as temp:
            copie = Path(temp) / "test"
            shutil.copytree(VALIDATION, copie)
            power = pd.read_csv(copie / "power_init.csv")
            power.iloc[:, 1:] = power.iloc[:, 1:] * 3 + 1
            power.to_csv(copie / "power_init.csv", index=False)
            variables = []
            for dossier in (VALIDATION, copie):
                test = donnees.lire_dossier_test(dossier)
                variables.append(gbm_sacs.construire(test["X"], test["dmop"], "V2", ["ATTTF321", "AOOOF01"]))
            pd.testing.assert_frame_equal(variables[0], variables[1])


if __name__ == "__main__":
    unittest.main()
