"""Tests ciblés ; les contrôles locaux complets s'activent après prepare.py."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import prepare as p
from make_validation import construire


RACINE = Path(__file__).resolve().parent


def petits_contextes():
    return {
        "dmop": pd.DataFrame({"ut_ms": [0, p.H, 3 * p.H], "subsystem": ["A", "A", "B"]}),
        "ftl": pd.DataFrame({"utb_ms": [0], "ute_ms": [4 * p.H], "type": ["EARTH"], "flagcomms": [True]}),
        "evtf": pd.DataFrame({"ut_ms": [0, p.H, 2 * p.H], "description": ["CODE_START", "CODE_END", "POINT"]}),
        "saaf": pd.DataFrame({"ut_ms": [0, p.H, 2 * p.H, 3 * p.H],
                              **{angle: [10., 20., 30., 40.] for angle in p.ANGLES}}),
        "ltdata": pd.DataFrame({"ut_ms": [0], **{c: [1.] for c in p.COLONNES["ltdata"][1:]}}),
    }


def petit_schema(tables, heures):
    schema = p.apprendre_categories(tables, ["NPWD1", "NPWD2"])
    x, _ = p.construire_entrees(tables, heures, schema)
    schema["feature_columns"] = x.columns.tolist()
    schema["fill_values"] = x.median().to_dict()
    return schema


class PreparationTest(unittest.TestCase):
    def test_courants_frontieres_moyenne_et_heure_vide(self):
        with tempfile.TemporaryDirectory() as dossier:
            fichier = Path(dossier) / "power.csv"
            pd.DataFrame({"ut_ms": [0, p.H - 1, p.H, 3 * p.H],
                          "NPWD1": [0., 2., 4., 8.]}).to_csv(fichier, index=False)
            y = p.cibles_horaires(fichier)
        self.assertEqual(y.index.tolist(), [0, p.H, 2 * p.H, 3 * p.H])
        np.testing.assert_allclose(y.NPWD1, [1., 4., np.nan, 8.], equal_nan=True)

    def test_grille_refuse_doublons_trous_et_decalages(self):
        for heures in ([0, 0], [0, 2 * p.H], [1, p.H + 1], [p.H, 0], [0., float(p.H)], []):
            with self.subTest(heures=heures), self.assertRaises(ValueError):
                p.verifier_heures(heures)

    def test_durees_plusieurs_heures_et_fin_exclue(self):
        heures = p.verifier_heures(range(0, 4 * p.H, p.H))
        np.testing.assert_allclose(p.durees_union(heures, [p.H // 2], [5 * p.H // 2]), [30, 60, 30, 0])
        np.testing.assert_allclose(p.durees_union(heures, [-p.H], [p.H]), [60, 0, 0, 0])
        np.testing.assert_allclose(p.durees_union(heures, [p.H], [2 * p.H]), [0, 60, 0, 0])

    def test_ftl_union_par_code_et_chevauchements(self):
        heures = p.verifier_heures([0, p.H])
        ftl = pd.DataFrame({"utb_ms": [0, p.H // 2, p.H // 2, p.H],
                            "ute_ms": [p.H, 2 * p.H, 2 * p.H, 2 * p.H],
                            "type": ["A", "A", "B", "B"], "flagcomms": [True, True, False, False]})
        x = p.preparer_ftl(ftl, heures, {"ftl_codes": ["A"]})
        np.testing.assert_allclose(x.ftl_A_minutes, [60, 60])
        np.testing.assert_allclose(x.ftl_other_minutes, [30, 60])
        np.testing.assert_allclose(x.ftl_coverage_minutes, [60, 60])
        np.testing.assert_allclose(x.ftl_comms_minutes, [60, 60])
        np.testing.assert_allclose(x.ftl_overlap_minutes, [30, 60])

    def test_dmop_categories_nouvelles_et_commande_a_la_frontiere(self):
        heures = p.verifier_heures([0, p.H, 2 * p.H])
        table = pd.DataFrame({"ut_ms": [p.H, 3 * p.H], "subsystem": ["NOUVEAU", "A"]})
        x = p.preparer_dmop(table, heures, {"dmop_codes": ["A"]})
        np.testing.assert_array_equal(x.dmop_other_count, [0, 1, 0])
        np.testing.assert_array_equal(x.dmop_A_count, [0, 0, 0])
        np.testing.assert_array_equal(x.dmop_no_previous_command, [1, 0, 0])
        np.testing.assert_allclose(x.dmop_hours_since_last, [np.nan, 1, 2], equal_nan=True)

    def test_evtf_marqueurs_et_parametres(self):
        descriptions = pd.Series(["OCC_MARS_START_/_RA_181.69_/_DE_-00.08", "NNO_CON_END_SESC_3",
                                  "4000_KM_DESCEND", "PERICENTRE_PASSAGE_07544"])
        codes = p.codes_evenements(descriptions)
        marqueurs, familles = p.bornes_evenements(codes)
        self.assertEqual(codes.iloc[0], "OCC_MARS_START")
        self.assertEqual(familles.iloc[1], "NNO_CON_SESC_3")
        self.assertTrue(pd.isna(marqueurs.iloc[2]))
        self.assertEqual(codes.iloc[3], "PERICENTRE_PASSAGE")

    def test_evtf_durees_et_comptages_separes(self):
        heures = p.verifier_heures([0, p.H, 2 * p.H])
        table = pd.DataFrame({"ut_ms": [0, p.H, 2 * p.H, 2 * p.H],
                              "description": ["CODE_START", "POINT_NOUVEAU", "CODE_END", "AUTRE_START"]})
        x, anomalies = p.preparer_evtf(table, heures, {"evtf_intervals": ["CODE"]})
        np.testing.assert_allclose(x.evtf_CODE_minutes, [60, 60, 0])
        np.testing.assert_array_equal(x.evtf_point_count, [0, 1, 0])
        np.testing.assert_array_equal(x.evtf_start_count, [1, 0, 1])
        np.testing.assert_array_equal(x.evtf_end_count, [0, 0, 1])
        np.testing.assert_array_equal(x.evtf_other_boundary_count, [0, 0, 1])
        self.assertEqual(anomalies, {"CODE": 0})

    def test_evtf_orphelins_et_debuts_repetes_inconnus(self):
        heures = p.verifier_heures(range(0, 4 * p.H, p.H))
        temps = [p.H // 2, p.H, 3 * p.H // 2, 2 * p.H, 5 * p.H // 2]
        durees, inconnu, nombre = p.apparier_evenements(temps, ["END", "START", "START", "END", "START"], heures)
        self.assertTrue(np.isnan(durees).all())
        np.testing.assert_array_equal(inconnu, [1, 1, 1, 1])
        self.assertEqual(nombre, 3)

    def test_saaf_moyenne_ponderee_et_pas_moyenne_des_lignes(self):
        table = pd.DataFrame({"ut_ms": [0, 10_000, 60_000],
                              **{angle: [0., 60., 120.] for angle in p.ANGLES}})
        x = p.preparer_saaf(table, p.verifier_heures([0]), {"saaf_hold_ms": 60_000})
        self.assertAlmostEqual(x.saaf_sa_mean.iloc[0], 85.)
        self.assertAlmostEqual(x.saaf_sa_range.iloc[0], 120.)
        self.assertAlmostEqual(x.saaf_coverage_fraction.iloc[0], 120 / 3600)

    def test_saaf_trou_non_prolonge_et_morceau_frontiere(self):
        table = pd.DataFrame({"ut_ms": [0, 2 * p.H], **{angle: [3., 9.] for angle in p.ANGLES}})
        heures = p.verifier_heures([0, p.H, 2 * p.H])
        x = p.preparer_saaf(table, heures, {"saaf_hold_ms": 60_000})
        self.assertTrue(np.isnan(x.saaf_sa_mean.iloc[1]))
        np.testing.assert_allclose(x.saaf_coverage_fraction, [1 / 60, 0, 1 / 60])
        table = pd.DataFrame({"ut_ms": [p.H - 30_000], **{angle: [7.] for angle in p.ANGLES}})
        x = p.preparer_saaf(table, heures, {"saaf_hold_ms": 60_000})
        np.testing.assert_allclose(x.saaf_coverage_fraction, [1 / 120, 1 / 120, 0])
        np.testing.assert_allclose(x.saaf_sa_mean.iloc[:2], [7., 7.])

    def test_ltdata_une_journee_absente_reste_inconnue(self):
        table = pd.DataFrame({"ut_ms": [0, 2 * p.JOUR],
                              **{c: [1., 3.] for c in p.COLONNES["ltdata"][1:]}})
        heures = p.verifier_heures(range(0, 3 * p.JOUR, p.H))
        x = p.preparer_ltdata(table, heures)
        self.assertEqual(int(x.ltdata_missing.sum()), 24)
        self.assertTrue(x.iloc[24:48, :-1].isna().all().all())
        self.assertEqual(x.ltdata_sunmars_km.iloc[-1], 3.)

    def test_sources_vides_remplissage_et_schema_fixe(self):
        tables = petits_contextes()
        heures = p.verifier_heures(range(0, 4 * p.H, p.H))
        schema = petit_schema(tables, heures)
        schema_avant = json.dumps(schema, sort_keys=True)
        vides = {nom: table.iloc[:0].copy() for nom, table in tables.items()}
        x, _ = p.construire_entrees(vides, heures, schema)
        self.assertTrue(x.saaf_sa_mean.isna().all())
        self.assertTrue(x.evtf_CODE_minutes.isna().all())
        rempli = p.remplir_entrees(x, schema)
        self.assertTrue(np.isfinite(rempli.to_numpy()).all())
        self.assertEqual(rempli.saaf_sa_mean.iloc[0], 25.)
        self.assertEqual(rempli.ltdata_missing.sum(), 4)
        self.assertEqual(rempli.columns.tolist(), schema["feature_columns"])
        self.assertEqual(json.dumps(schema, sort_keys=True), schema_avant)
        with self.assertRaises(ValueError):
            p.remplir_entrees(x[x.columns[::-1]], schema)

    def test_fichier_absent_et_intervalles_invalides_refuses(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            with self.assertRaises(FileNotFoundError):
                p.lire_contextes(dossier)
            tables = petits_contextes()
            tables["ftl"].loc[0, "ute_ms"] = 0
            for nom, table in tables.items():
                table.to_csv(dossier / f"{nom}.csv", index=False)
            with self.assertRaisesRegex(ValueError, "durée non positive"):
                p.lire_contextes(dossier)

    def test_changer_les_courants_ne_change_pas_X(self):
        with tempfile.TemporaryDirectory() as temporaire:
            dossier = Path(temporaire)
            tables = petits_contextes()
            schema = petit_schema(tables, p.verifier_heures(range(0, 4 * p.H, p.H)))
            for nom, table in tables.items():
                table.to_csv(dossier / f"{nom}.csv", index=False)
            schema_path = dossier / "schema.json"
            schema_path.write_text(json.dumps(schema))
            pd.DataFrame({"ut_ms": [2 * p.H, 3 * p.H]}).to_csv(dossier / "heures_a_predire.csv", index=False)
            power = pd.DataFrame({"ut_ms": [0, p.H], "NPWD1": [1., 2.], "NPWD2": [3., 4.]})
            lecture = pd.read_csv

            def interdire_power(chemin, *args, **kwargs):
                if Path(chemin).name == "power.csv":
                    raise AssertionError("X ne doit jamais lire power.csv")
                return lecture(chemin, *args, **kwargs)

            with patch.object(p.pd, "read_csv", side_effect=interdire_power):
                power.to_csv(dossier / "power_init.csv", index=False)
                power.to_csv(dossier / "power.csv", index=False)
                p.preparer_test(dossier, schema_path, dossier / "sortie1")
                power[["NPWD1", "NPWD2"]] *= 10_000
                power.to_csv(dossier / "power_init.csv", index=False)
                power.to_csv(dossier / "power.csv", index=False)
                p.preparer_test(dossier, schema_path, dossier / "sortie2")
            assert_frame_equal(lecture(dossier / "sortie1/X.csv"), lecture(dossier / "sortie2/X.csv"))
            self.assertFalse(lecture(dossier / "sortie1/y_init.csv").equals(lecture(dossier / "sortie2/y_init.csv")))


@unittest.skipUnless((RACINE / "data/prepared/annee2/y.csv").is_file(),
                     "Exécuter d'abord python prepare.py --data-dir data --out data/prepared")
class DonneesLocalesTest(unittest.TestCase):
    def test_deux_periodes_reference_officielle_et_format_test(self):
        proteges = list((RACINE / "data").glob("annee*/*.csv"))
        proteges += [RACINE / nom for nom in ("sujet.md", "exploration.ipynb", "make_validation.py",
                                             "evaluate.py", "requirements.txt", ".gitignore", "data/manifest.json")]

        def empreintes():
            resultat = {}
            for fichier in proteges:
                with fichier.open("rb") as flux:
                    resultat[str(fichier)] = hashlib.file_digest(flux, "sha256").hexdigest()
            return resultat

        avant = empreintes()
        schema_path = RACINE / "data/prepared/schema.json"
        schema = json.loads(schema_path.read_text())
        with tempfile.TemporaryDirectory() as temporaire:
            for periode, trous in (("annee1", 32), ("annee2", 246)):
                with self.subTest(periode=periode):
                    source = RACINE / "data" / periode
                    prepare = RACINE / "data/prepared" / periode
                    x = pd.read_csv(prepare / "X.csv").set_index("ut_ms")
                    y = pd.read_csv(prepare / "y.csv").set_index("ut_ms")
                    self.assertEqual(len(x), 16_488)
                    self.assertEqual(y.shape, (16_488, 33))
                    self.assertEqual(x.columns.tolist(), schema["feature_columns"])
                    self.assertEqual(y.columns.tolist(), schema["target_columns"])
                    self.assertTrue(np.isfinite(x.to_numpy()).all())
                    self.assertFalse(np.isinf(y.to_numpy()).any())
                    self.assertEqual(int(y.isna().all(axis=1).sum()), trous)
                    self.assertEqual(int(y.isna().sum().sum()), trous * 33)
                    p.verifier_heures(x.index)
                    self.assertTrue(x.index.equals(y.index))
                    validation = Path(temporaire) / periode
                    construire(source / "power.csv", source, validation)
                    verite = pd.read_csv(validation / "verite_horaire.csv").set_index("ut_ms")
                    assert_frame_equal(y.loc[verite.index], verite, rtol=1e-12, atol=1e-12)
                    dossier_test = validation / "test"
                    self.assertFalse((dossier_test / "power.csv").exists())
                    sortie = validation / "prepare"
                    p.preparer_test(dossier_test, schema_path, sortie)
                    heures = pd.read_csv(dossier_test / "heures_a_predire.csv").ut_ms
                    xtest = pd.read_csv(sortie / "X.csv").set_index("ut_ms")
                    self.assertEqual(len(xtest), 16_319)
                    assert_frame_equal(x.loc[heures], xtest, rtol=1e-12, atol=1e-12)
                    attendu_init = p.cibles_horaires(dossier_test / "power_init.csv", schema["target_columns"])
                    obtenu_init = pd.read_csv(sortie / "y_init.csv").set_index("ut_ms")
                    assert_frame_equal(attendu_init, obtenu_init, rtol=1e-12, atol=1e-12)
        self.assertEqual(avant, empreintes(), "Un fichier à préserver a changé")


if __name__ == "__main__":
    unittest.main()
