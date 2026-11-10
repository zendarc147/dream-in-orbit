"""Régularisations de robustesse : masquage SAAF par blocs et abandon groupé des états de commandes."""
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dio import donnees
from dio import monde

RACINE = Path(__file__).resolve().parent.parent
DONNEES = (RACINE / "data/prepared/annee1/X.csv").is_file() and (RACINE / "data/annee1/dmop.csv").is_file()
PETIT = {"graines": [0], "pas": 2, "lot": 2, "jours_sim": 10, "morceau_h": 120, "taille": 8, "ema": 0.0,
         "threads": 1}


class MasquageSaafTest(unittest.TestCase):
    def setUp(self):
        self.schema = {"fill_values": {"saaf_coverage_fraction": 1.0, "saaf_sa_mean": 0.5, "saaf_missing": 0.0}}
        self.x = pd.DataFrame({"saaf_coverage_fraction": 1.0, "saaf_sa_mean": np.linspace(0, 1, 2000),
                               "saaf_missing": 0, "ftl_x": 3.0}, index=np.arange(2000) * donnees.H)

    def test_fraction_nulle_ne_change_rien(self):
        pd.testing.assert_frame_equal(monde.masquer_saaf(self.x, self.schema, 0.0), self.x)

    def test_table_source_intacte(self):
        avant = self.x.copy()
        monde.masquer_saaf(self.x, self.schema, 0.3, graine=1)
        pd.testing.assert_frame_equal(self.x, avant)

    def test_heures_masquees(self):
        m = monde.masquer_saaf(self.x, self.schema, 0.1, graine=2)
        masque = m.saaf_missing.to_numpy() == 1
        self.assertEqual(masque.sum(), 200)
        self.assertTrue((m.saaf_sa_mean[masque] == 0.5).all())
        self.assertTrue((m.saaf_coverage_fraction[masque] == 0.0).all())
        pd.testing.assert_frame_equal(m[~masque], self.x[~masque])
        pd.testing.assert_series_equal(m.ftl_x, self.x.ftl_x)
        pd.testing.assert_frame_equal(m, monde.masquer_saaf(self.x, self.schema, 0.1, graine=2))


class AbandonEtatsTest(unittest.TestCase):
    def test_probabilite_un_met_tous_les_etats_a_zero(self):
        x = torch.randn(4, 10, 5)
        sortie = monde.abandonner_etats(x, [1, 3], 1.0)
        self.assertTrue((sortie[..., [1, 3]] == 0).all())
        torch.testing.assert_close(sortie[..., [0, 2, 4]], x[..., [0, 2, 4]])

    def test_probabilite_nulle_conserve_le_tenseur(self):
        x = torch.randn(4, 10, 5)
        etat = torch.get_rng_state()
        self.assertIs(monde.abandonner_etats(x, [1, 3], 0.0), x)
        self.assertTrue(torch.equal(torch.get_rng_state(), etat))

    def test_abandon_par_episode_entier(self):
        torch.manual_seed(0)
        sortie = monde.abandonner_etats(torch.ones(64, 10, 3), [0, 1], 0.5)
        abandonnes = sortie[:, 0, 0] == 0
        self.assertTrue(0 < abandonnes.sum() < 64)
        self.assertTrue((sortie[abandonnes][..., :2] == 0).all())
        self.assertTrue((sortie[~abandonnes] == 1).all())
        self.assertTrue((sortie[..., 2] == 1).all())


class CompatibiliteTest(unittest.TestCase):
    @unittest.skipUnless((RACINE / "poids/membres.json").is_file(), "poids/ absent")
    def test_poids_finaux_se_rechargent(self):
        for membre in ("membre0", "membre1"):
            self.assertEqual(len(monde.charger(RACINE / "poids" / membre)[3]), 5)

    def test_anciens_poids_sans_regularisations_se_rechargent(self):
        """Poids d'avant les régularisations (commit 8fa1d0e) : options absentes, valeurs par défaut nulles."""
        for membre in ("membre0", "membre1"):
            ancien = subprocess.run(["git", "show", f"8fa1d0e:poids/{membre}/monde.pt"], cwd=RACINE,
                                    capture_output=True)
            if ancien.returncode:
                self.skipTest("historique Git absent")
            with tempfile.TemporaryDirectory() as temp:
                (Path(temp) / "monde.pt").write_bytes(ancien.stdout)
                _, config, _, modeles = monde.charger(temp)
            self.assertNotIn("masque_saaf", config)
            self.assertNotIn("abandon_etats", config)
            self.assertEqual(len(modeles), 5)


@unittest.skipUnless(DONNEES, "données préparées absentes")
class EntrainementTest(unittest.TestCase):
    def entrainer(self, dossier, **options):
        monde.entrainer(["annee1"], dossier, **PETIT, **options)
        return torch.load(Path(dossier) / "monde.pt", weights_only=True)

    def test_regularisations(self):
        with tempfile.TemporaryDirectory() as temp:
            # Les régularisations sont actives par défaut : la référence les coupe explicitement.
            reference = self.entrainer(Path(temp) / "ref", masque_saaf=0.0, abandon_etats=0.0)
            masque = self.entrainer(Path(temp) / "masque", masque_saaf=0.1, abandon_etats=1.0)
            # Copies masquées : statistiques, variables et sélection inchangées.
            for nom, valeur in reference["stats"].items():
                self.assertTrue(torch.equal(valeur, masque["stats"][nom]), nom)
            self.assertEqual(reference["variables"], masque["variables"])
            gardees = [n for n, g in zip(masque["variables"], masque["stats"]["garder"]) if g]
            etats = masque["config"]["_etats"]
            self.assertTrue(etats)
            self.assertEqual(etats, [i for i, n in enumerate(gardees) if n.startswith("dmopcmd_etat_")])
            self.assertFalse(any(gardees[i].startswith("dmopcmd_impulsion_") for i in etats))
            # Prédiction déterministe, quel que soit l'état du générateur aléatoire.
            modele = monde.charger(Path(temp) / "masque")[3][0]
            x, y = torch.randn(1, 200, len(gardees)), torch.randn(1, 168, 66)
            torch.manual_seed(1)
            a = modele(x[:, :168], y, x[:, 168:])
            torch.manual_seed(2)
            torch.testing.assert_close(modele(x[:, :168], y, x[:, 168:]), a, rtol=0, atol=0)

    def test_bloc_exclu_absent_des_statistiques(self):
        """Hors bloc : les courants du bloc écarté (et de sa marge) ne touchent pas l'entraînement."""
        _, _, y = donnees.charger_annee("annee1")
        n, (a, b) = len(y), (5496, 10992)
        with tempfile.TemporaryDirectory() as temp:
            paquet = self.entrainer(Path(temp) / "hors", exclure=[a / n, b / n])
        gardees = np.concatenate([y[:a - 168], y[b + 168:]])
        np.testing.assert_allclose(paquet["stats"]["y_moyenne"].numpy(), np.nanmean(gardees, axis=0))


class ExclusionTest(unittest.TestCase):
    def test_morceaux_autour_du_bloc(self):
        heures = np.arange(1000)
        x = pd.DataFrame({"a": np.arange(1000.0)}, index=heures)
        morceaux = monde.exclure_bloc((heures, x, np.arange(1000.0)[:, None], None), 0.4, 0.6, marge=50)
        self.assertEqual([(m[0][0], m[0][-1]) for m in morceaux], [(0, 349), (650, 999)])
        self.assertEqual(len(monde.exclure_bloc((heures, x, x.to_numpy(), None), 0.0, 0.5, marge=50)), 1)


if __name__ == "__main__":
    unittest.main()
