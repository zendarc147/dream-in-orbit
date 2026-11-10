"""Correction de niveau à la prédiction : borne du résidu initial et facteur des poids."""
import unittest
import tempfile
from pathlib import Path

import torch

from dio.monde import CONFIG_DEFAUT, MondeGRU, charger


class TestBorneNiveau(unittest.TestCase):
    def test_borne_prediction_utilise_uniquement_la_variance_apprise(self):
        stats = {"garder": torch.ones(2, dtype=torch.bool), "y_moyenne": torch.zeros(2),
                 "y_ecart": torch.tensor([.1, 1e-6])}
        m = MondeGRU(2, 2, 4, 1, 0, (stats["y_moyenne"], stats["y_ecart"]))
        config = {**CONFIG_DEFAUT, "taille": 4, "couches": 1, "dropout": 0., "borne_niveau_prediction": 10.}
        paquet = {"config": config, "stats": stats, "circuits": ["a", "b"],
                  "etats": [m.state_dict()]}
        with tempfile.TemporaryDirectory() as temp:
            chemin = Path(temp) / "monde.pt"
            torch.save(paquet, chemin)
            modele = charger(temp)[3][0]
            torch.testing.assert_close(modele.borne_niveau, torch.tensor([10., 5.]))
            for nom, valeur in m.state_dict().items():
                torch.testing.assert_close(modele.state_dict()[nom], valeur)
            del config["borne_niveau_prediction"]
            torch.save(paquet, chemin)
            self.assertEqual(charger(temp)[3][0].borne_niveau, 5.)

    def test_facteur_multiplie_seulement_la_correction(self):
        """Le facteur multiplie les deux poids de correction ; absent, les poids sont inchangés."""
        stats = {"garder": torch.ones(2, dtype=torch.bool), "y_moyenne": torch.zeros(2), "y_ecart": torch.ones(2)}
        m = MondeGRU(2, 2, 4, 1, 0, (stats["y_moyenne"], stats["y_ecart"]))
        with torch.no_grad():
            m.poids_niveau.fill_(.2)
            m.poids_lent.fill_(.1)
        config = {**CONFIG_DEFAUT, "taille": 4, "couches": 1, "dropout": 0., "facteur_niveau_prediction": 1.5}
        paquet = {"config": config, "stats": stats, "circuits": ["a", "b"], "etats": [m.state_dict()]}
        with tempfile.TemporaryDirectory() as temp:
            torch.save(paquet, Path(temp) / "monde.pt")
            modele = charger(temp)[3][0]
            torch.testing.assert_close(modele.poids_niveau, torch.full((2,), .3))
            torch.testing.assert_close(modele.poids_lent, torch.full((2,), .15))
            torch.testing.assert_close(modele.tete.weight, m.tete.weight)
            del config["facteur_niveau_prediction"]
            torch.save(paquet, Path(temp) / "monde.pt")
            torch.testing.assert_close(charger(temp)[3][0].poids_niveau, torch.full((2,), .2))


if __name__ == "__main__":
    unittest.main()
