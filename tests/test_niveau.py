"""Borne du résidu initial élargie à la prédiction selon la variance apprise."""
import unittest
import tempfile
from pathlib import Path

import torch

from dio.monde import CONFIG_DEFAUT, MondeGRU, charger


class TestBorneNiveau(unittest.TestCase):
    def test_borne_prediction_utilise_uniquement_la_variance_apprise(self):
        stats = {"garder": torch.ones(2, dtype=torch.bool), "y_moyenne": torch.zeros(2),
                 "y_ecart": torch.tensor([.1, 1e-6])}
        m = MondeGRU(2, 2, 4, 1, 0, niveau=True,
                     y_stats=(stats["y_moyenne"], stats["y_ecart"]))
        config = {**CONFIG_DEFAUT, "taille": 4, "couches": 1, "dropout": 0.,
                  "niveau_lent": False, "borne_niveau_prediction": 10.}
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


if __name__ == "__main__":
    unittest.main()
