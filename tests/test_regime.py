"""Tête de régime partagée (NPWD2551, NPWD2451, NPWD2851), perte auxiliaire et masquage du contexte."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dio import donnees
from dio import monde

RACINE = Path(__file__).resolve().parent.parent
DONNEES = (RACINE / "data/prepared/annee1/X.csv").is_file() and (RACINE / "data/annee1/dmop.csv").is_file()
REGIME = {"circuits": [1, 3, 4], "k": 2}


def modele(regime=REGIME, graine=0):
    torch.manual_seed(graine)
    return monde.MondeGRU(5, 6, 8, 2, 0.0, (torch.zeros(6), torch.ones(6)), regime=regime).eval()


class TeteRegimeTest(unittest.TestCase):
    def test_sans_regime_architecture_inchangee(self):
        sans, avec = modele(None), modele()
        etat = sans.state_dict()
        self.assertFalse(any("regime" in k for k in etat))
        # Les couches communes ont la même initialisation, la tête de régime étant créée en dernier.
        for k, v in etat.items():
            self.assertTrue(torch.equal(v, avec.state_dict()[k]), k)

    def test_probabilites_et_esperance_du_melange(self):
        m = modele()
        with torch.no_grad():
            m.tete_regime[-1].weight.normal_()
        etat = torch.randn(2, 7, 8)
        sortie, logits = m.sortie_tete(etat)
        p = logits.softmax(-1)
        self.assertTrue(torch.isfinite(p).all())
        torch.testing.assert_close(p.sum(-1), torch.ones(2, 7))
        niveaux = m.niveaux_regime(etat).view(2, 7, 3, 2)
        attendu = p[..., None, 0] * niveaux[..., 0] + p[..., None, 1] * niveaux[..., 1]
        torch.testing.assert_close(sortie[..., REGIME["circuits"]], attendu)

    def test_seules_les_trois_colonnes_utilisent_la_tete(self):
        m = modele()
        etat = torch.randn(1, 5, 8)
        avant = m.sortie_tete(etat)[0]
        with torch.no_grad():
            m.tete_regime[-1].bias.copy_(torch.tensor([3.0, -3.0]))
            m.niveaux_regime.bias.add_(1.0)
        apres = m.sortie_tete(etat)[0]
        autres = [j for j in range(6) if j not in REGIME["circuits"]]
        torch.testing.assert_close(apres[..., autres], avant[..., autres], rtol=0, atol=0)
        self.assertFalse(torch.allclose(apres[..., REGIME["circuits"]], avant[..., REGIME["circuits"]]))

    def test_initialisation_reproduit_la_tete_actuelle(self):
        m = modele()
        m.initialiser_regime(np.array([[0.0, 1.0], [0.2, 0.9], [0.1, 0.3]]))
        etat = torch.randn(2, 9, 8)
        torch.testing.assert_close(m.sortie_tete(etat)[0], m.tete(etat))
        niveaux = m.niveaux_regime(etat).view(2, 9, 3, 2)
        self.assertTrue((niveaux[..., 1] > niveaux[..., 0]).all())

    def test_prediction_deterministe(self):
        m = modele()
        x, y = torch.randn(1, 60, 5), torch.randn(1, 24, 12)
        torch.manual_seed(1)
        a = m(x[:, :24], y, x[:, 24:])
        torch.manual_seed(2)
        torch.testing.assert_close(m(x[:, :24], y, x[:, 24:]), a, rtol=0, atol=0)

    def test_simulation_par_morceaux_identique(self):
        m = modele()
        with torch.no_grad():
            m.poids_niveau.fill_(0.5)
            m.poids_lent.fill_(0.3)
            m.tete_regime[-1].weight.normal_()
        x, y_init, ti, morceau = torch.randn(1, 130, 5), torch.randn(1, 24, 12), 24, 40
        complet = m(x[:, :ti], y_init, x[:, ti:])
        h, resume, sorties = m.etat_initial(x[:, :ti], y_init), m.resume_niveau(x[:, :ti], y_init), []
        for a in range(0, 130 - ti, morceau):
            sortie, h = m.simuler(x[:, ti + a:ti + a + morceau], h, resume, a)
            sorties.append(sortie)
        torch.testing.assert_close(torch.cat(sorties, dim=1), complet)


class PerteRegimeTest(unittest.TestCase):
    def test_etiquette_majoritaire(self):
        seuils = torch.tensor([[0.5], [0.5], [0.5]])
        y = torch.tensor([[0.9, 0.1, 0.8], [0.1, 0.1, 0.9], [0.6, 0.7, 0.0]])
        self.assertEqual(monde.etiquettes_regime(y, seuils).tolist(), [1, 0, 1])

    def test_gradients_finis_avec_manquants_et_isoles_du_gru(self):
        m = modele().train()
        x, y_init = torch.randn(2, 50, 5), torch.randn(2, 24, 12)
        cible = torch.rand(2, 26, 6)
        cible[0, :5] = float("nan")
        cible[1, 3, 4] = float("nan")
        sortie, _, logits = m.simuler(x[:, 24:], m.etat_initial(x[:, :24], y_init),
                                      m.resume_niveau(x[:, :24], y_init), avec_logits=True)
        seuils, poids = torch.full((3, 1), 0.5), torch.tensor([1.0, 2.0])
        ce = monde.perte_regime(logits, cible[..., REGIME["circuits"]], seuils, poids)
        self.assertTrue(torch.isfinite(ce))
        ce.backward(retain_graph=True)
        # La perte auxiliaire n'entraîne que la tête de régime : GRU et autres circuits intacts.
        for nom, p in m.named_parameters():
            if not nom.startswith("tete_regime"):
                self.assertTrue(p.grad is None or not p.grad.any(), nom)
        m.zero_grad()
        masque = torch.isfinite(cible)
        perte = ((sortie - torch.nan_to_num(cible)) ** 2)[masque].mean() + 0.1 * ce
        perte.backward()
        for nom, p in m.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), nom)

    def test_stats_apprises_sur_les_heures_donnees(self):
        rng = np.random.default_rng(0)
        y = np.where(rng.random((2000, 4)) < 0.4, 1e-3, rng.uniform(0.3, 1.0, (2000, 4)))
        y[5, 2] = np.nan
        stats = monde.stats_regime(y, [0, 2, 3], 2)
        self.assertEqual(stats["regime_seuils"].shape, (3, 1))
        self.assertTrue(((stats["regime_seuils"] > 1e-3) & (stats["regime_seuils"] < 0.3)).all())
        self.assertTrue((stats["regime_centres"][:, 0] < stats["regime_centres"][:, 1]).all())
        self.assertGreater(stats["regime_poids"][0], stats["regime_poids"][1])  # bas plus rare : plus lourd


class AbandonContexteTest(unittest.TestCase):
    def test_probabilite_nulle(self):
        x = torch.randn(3, 100, 4)
        etat = torch.get_rng_state()
        self.assertIs(monde.abandonner_contexte(x, [1], 0.0, 24), x)
        self.assertTrue(torch.equal(torch.get_rng_state(), etat))

    def test_blocs_contigus_apres_power_init(self):
        torch.manual_seed(0)
        x = torch.ones(16, 300, 4)
        sortie = monde.abandonner_contexte(x, [0, 2], 1.0, 24)
        self.assertTrue((sortie[:, :24] == 1).all())
        self.assertTrue((sortie[..., [1, 3]] == 1).all())
        for b in range(16):
            zeros = torch.nonzero(sortie[b, :, 0] == 0).flatten()
            self.assertGreaterEqual(len(zeros), 24)
            self.assertEqual(int(zeros[-1] - zeros[0]) + 1, len(zeros))  # un seul bloc
            self.assertTrue(torch.equal(sortie[b, :, 0], sortie[b, :, 2]))  # groupes masqués ensemble


@unittest.skipUnless(DONNEES, "données préparées absentes")
class EntrainementRegimeTest(unittest.TestCase):
    def test_entrainement_statistiques_et_rechargement(self):
        options = {"graines": [0], "pas": 3, "lot": 2, "jours_sim": 10, "morceau_h": 120, "taille": 8,
                   "ema": 0.0, "threads": 1, "regime_partage": True, "lambda_regime": 0.1,
                   "abandon_regime_contexte": 1.0, "curriculum_jours": [5, 10]}
        with tempfile.TemporaryDirectory() as temp:
            monde.entrainer(["annee1"], temp, **options)
            paquet, config, stats, modeles = monde.charger(temp)
        self.assertEqual(paquet["annees"], ["annee1"])
        # Seuils et centres : appris sur annee1 seulement, sérialisés, différents de ceux d'annee2.
        _, _, y1 = donnees.charger_annee("annee1")
        _, _, y2 = donnees.charger_annee("annee2")
        cibles = config["_regime"]["circuits"]
        self.assertEqual([paquet["circuits"][j] for j in cibles], monde.CIBLES_REGIME)
        attendu = monde.stats_regime(y1, cibles, 2)
        for nom, valeur in attendu.items():
            np.testing.assert_allclose(stats[nom], valeur, rtol=1e-6)
        self.assertFalse(np.allclose(monde.stats_regime(y2, cibles, 2)["regime_seuils"], stats["regime_seuils"]))
        np.testing.assert_allclose(stats["y_moyenne"], np.nanmean(y1, axis=0), rtol=1e-6)
        # Contexte masqué : éclipses, modes FTL et DMOP, jamais saaf ni ltdata hors éclipses.
        gardees = [n for n, g in zip(paquet["variables"], stats["garder"]) if g]
        masquees = {gardees[i] for i in config["_contexte"]}
        self.assertIn("evtf_MAR_UMBRA_minutes", masquees)
        self.assertIn("ftl_NADIR_minutes", masquees)
        self.assertIn("dmop_total_count", masquees)
        self.assertFalse({n for n in masquees if n.startswith(("saaf_", "ltdata_sun", "ftl_comms"))})
        # Rechargement : tête de régime présente, prédiction déterministe et finie.
        m = modeles[0]
        self.assertIsNotNone(m.regime)
        x, y = torch.randn(1, 200, len(gardees)), torch.randn(1, 168, 66)
        torch.manual_seed(1)
        a = m(x[:, :168], y, x[:, 168:])
        torch.manual_seed(2)
        torch.testing.assert_close(m(x[:, :168], y, x[:, 168:]), a, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()
