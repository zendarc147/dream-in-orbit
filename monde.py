"""World model GRU : simule les 33 courants heure par heure à partir du contexte.

Architecture :
- encodeur : un GRU lit les 7 jours de power_init (courants + contexte) et produit
  l'état initial h0 du simulateur ;
- simulateur : un GRU à 2 couches avance heure par heure sur le contexte seul, sans
  jamais réinjecter ses propres prédictions (pas d'accumulation d'erreur par rebouclage) ;
- tête : une couche linéaire donne les 33 courants à l'échelle brute.

Entraînement : épisodes aléatoires de 7 jours de démarrage + N jours de simulation,
perte MSE sur l'échelle brute (comme la métrique), heures inconnues masquées.
Les épisodes longs sont découpés en morceaux (BPTT tronquée) : l'état caché passe
d'un morceau au suivant, un pas d'optimisation par morceau. Le simulateur apprend
ainsi à rester stable sur l'horizon réel (un an) et pas seulement sur 60 jours.
Chaque variable est bornée à sa plage d'entraînement : pas d'extrapolation hors domaine.
Plusieurs graines forment un ensemble : moyenne = prédiction, écart-type = incertitude.
"""
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

import dmop_etats
import donnees
import variables

CONFIG_DEFAUT = {
    "variables": "v1",       # jeu de variables d'entrée (voir variables.py)
    "etats_dmop": False,     # ajouter l'état « dernière commande » des sous-systèmes dmop
    "sans": [],              # groupes de variables retirés (ablations)
    "memoire": True,         # False : ni encodeur ni récurrence (ablation)
    "statique": False,       # donner à chaque heure le niveau moyen des circuits sur power_init
    "bruit": 0.3,            # écart-type du bruit gaussien ajouté aux entrées normalisées (entraînement)
    "taille": 128,
    "couches": 2,
    "dropout": 0.1,
    "jours_init": 7,
    "jours_sim": 365,        # longueur d'un épisode de simulation
    "morceau_h": 720,        # BPTT tronquée par morceaux de 30 jours (0 : un seul morceau)
    "lot": 32,
    "pas": 300,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "graines": [0, 1, 2, 3, 4],
    "threads": 4,
}
H_JOUR = 24
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"


def normaliser_entrees(f, stats):
    """Garder les variables informatives, borner à la plage d'entraînement, centrer-réduire (±5)."""
    garder = stats["garder"].astype(bool)
    f = np.clip(f[..., garder], stats["x_min"][garder], stats["x_max"][garder])
    return np.clip((f - stats["x_moyenne"][garder]) / stats["x_ecart"][garder], -5, 5)


def calculer_stats(f_tout, y_tout):
    """Statistiques d'entraînement. Une variable dont la valeur modale couvre ≥ 99,5 % des
    heures est écartée : quasi constante, sa normalisation explose dès qu'elle change."""
    ecart = f_tout.std(axis=0)
    mode = np.array([np.unique(c, return_counts=True)[1].max() / len(c) for c in f_tout.T])
    return {
        "garder": (mode < 0.995) & (ecart > 1e-9),
        "x_moyenne": f_tout.mean(axis=0),
        "x_min": f_tout.min(axis=0),
        "x_max": f_tout.max(axis=0),
        "x_ecart": np.where(ecart > 1e-9, ecart, 1.0),
        "y_moyenne": np.nanmean(y_tout, axis=0),
        "y_ecart": np.maximum(np.nanstd(y_tout, axis=0), 1e-6),
    }


class MondeGRU(nn.Module):
    def __init__(self, n_entrees, n_circuits, taille, couches, dropout, memoire=True, statique=False):
        super().__init__()
        self.memoire = memoire
        self.statique = statique and memoire
        self.couches, self.taille = couches, taille
        # Avec « statique », le niveau moyen de chaque circuit sur power_init (et la part
        # d'heures connues) accompagne le contexte à chaque heure : il ne s'efface pas.
        entree = n_entrees + (2 * n_circuits if self.statique else 0)
        self.projection = nn.Sequential(nn.Linear(entree, taille), nn.GELU())
        if memoire:
            # L'encodeur voit le contexte, les courants normalisés et leur masque de présence.
            self.encodeur = nn.GRU(n_entrees + 2 * n_circuits, taille, batch_first=True)
            self.vers_h0 = nn.Linear(taille, couches * taille)
            self.simulateur = nn.GRU(taille, taille, num_layers=couches, batch_first=True,
                                     dropout=dropout if couches > 1 else 0.0)
        else:
            self.simulateur = nn.Sequential(nn.Linear(taille, taille), nn.GELU(), nn.Dropout(dropout),
                                            nn.Linear(taille, taille), nn.GELU())
        self.tete = nn.Linear(taille, n_circuits)

    def etat_initial(self, x_init, y_init):
        """x_init (B, Ti, F), y_init (B, Ti, 2C) courants normalisés + masque -> h0 (couches, B, taille)."""
        if not self.memoire:
            return None
        _, h = self.encodeur(torch.cat([x_init, y_init], dim=-1))
        h0 = torch.tanh(self.vers_h0(h[-1]))
        return h0.view(-1, self.couches, self.taille).transpose(0, 1).contiguous()

    @staticmethod
    def resume_init(y_init):
        """Moyenne normalisée de chaque circuit sur les heures connues + part d'heures connues."""
        n = y_init.shape[-1] // 2
        valeurs, masque = y_init[..., :n], y_init[..., n:]
        comptes = masque.sum(dim=1)
        return torch.cat([valeurs.sum(dim=1) / comptes.clamp(min=1), comptes / y_init.shape[1]], dim=-1)

    def simuler(self, x_sim, h, resume=None):
        """Avancer sur le contexte seul : retourne (courants (B, Ts, C), état final)."""
        if self.statique:
            x_sim = torch.cat([x_sim, resume[:, None].expand(-1, x_sim.shape[1], -1)], dim=-1)
        z = self.projection(x_sim)
        if not self.memoire:
            return self.tete(self.simulateur(z)), None
        sortie, h = self.simulateur(z, h)
        return self.tete(sortie), h

    def forward(self, x_init, y_init, x_sim):
        return self.simuler(x_sim, self.etat_initial(x_init, y_init), self.resume_init(y_init))[0]


def normaliser_courants(y, moyenne, ecart):
    """Courants normalisés (NaN -> 0) concaténés à leur masque de présence."""
    connus = np.isfinite(y)
    yn = np.where(connus, (y - moyenne) / ecart, 0.0)
    return np.concatenate([yn, connus.astype(np.float64)], axis=-1)


def preparer_annees(annees, config):
    schema = donnees.charger_schema()
    tables = {annee: donnees.lire_dmop(annee) for annee in annees}
    vocabulaire = dmop_etats.apprendre_vocabulaire(list(tables.values())) if config["etats_dmop"] else None
    periodes = []
    for annee in annees:
        _, x, y = donnees.charger_annee(annee, schema)
        f, noms = variables.construire(x, config["variables"], config["sans"], tables[annee], vocabulaire)
        periodes.append((f, y))
    return schema, periodes, noms, vocabulaire


def entrainer_un(periodes, stats, config, graine, n_circuits):
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    ti, ts = config["jours_init"] * H_JOUR, config["jours_sim"] * H_JOUR
    morceau = config["morceau_h"] or ts
    appareil = APPAREIL
    modele = MondeGRU(int(stats["garder"].sum()), n_circuits, config["taille"], config["couches"],
                      config["dropout"], config["memoire"], config.get("statique", False)).to(appareil)
    with torch.no_grad():
        modele.tete.bias.copy_(torch.as_tensor(stats["y_moyenne"], dtype=torch.float32))
    optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    plan = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr"], total_steps=config["pas"],
                                               pct_start=0.1)
    debut, cumul, pas = time.time(), 0.0, 0
    modele.train()
    while pas < config["pas"]:
        # Épisode : longueur commune au lot, bornée par la plus courte période.
        sim = min(ts, min(len(f) for f, _ in periodes) - ti)
        tailles = np.array([len(f) - ti - sim + 1 for f, _ in periodes], dtype=float)
        choix = rng.choice(len(periodes), size=config["lot"], p=tailles / tailles.sum())
        xs, ys = [], []
        for p in choix:
            f, y = periodes[p]
            s = rng.integers(0, len(f) - ti - sim + 1)
            xs.append(f[s:s + ti + sim])
            ys.append(y[s:s + ti + sim])
        x = torch.as_tensor(normaliser_entrees(np.stack(xs), stats), dtype=torch.float32, device=appareil)
        if config.get("bruit", 0.0):
            # Régularisation : le modèle ne doit pas dépendre d'une valeur d'entrée exacte.
            x = x + config["bruit"] * torch.randn_like(x)
        y = np.stack(ys)
        y_init = torch.as_tensor(normaliser_courants(y[:, :ti], stats["y_moyenne"], stats["y_ecart"]),
                                 dtype=torch.float32, device=appareil)
        cible = torch.as_tensor(y[:, ti:], dtype=torch.float32, device=appareil)
        h = None
        for a in range(0, sim, morceau):
            if pas >= config["pas"]:
                break
            if a == 0:
                h = modele.etat_initial(x[:, :ti], y_init)
                resume = modele.resume_init(y_init)
            elif h is not None:
                h = h.detach()
            sortie, h = modele.simuler(x[:, ti + a:ti + a + morceau], h, resume)
            c = cible[:, a:a + morceau]
            masque = torch.isfinite(c)
            if not masque.any():
                continue
            perte = ((sortie - torch.nan_to_num(c)) ** 2)[masque].mean()
            optim.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optim.step()
            plan.step()
            cumul = 0.98 * cumul + 0.02 * perte.item() if pas else perte.item()
            if pas % 100 == 0 or pas == config["pas"] - 1:
                print(f"  graine {graine} pas {pas:5d} RMSE lissée {cumul ** 0.5:.4f}"
                      f" ({time.time() - debut:.0f} s)", flush=True)
            pas += 1
    return modele.cpu()


def _entrainer_graine(arguments):
    """Travailleur d'un processus : une graine, un thread (le GRU passe mal à l'échelle en threads)."""
    periodes, stats, config, graine, n_circuits = arguments
    torch.set_num_threads(1)
    modele = entrainer_un(periodes, stats, config, graine, n_circuits)
    return {k: v.half() for k, v in modele.state_dict().items()}


def entrainer(annees, dossier, **options):
    config = {**CONFIG_DEFAUT, **options}
    torch.set_num_threads(int(os.environ.get("MONDE_THREADS", config["threads"])))
    schema, periodes, noms, vocabulaire = preparer_annees(annees, config)
    f_tout = np.concatenate([f for f, _ in periodes])
    y_tout = np.concatenate([y for _, y in periodes])
    stats = calculer_stats(f_tout, y_tout)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    n_circuits = len(schema["target_columns"])
    processus = min(len(config["graines"]), int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1)))
    if APPAREIL == "cpu" and processus > 1:
        # Sur CPU, plusieurs graines en parallèle (un processus mono-thread chacune).
        contexte = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(processus, mp_context=contexte) as pool:
            etats = list(pool.map(_entrainer_graine,
                                  [(periodes, stats, config, g, n_circuits) for g in config["graines"]]))
    else:
        etats = []
        for graine in config["graines"]:
            modele = entrainer_un(periodes, stats, config, graine, n_circuits)
            etats.append({k: v.half() for k, v in modele.state_dict().items()})
    torch.save({
        "config": config, "variables": noms, "vocabulaire": vocabulaire, "circuits": schema["target_columns"], "annees": list(annees),
        "stats": {k: torch.as_tensor(v) for k, v in stats.items()}, "etats": etats,
    }, dossier / "monde.pt")
    print(f"{len(etats)} world model(s) entraîné(s) sur {', '.join(annees)}")


def charger(dossier):
    paquet = torch.load(Path(dossier) / "monde.pt", weights_only=True)
    config = paquet["config"]
    stats = {k: v.numpy() for k, v in paquet["stats"].items()}
    modeles = []
    for etat in paquet["etats"]:
        modele = MondeGRU(int(stats["garder"].sum()), len(paquet["circuits"]), config["taille"],
                          config["couches"], config["dropout"], config["memoire"],
                          config.get("statique", False))
        modele.load_state_dict({k: v.float() for k, v in etat.items()})
        modeles.append(modele.eval())
    return paquet, config, stats, modeles


@torch.no_grad()
def simuler(dossier, x_df, y_init, n_init, dmop=None):
    """Retourner (moyenne, écart-type) de l'ensemble sur toute la grille (heures init comprises)."""
    paquet, config, stats, modeles = charger(dossier)
    torch.set_num_threads(config["threads"])
    f, noms = variables.construire(x_df, config["variables"], config["sans"], dmop,
                                   paquet.get("vocabulaire"))
    if noms != paquet["variables"]:
        raise ValueError("Variables différentes de celles de l'entraînement")
    x = torch.as_tensor(normaliser_entrees(f, stats)[None], dtype=torch.float32)
    yi = torch.as_tensor(normaliser_courants(y_init[None, :n_init], stats["y_moyenne"], stats["y_ecart"]),
                         dtype=torch.float32)
    sorties = np.stack([m(x[:, :n_init], yi, x[:, n_init:])[0].numpy() for m in modeles])
    # Les heures initiales sont connues : on y recopie la vérité (ou la moyenne si absente).
    debut = np.where(np.isfinite(y_init[:n_init]), y_init[:n_init], stats["y_moyenne"])
    moyenne = np.concatenate([debut, sorties.mean(axis=0)])
    ecart = np.concatenate([np.zeros_like(debut), sorties.std(axis=0)])
    return np.clip(moyenne, 0, None), ecart


def predire(test_dir, out, dossier, sortie_incertitude=None):
    paquet = torch.load(Path(dossier) / "monde.pt", weights_only=True)
    test = donnees.lire_dossier_test(test_dir)
    moyenne, ecart = simuler(dossier, test["X"], test["y_init"], test["n_init"], test["dmop"])
    table = donnees.ecrire_predictions(test["heures"], moyenne, test["heures_a_predire"], paquet["circuits"], out)
    if sortie_incertitude:
        donnees.ecrire_predictions(test["heures"], ecart, test["heures_a_predire"], paquet["circuits"],
                                   sortie_incertitude)
    return table


def resume(dossier):
    paquet = torch.load(Path(dossier) / "monde.pt", weights_only=True)
    return json.dumps(paquet["config"])
