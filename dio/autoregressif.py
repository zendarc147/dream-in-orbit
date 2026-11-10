"""World model GRU autorégressif (étude de la phase 4, docs/JOURNAL_ENSEMBLE.md, non livré).

Différences avec monde.py :
- le simulateur reçoit à chaque heure, en plus du contexte, les 33 courants de l'heure
  précédente (normalisés) : la vérité ou sa propre prédiction ;
- scheduled sampling : la probabilité de lui donner la vérité (quand elle existe) décroît
  linéairement de 1 à 0 sur la première moitié des pas, puis reste nulle : en fin
  d'entraînement il se nourrit de ses propres sorties, comme à la prédiction ;
- flux continus : chaque élément du lot est un flux qui avance de morceau en morceau dans sa
  période, état caché et dernière sortie conservés (détachés) ; arrivé au bout, il repart d'un
  point tiré au hasard, avec un nouvel état initial lu sur 7 jours réels.
Prédiction : simulation libre complète, partant de la dernière heure connue de power_init.
Variables : celles de monde.py (même préparation, mêmes statistiques), sans branche de commandes
ni correction de niveau : le retour des sorties doit porter seul le niveau initial.
"""
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import donnees, monde

CONFIG_DEFAUT = {
    "variables": "v2r", "commandes": None, "branche_commandes": None, "sans": [], "familles": True,
    "masque_saaf": 0.0, "bruit": 0.0, "taille": 128, "couches": 2, "jours_init": 7, "morceau_h": 720,
    "lot": 32, "pas": 600, "lr": 3e-3, "weight_decay": 1e-4, "ema": 0.99, "fin_echantillonnage": 0.5,
    "bruit_retour": 0.0, "graines": [0, 1, 2, 3, 4],
}


class MondeAR(nn.Module):
    def __init__(self, n_entrees, n_circuits, taille, couches):
        super().__init__()
        self.couches, self.taille = couches, taille
        self.projection = nn.Sequential(nn.Linear(n_entrees, taille), nn.GELU())
        self.encodeur = nn.GRU(n_entrees + 2 * n_circuits, taille, batch_first=True)
        self.vers_h0 = nn.Linear(taille, couches * taille)
        self.cellules = nn.ModuleList([nn.GRUCell(taille + n_circuits if i == 0 else taille, taille)
                                       for i in range(couches)])
        self.tete = nn.Linear(taille, n_circuits)
        nn.init.zeros_(self.tete.weight)
        nn.init.zeros_(self.tete.bias)

    def etat_initial(self, x_init, y_init):
        _, h = self.encodeur(torch.cat([x_init, y_init], dim=-1))
        return list(torch.tanh(self.vers_h0(h[-1])).view(-1, self.couches, self.taille).unbind(1))

    def avancer(self, x, h, precedent, verite=None, p_verite=0.0, bruit=0.0):
        """x (B, T, F), h liste d'états (B, taille), precedent (B, C) courants normalisés de l'heure
        d'avant. verite (B, T, C) normalisée avec NaN : donnée au pas suivant avec la probabilité
        p_verite là où elle existe. Retourne (sorties normalisées (B, T, C), h, dernier retour)."""
        z, sorties = self.projection(x), []
        for t in range(x.shape[1]):
            entree = precedent + bruit * torch.randn_like(precedent) if bruit else precedent
            e = torch.cat([z[:, t], entree], dim=-1)
            for i, cellule in enumerate(self.cellules):
                h[i] = cellule(e, h[i])
                e = h[i]
            sortie = precedent + self.tete(e)  # résidu : le modèle prédit la variation horaire
            sorties.append(sortie)
            precedent = sortie.clamp(-5, 5)
            if verite is not None and p_verite > 0:
                v = verite[:, t]
                prendre = torch.isfinite(v) & (torch.rand_like(v[:, :1]) < p_verite)
                precedent = torch.where(prendre, torch.nan_to_num(v), precedent)
        return torch.stack(sorties, dim=1), h, precedent


def dernier_connu(yn_init, masque):
    """(B, C) : dernière valeur normalisée connue de chaque circuit sur power_init (0 = moyenne)."""
    t = torch.arange(masque.shape[1], device=masque.device)[None, :, None]
    position = torch.where(masque > 0, t, -1).amax(dim=1)
    valeur = torch.gather(yn_init, 1, position.clamp(min=0)[:, None]).squeeze(1)
    return torch.where(position >= 0, valeur, 0.0)


def entrainer_un(periodes, stats, config, graine, n_circuits):
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    ti, morceau, lot = config["jours_init"] * 24, config["morceau_h"], config["lot"]
    moyenne, ecart = stats["y_moyenne"], stats["y_ecart"]
    m_t, e_t = torch.as_tensor(moyenne, dtype=torch.float32), torch.as_tensor(ecart, dtype=torch.float32)
    modele = MondeAR(int(stats["garder"].sum()), n_circuits, config["taille"], config["couches"])
    optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    plan = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr"], total_steps=config["pas"], pct_start=0.1)
    moyen = torch.optim.swa_utils.AveragedModel(
        modele, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["ema"])) if config["ema"] else None
    tailles = np.array([len(f) - ti - morceau for f, _ in periodes], dtype=float)
    flux = [None] * lot  # (période, position) de chaque flux ; None = à (re)démarrer
    h, precedent = None, None
    debut, cumul = time.time(), 0.0
    for pas in range(config["pas"]):
        p_verite = max(0.0, 1.0 - pas / (config["fin_echantillonnage"] * config["pas"]))
        nouveaux = [b for b in range(lot) if flux[b] is None or flux[b][1] + morceau > len(periodes[flux[b][0]][0])]
        if nouveaux:
            xi, yi = [], []
            for b in nouveaux:
                p = rng.choice(len(periodes), p=tailles / tailles.sum())
                s = int(rng.integers(0, len(periodes[p][0]) - ti - morceau + 1))
                flux[b] = (p, s + ti)
                f, y = periodes[p]
                xi.append(monde.normaliser_entrees(f[s:s + ti], stats))
                yi.append(monde.normaliser_courants(y[s:s + ti], moyenne, ecart))
            xi = torch.as_tensor(np.stack(xi), dtype=torch.float32)
            yi = torch.as_tensor(np.stack(yi), dtype=torch.float32)
            h0 = modele.etat_initial(xi, yi)
            p0 = dernier_connu(yi[..., :n_circuits], yi[..., n_circuits:])
            if h is None:
                h, precedent = h0, p0
            else:
                index = torch.as_tensor(nouveaux)
                h = [hh.index_copy(0, index, h0i) for hh, h0i in zip(h, h0)]
                precedent = precedent.index_copy(0, index, p0)
        xs, ys = [], []
        for b in range(lot):
            p, s = flux[b]
            f, y = periodes[p]
            xs.append(monde.normaliser_entrees(f[s:s + morceau], stats))
            ys.append(y[s:s + morceau])
            flux[b] = (p, s + morceau)
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32)
        cible = torch.as_tensor(np.stack(ys), dtype=torch.float32)
        verite = (cible - m_t) / e_t
        sortie, h, precedent = modele.avancer(x, h, precedent, verite, p_verite, config["bruit_retour"])
        brute = sortie * e_t + m_t
        masque = torch.isfinite(cible)
        perte = ((brute - torch.nan_to_num(cible)) ** 2)[masque].mean()
        optim.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optim.step()
        plan.step()
        if moyen is not None:
            moyen.update_parameters(modele)
        h, precedent = [hh.detach() for hh in h], precedent.detach()
        cumul = 0.98 * cumul + 0.02 * perte.item() if pas else perte.item()
        if pas % 100 == 0 or pas == config["pas"] - 1:
            print(f"  graine {graine} pas {pas:5d} p_verite {p_verite:.2f} RMSE lissée {cumul ** 0.5:.4f}"
                  f" ({time.time() - debut:.0f} s)", flush=True)
    return (moyen.module if moyen is not None else modele)


def _entrainer_graine(arguments):
    torch.set_num_threads(1)
    modele = entrainer_un(*arguments)
    return {k: v.half() for k, v in modele.state_dict().items()}


def entrainer(annees, dossier, **options):
    config = {**CONFIG_DEFAUT, **options}
    schema, periodes, noms, voc, _ = monde.preparer_annees(annees, config)
    stats = monde.calculer_stats(np.concatenate([f for f, _ in periodes]), np.concatenate([y for _, y in periodes]))
    n_circuits = len(schema["target_columns"])
    processus = min(len(config["graines"]), int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1)))
    with ProcessPoolExecutor(processus, mp_context=multiprocessing.get_context("spawn")) as pool:
        etats = list(pool.map(_entrainer_graine, [(periodes, stats, config, g, n_circuits) for g in config["graines"]]))
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "variables": noms, "commandes": voc, "circuits": schema["target_columns"],
                "stats": {k: torch.as_tensor(v) for k, v in stats.items()}, "etats": etats}, dossier / "ar.pt")
    print(f"{len(etats)} world model(s) autorégressif(s) entraîné(s) sur {', '.join(annees)}")


@torch.no_grad()
def predire(test_dir, out, dossier):
    paquet = torch.load(Path(dossier) / "ar.pt", weights_only=True)
    config, stats = paquet["config"], {k: v.numpy() for k, v in paquet["stats"].items()}
    test = donnees.lire_dossier_test(test_dir)
    n_init, n_circuits = test["n_init"], len(paquet["circuits"])
    f, noms = monde.construire_variables(test["X"], config, test["dmop"], paquet["commandes"])
    if noms != paquet["variables"]:
        raise ValueError("Variables différentes de celles de l'entraînement")
    x = torch.as_tensor(monde.normaliser_entrees(f, stats)[None], dtype=torch.float32)
    yi = torch.as_tensor(monde.normaliser_courants(test["y_init"][None, :n_init], stats["y_moyenne"],
                                                   stats["y_ecart"]), dtype=torch.float32)
    sorties = []
    for etat in paquet["etats"]:
        modele = MondeAR(int(stats["garder"].sum()), n_circuits, config["taille"], config["couches"])
        modele.load_state_dict({k: v.float() for k, v in etat.items()})
        modele.eval()
        h = modele.etat_initial(x[:, :n_init], yi)
        sortie, _, _ = modele.avancer(x[:, n_init:], h, dernier_connu(yi[..., :n_circuits], yi[..., n_circuits:]))
        sorties.append(sortie[0].numpy() * stats["y_ecart"] + stats["y_moyenne"])
    debut = np.where(np.isfinite(test["y_init"][:n_init]), test["y_init"][:n_init], stats["y_moyenne"])
    moyenne = np.clip(np.concatenate([debut, np.mean(sorties, axis=0)]), 0, None)
    return donnees.ecrire_predictions(test["heures"], moyenne, test["heures_a_predire"], paquet["circuits"], out)
