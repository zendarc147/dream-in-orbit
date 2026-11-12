"""World model GRU « recette » : plus petit, plus régularisé, avec réinjection de la sortie (point 9 de
docs/JOURNAL_RECETTE.md ; étude, comparée au GRU mixte installé).

- Entrée de chaque heure : contexte (variables v2r + familles de commandes commandes.famille_variante,
  sans les comptes exacts avec lettre de variante) concaténé à la consommation de l'heure précédente,
  standardisée (vérité ou propre prédiction ; option reinjection=False : pas de retour) ;
- projection Linear → GELU → Dropout, une couche GRU, tête Linear → GELU → Linear (sortie standardisée) ;
- dropout 0.3, dropout des variables 0.2 ;
- 16 flux continus par époque, chacun sur un seizième de la période, à partir d'un décalage aléatoire tiré à
  chaque époque ; segments de 168 h, état caché et dernière sortie transmis puis détachés entre segments ;
- réinjection de la vérité les 15 premières époques, puis probabilité décroissant linéairement jusqu'à 0 à la
  dernière époque ; bruit gaussien 0.05 sur la valeur réinjectée ;
- AdamW (lr max 1e-3, weight decay 1e-3), OneCycle (montée sur 15 % des pas), gradient borné à 1 ;
- entrées standardisées sur l'entraînement (écart-type < 1e-3 remplacé par 1), puis bornées à ±10, sans
  borne préalable aux min/max d'entraînement ; perte MSE sur l'échelle brute, cibles absentes masquées.
Prédiction : les heures de power_init avec la vérité réinjectée, puis simulation libre. Moyenne des graines.
"""
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from baselines.boosting import recaler
from . import caracteristiques, donnees, variables

CONFIG_DEFAUT = {
    "variables": "v2r", "familles_variante": True, "retirer": [r"dmop_(?!other_count|total_count)\w+_count"],
    "taille": 128, "dropout": 0.3, "dropout_variables": 0.2, "graines": [0, 1, 2, 3, 4], "epoques": 80,
    "flux": 16, "segment_h": 168, "epoques_verite": 15, "bruit_retour": 0.05, "lr": 1e-3, "weight_decay": 1e-3,
    "montee": 0.15, "reinjection": True,
    # Recette complète (docs/JOURNAL_RECETTE_COMPLETE.md) : variables « S » de dio/caracteristiques.py (options
    # d'ablation dans « recette ») ; B : premier segment de chaque flux avec la vérité réinjectée ; normalisation
    # « minmax » : bornes min/max d'entraînement puis clip ±5 (convention de monde.py, ablation).
    "recette": None, "premier_segment_verite": False, "normalisation": "standard",
}
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
BORNE = 10.0


class Recette(nn.Module):
    def __init__(self, n_variables, n_circuits, config):
        super().__init__()
        self.reinjection = config["reinjection"]
        taille, n_entrees = config["taille"], n_variables + (n_circuits if config["reinjection"] else 0)
        self.abandon_variables = nn.Dropout(config["dropout_variables"])
        self.projection = nn.Sequential(nn.Linear(n_entrees, taille), nn.GELU(), nn.Dropout(config["dropout"]))
        self.cellule = nn.GRUCell(taille, taille)
        self.tete = nn.Sequential(nn.Linear(taille, taille), nn.GELU(), nn.Linear(taille, n_circuits))

    def avancer(self, x, h, precedent, verite=None, p_verite=0.0, bruit=0.0):
        """x (B, T, F) standardisé, h (B, taille), precedent (B, C) sortie standardisée de l'heure d'avant.
        verite (B, T, C) standardisée avec NaN : réinjectée à la place de la prédiction avec la probabilité
        p_verite là où elle existe. Retourne (sorties standardisées (B, T, C), h, dernière valeur réinjectée)."""
        x, sorties = self.abandon_variables(x), []
        for t in range(x.shape[1]):
            entree = torch.cat([x[:, t], precedent], dim=-1) if self.reinjection else x[:, t]
            h = self.cellule(self.projection(entree), h)
            sortie = self.tete(h)
            sorties.append(sortie)
            precedent = sortie
            if verite is not None and p_verite > 0:
                v = verite[:, t]
                prendre = torch.isfinite(v) & (torch.rand_like(v[:, :1]) < p_verite)
                precedent = torch.where(prendre, torch.nan_to_num(v), precedent)
            if bruit:
                precedent = precedent + bruit * torch.randn_like(precedent)
        return torch.stack(sorties, dim=1), h, precedent


def construire(x, dmop, config, familles):
    valeurs, noms = variables.construire(x, config["variables"], dmop=dmop, familles=familles,
                                         variante=config["familles_variante"])
    retirer = re.compile("|".join(config["retirer"]) or "(?!)")
    garder = [not retirer.fullmatch(n) for n in noms]
    return valeurs[:, garder], [n for n, g in zip(noms, garder) if g]


def colonnes_s(table, config):
    """Colonnes GRU de la recette S, moins celles de config["retirer"] (expressions régulières)."""
    retirer = re.compile("|".join(config["retirer"]) or "(?!)")
    return [c for c in caracteristiques.colonnes_gru(table) if not retirer.fullmatch(c)]


def preparer_s(annees, config, schema):
    """Variables S par période (vocabulaire appris sur ces seules périodes) ; périodes contiguës fusionnées
    (aucune variable GRU ne regarde les heures voisines)."""
    brutes = [caracteristiques.charger_periode(a, schema["target_columns"]) for a in annees]
    vocab = caracteristiques.apprendre([t for _, t, _ in brutes])
    periodes, fin = [], None
    for h, t, y in brutes:
        table = caracteristiques.construire(t, h, vocab, config["recette"])
        noms = colonnes_s(table, config)
        f = table[noms].to_numpy(dtype=np.float64)
        if periodes and fin + donnees.H == h[0]:
            f0, y0 = periodes.pop()
            f, y = np.concatenate([f0, f]), np.concatenate([y0, y])
        periodes.append((f, y))
        fin = h[-1]
    return schema, periodes, noms, vocab


def preparer(annees, config):
    """Périodes (variables, courants) ; années contiguës fusionnées en une période."""
    schema = donnees.charger_schema()
    if config["variables"] == "S":
        return preparer_s(annees, config, schema)
    blocs = []
    for annee in annees:
        heures, x, y = donnees.charger_annee(annee, schema)
        dmop = donnees.lire_dmop(annee)
        if blocs and blocs[-1][0][-1] + donnees.H == heures[0]:
            avant = blocs.pop()
            heures, x, y = np.concatenate([avant[0], heures]), pd.concat([avant[1], x]), np.concatenate([avant[2], y])
            dmop = pd.concat([avant[3], dmop], ignore_index=True)
        blocs.append((heures, x, y, dmop))
    familles = variables.apprendre_familles([donnees.lire_dmop(a) for a in annees], config["familles_variante"])
    periodes = []
    for _, x, y, dmop in blocs:
        f, noms = construire(x, dmop, config, familles)
        periodes.append((f, y))
    return schema, periodes, noms, familles


def statistiques(periodes):
    f = np.concatenate([p for p, _ in periodes])
    y = np.concatenate([c for _, c in periodes])
    ecart = f.std(axis=0)
    return {"x_moyenne": f.mean(axis=0), "x_ecart": np.where(ecart < 1e-3, 1.0, ecart),
            "x_min": f.min(axis=0), "x_max": f.max(axis=0),
            "y_moyenne": np.nanmean(y, axis=0), "y_ecart": np.maximum(np.nanstd(y, axis=0), 1e-6)}


def standardiser(f, stats, normalisation="standard"):
    if normalisation == "minmax":
        f = np.clip(f, stats["x_min"], stats["x_max"])
        return np.clip((f - stats["x_moyenne"]) / stats["x_ecart"], -5.0, 5.0)
    return np.clip((f - stats["x_moyenne"]) / stats["x_ecart"], -BORNE, BORNE)


def probabilite_verite(epoque, config):
    """1 pendant les epoques_verite premières époques, puis linéaire jusqu'à 0 à la dernière."""
    debut, fin = config["epoques_verite"], config["epoques"] - 1
    return 1.0 if epoque < debut else max(0.0, 1.0 - (epoque - debut + 1) / max(fin - debut + 1, 1))


def plan_epoque(total, config, rng):
    """(débuts des flux, segments par époque) : chaque flux couvre un seizième des heures (en boucle) à partir
    d'un décalage aléatoire commun, tiré à chaque époque."""
    pas = int(np.ceil(total / config["flux"] / config["segment_h"]))
    decalage = int(rng.integers(0, total))
    debuts = (decalage + np.arange(config["flux"]) * (total // config["flux"])) % total
    return debuts, pas


def morceau(serie, debut, longueur):
    """longueur heures à partir de debut, en boucle sur la série."""
    return serie[(debut + np.arange(longueur)) % len(serie)]


def entrainer_un(periodes, stats, config, graine):
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    # Les périodes sont mises bout à bout : un flux qui passe la fin d'une période repart avec son état
    # (ponytail: une seule période dans tous les usages actuels ; couper les flux si plusieurs périodes).
    f = standardiser(np.concatenate([p for p, _ in periodes]), stats, config.get("normalisation", "standard"))
    y = np.concatenate([c for _, c in periodes])
    yn = (y - stats["y_moyenne"]) / stats["y_ecart"]
    n_circuits = y.shape[1]
    m_t, e_t = (torch.as_tensor(stats[k], dtype=torch.float32, device=APPAREIL) for k in ("y_moyenne", "y_ecart"))
    modele = Recette(f.shape[1], n_circuits, config).to(APPAREIL)
    optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    _, pas_par_epoque = plan_epoque(len(f), config, np.random.default_rng(0))
    plan = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr"], pct_start=config["montee"],
                                               total_steps=config["epoques"] * pas_par_epoque)
    debut_chrono, lot, seg = time.time(), config["flux"], config["segment_h"]
    for epoque in range(config["epoques"]):
        modele.train()
        debuts, _ = plan_epoque(len(f), config, rng)
        p_verite = probabilite_verite(epoque, config)
        h = torch.zeros(lot, config["taille"], device=APPAREIL)
        avant = np.stack([morceau(yn, d - 1, 1)[0] for d in debuts])  # vérité de l'heure précédant chaque flux
        precedent = torch.as_tensor(np.nan_to_num(avant), dtype=torch.float32, device=APPAREIL)
        pertes = []
        for k in range(pas_par_epoque):
            positions = debuts + k * seg
            x = torch.as_tensor(np.stack([morceau(f, d, seg) for d in positions]), dtype=torch.float32,
                                device=APPAREIL)
            cible = torch.as_tensor(np.stack([morceau(y, d, seg) for d in positions]), dtype=torch.float32,
                                    device=APPAREIL)
            p = 1.0 if k == 0 and config.get("premier_segment_verite") else p_verite
            sortie, h, precedent = modele.avancer(x, h, precedent, (cible - m_t) / e_t, p,
                                                  config["bruit_retour"])
            masque = torch.isfinite(cible)
            perte = ((sortie * e_t + m_t - torch.nan_to_num(cible)) ** 2)[masque].mean()
            optim.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optim.step()
            plan.step()
            h, precedent = h.detach(), precedent.detach()
            pertes.append(perte.item())
        if epoque % 10 == 0 or epoque == config["epoques"] - 1:
            print(f"  graine {graine} époque {epoque:3d} p_verite {p_verite:.2f} RMSE {np.mean(pertes) ** 0.5:.4f}"
                  f" ({time.time() - debut_chrono:.0f} s)", flush=True)
    return modele.cpu().eval()


def _entrainer_graine(arguments):
    torch.set_num_threads(1)
    return {k: v.half() for k, v in entrainer_un(*arguments).state_dict().items()}


def entrainer(annees, dossier, **options):
    config = {**CONFIG_DEFAUT, **options}
    schema, periodes, noms, familles = preparer(annees, config)
    stats = statistiques(periodes)
    processus = min(len(config["graines"]), int(os.environ.get("MONDE_PROCESSUS", os.cpu_count() or 1)))
    with ProcessPoolExecutor(processus, mp_context=multiprocessing.get_context("spawn")) as pool:
        etats = list(pool.map(_entrainer_graine, [(periodes, stats, config, g) for g in config["graines"]]))
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "variables": noms, "familles": familles, "circuits": schema["target_columns"],
                "stats": {k: torch.as_tensor(v) for k, v in stats.items()}, "etats": etats}, dossier / "recette.pt")
    print(f"{len(etats)} GRU recette entraîné(s) sur {', '.join(annees)} ({len(noms)} variables)")


@torch.no_grad()
def predire(test_dir, out, dossier, recalage=0.0, demi_vie=1440):
    """recalage : comme le boosting (baselines.boosting.recaler), décalage de chaque circuit d'une fraction
    de l'écart moyen entre power_init et la simulation libre du modèle sur ces mêmes heures, atténué avec
    une demi-vie en heures. 0 : recette telle quelle."""
    paquet = torch.load(Path(dossier) / "recette.pt", weights_only=True)
    config, stats = paquet["config"], {k: v.numpy() for k, v in paquet["stats"].items()}
    test = donnees.lire_dossier_test(test_dir)
    n_init = test["n_init"]
    if config["variables"] == "S":
        table = caracteristiques.construire(test["tables"], test["heures"], paquet["familles"], config["recette"])
        noms = colonnes_s(table, config)
        f = table[noms].to_numpy(dtype=np.float64)
    else:
        f, noms = construire(test["X"], test["dmop"], config, paquet["familles"])
    if noms != paquet["variables"]:
        raise ValueError("Variables différentes de celles de l'entraînement")
    x = torch.as_tensor(standardiser(f, stats, config.get("normalisation", "standard"))[None], dtype=torch.float32)
    verite = torch.as_tensor(((test["y_init"][:n_init] - stats["y_moyenne"]) / stats["y_ecart"])[None],
                             dtype=torch.float32)
    sorties = []
    for etat in paquet["etats"]:
        modele = Recette(x.shape[-1], len(paquet["circuits"]), config)
        modele.load_state_dict({k: v.float() for k, v in etat.items()})
        modele.eval()
        h, precedent = torch.zeros(1, config["taille"]), torch.zeros(1, len(paquet["circuits"]))
        _, h, precedent = modele.avancer(x[:, :n_init], h, precedent, verite, 1.0)  # 7 jours réels réinjectés
        sortie, _, _ = modele.avancer(x[:, n_init:], h, precedent)
        sortie = sortie[0].numpy() * stats["y_ecart"] + stats["y_moyenne"]
        if recalage:
            libre, _, _ = modele.avancer(x[:, :n_init], torch.zeros(1, config["taille"]),
                                         torch.zeros(1, len(paquet["circuits"])))
            libre = libre[0].numpy() * stats["y_ecart"] + stats["y_moyenne"]
            sortie = recaler(np.concatenate([libre, sortie]), test["y_init"], n_init, recalage, demi_vie)[n_init:]
        sorties.append(sortie)
    debut = np.where(np.isfinite(test["y_init"][:n_init]), test["y_init"][:n_init], stats["y_moyenne"])
    moyenne = np.clip(np.concatenate([debut, np.mean(sorties, axis=0)]), 0, None)
    return donnees.ecrire_predictions(test["heures"], moyenne, test["heures_a_predire"], paquet["circuits"], out)
