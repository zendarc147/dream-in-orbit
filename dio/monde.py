"""World model GRU : simule les 33 courants heure par heure à partir du contexte.

Architecture :
- encodeur : un GRU lit les 7 jours de power_init (courants + contexte) et produit
  l'état initial h0 du simulateur ;
- simulateur : un GRU à 2 couches avance heure par heure sur le contexte seul, sans
  jamais réinjecter ses propres prédictions (pas d'accumulation d'erreur par rebouclage) ;
- tête : une couche linéaire donne les 33 courants à l'échelle brute ;
- branche de commandes : les commandes AOOO/ATTT (commandes.py) passent par un petit réseau
  à part dont la sortie s'ajoute au seul NPWD2881 (le GRU ne les voit pas) ;
- correction de niveau : l'écart entre power_init et la prédiction du simulateur sur ces
  7 jours, ajouté à la sortie avec un poids appris, s'estompe selon deux demi-vies apprises.
Option non retenue (regime_partage, désactivée par défaut) : tête de régime commune à NPWD2551,
NPWD2451 et NPWD2851, espérance d'un mélange de niveaux, cross-entropie auxiliaire sur des seuils
appris à l'entraînement. Validée sans gain dans les deux directions.

Entraînement : épisodes aléatoires de 7 jours de démarrage + N jours de simulation,
perte MSE sur l'échelle brute (comme la métrique), heures inconnues masquées.
Les épisodes longs sont découpés en morceaux (BPTT tronquée) : l'état caché passe
d'un morceau au suivant, un pas d'optimisation par morceau. Le simulateur apprend
ainsi à rester stable sur l'horizon réel (un an) et pas seulement sur 60 jours.
Chaque variable est bornée à sa plage d'entraînement : pas d'extrapolation hors domaine.
Plusieurs graines forment un ensemble : moyenne = prédiction, écart-type = incertitude.
"""
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from . import commandes
from . import donnees
from . import variables

CONFIG_DEFAUT = {
    "variables": "v1",       # jeu de variables d'entrée (voir variables.py)
    "commandes": {"prefixes": ["AOOO", "ATTT"]},  # options de commandes.py (None : aucune variable de commandes)
    "branche_commandes": ["NPWD2881"],  # circuits recevant seuls les variables de commandes (None : elles vont au GRU)
    "sans": [],              # groupes de variables retirés (ablations)
    "memoire": True,         # False : ni encodeur ni récurrence ni correction de niveau (ablation)
    "bruit": 0.3,            # écart-type du bruit gaussien ajouté aux entrées normalisées (entraînement)
    "masque_saaf": 0.0,      # part des heures d'une copie de chaque période où SAAF est masqué par blocs (0 : pas de copie)
    "abandon_etats": 0.0,    # probabilité de mettre à 0 toutes les variables dmopcmd_etat_* d'un épisode (entraînement)
    "regime_partage": False,  # tête de régime commune à CIBLES_REGIME : mélange de K niveaux par circuit
    "regime_k": 2,           # nombre de régimes de la tête partagée
    "lambda_regime": 0.05,   # poids de la cross-entropie de régime (entraînement, avec regime_partage)
    "abandon_regime_contexte": 0.0,  # probabilité par épisode de masquer un bloc d'éclipses, modes FTL et DMOP
    "curriculum_jours": None,  # ex. [180, 365, 540] : jours_sim par parts égales des pas (None : jours_sim)
    "taille": 128,
    "couches": 2,
    "dropout": 0.1,
    "jours_init": 7,
    "jours_sim": 365,        # longueur d'un épisode de simulation
    "morceau_h": 720,        # BPTT tronquée par morceaux de 30 jours (0 : un seul morceau)
    "lot": 32,
    "pas": 600,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "borne_niveau_prediction": 10.0,  # borne élargie seulement à la prédiction, circuits variables à l'entraînement
    "ema": 0.99,             # décroissance de la moyenne exponentielle des poids (0 : pas de moyenne)
    "graines": [0, 1, 2, 3, 4],
    "threads": 4,
}
H_JOUR = 24
APPAREIL = "cuda" if torch.cuda.is_available() else "cpu"
CIBLES_REGIME = ["NPWD2551", "NPWD2451", "NPWD2851"]  # circuits de la tête de régime (regime_partage)


def kmeans_log(valeurs, k):
    """k-moyennes 1D déterministes sur log10 (valeurs ≥ 1e-4) : (centres, seuils) à l'échelle brute,
    seuil = milieu de deux centres voisins sur l'échelle log."""
    v = np.log10(np.maximum(valeurs[np.isfinite(valeurs)], 1e-4))
    centres = np.quantile(v, (np.arange(k) + 0.5) / k)
    for _ in range(100):
        classe = np.argmin(np.abs(v[:, None] - centres), axis=1)
        nouveaux = np.array([v[classe == i].mean() if (classe == i).any() else centres[i] for i in range(k)])
        if np.allclose(nouveaux, centres):
            break
        centres = nouveaux
    return 10 ** centres, 10 ** ((centres[1:] + centres[:-1]) / 2)


def seuils_regime(y):
    """Seuil bas/haut de chaque colonne de y (heures, circuits), appris sur ces seules heures."""
    return np.array([kmeans_log(y[:, j], 2)[1][0] for j in range(y.shape[1])])


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
    def __init__(self, n_entrees, n_circuits, taille, couches, dropout, y_stats, memoire=True, branche=None,
                 regime=None):
        super().__init__()
        # y_stats = (moyenne, écart) des courants, pour comparer power_init (normalisé) à la
        # prédiction du modèle sur les mêmes heures (correction de niveau sur le résidu).
        self.borne_niveau = 5.0  # élargie par circuit au chargement (borne_niveau_prediction)
        self.register_buffer("y_moyenne", torch.as_tensor(y_stats[0], dtype=torch.float32), persistent=False)
        self.register_buffer("y_ecart", torch.as_tensor(y_stats[1], dtype=torch.float32), persistent=False)
        # Branche de commandes : les variables de commandes (indices branche["entrees"]) ne vont
        # pas au GRU mais à un petit réseau heure par heure dont la sortie s'ajoute aux seuls
        # circuits branche["circuits"]. Les autres circuits ne peuvent pas s'en servir.
        self.branche = branche
        if branche:
            commandes = set(branche["entrees"])
            self.principales = [i for i in range(n_entrees) if i not in commandes]
            n_entrees = len(self.principales)
            self.reseau_commandes = nn.Sequential(nn.Linear(len(commandes), 32), nn.GELU(),
                                                  nn.Linear(32, len(branche["circuits"])))
            nn.init.zeros_(self.reseau_commandes[-1].weight)
            nn.init.zeros_(self.reseau_commandes[-1].bias)
        self.memoire = memoire
        if memoire:
            # Correction de niveau : écart normalisé de chaque circuit sur power_init, multiplié
            # par un poids appris et atténué par une demi-vie apprise (initialisée à 30 jours).
            # Le poids part de 0 : sans signal, le modèle de base est inchangé.
            self.poids_niveau = nn.Parameter(torch.zeros(n_circuits))
            self.log_demi_vie = nn.Parameter(torch.full((n_circuits,), float(np.log(720.0))))
            # Second terme, plus lent : une part du décalage de niveau peut durer des mois.
            self.poids_lent = nn.Parameter(torch.zeros(n_circuits))
            self.log_demi_vie_lente = nn.Parameter(torch.full((n_circuits,), float(np.log(4320.0))))
        self.couches, self.taille = couches, taille
        self.projection = nn.Sequential(nn.Linear(n_entrees, taille), nn.GELU())
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
        # Tête de régime partagée (regime["circuits"], regime["k"]) : une probabilité de régime commune
        # et, pour chaque circuit visé, K niveaux conditionnels ; sortie = espérance du mélange.
        # Créée après la tête : sans régime, l'initialisation des autres couches est inchangée.
        self.regime = regime
        if regime:
            self.tete_regime = nn.Sequential(nn.Linear(taille, 32), nn.GELU(), nn.Linear(32, regime["k"]))
            nn.init.zeros_(self.tete_regime[-1].weight)
            nn.init.zeros_(self.tete_regime[-1].bias)
            self.niveaux_regime = nn.Linear(taille, len(regime["circuits"]) * regime["k"])

    @torch.no_grad()
    def initialiser_regime(self, centres):
        """Niveaux de départ = tête actuelle, décalée des centres de régime appris (recentrés) : avec
        des probabilités uniformes, la sortie initiale est exactement celle de la tête actuelle."""
        cibles, k = self.regime["circuits"], self.regime["k"]
        centres = torch.as_tensor(centres, dtype=torch.float32, device=self.tete.bias.device)
        self.niveaux_regime.weight.copy_(self.tete.weight[cibles].repeat_interleave(k, dim=0))
        biais = self.tete.bias[cibles][:, None] + centres - centres.mean(dim=1, keepdim=True)
        self.niveaux_regime.bias.copy_(biais.flatten())

    def sortie_tete(self, etat):
        """Courants (B, T, C) et logits de régime (B, T, K), None sans tête de régime."""
        sortie = self.tete(etat)
        if not self.regime:
            return sortie, None
        logits = self.tete_regime(etat)
        niveaux = self.niveaux_regime(etat).unflatten(-1, (len(self.regime["circuits"]), self.regime["k"]))
        melange = (logits.softmax(-1).unsqueeze(-2) * niveaux).sum(-1)
        cibles = torch.as_tensor(self.regime["circuits"], device=sortie.device)
        return sortie.index_copy(-1, cibles, melange), logits

    def etat_initial(self, x_init, y_init):
        """x_init (B, Ti, F), y_init (B, Ti, 2C) courants normalisés + masque -> h0 (couches, B, taille)."""
        if not self.memoire:
            return None
        x = x_init[..., self.principales] if self.branche else x_init
        _, h = self.encodeur(torch.cat([x, y_init], dim=-1))
        h0 = torch.tanh(self.vers_h0(h[-1]))
        return h0.view(-1, self.couches, self.taille).transpose(0, 1).contiguous()

    def resume_niveau(self, x_init, y_init):
        """(B, C) : écart moyen entre power_init et la prédiction du simulateur sur ces heures
        (partant d'un état nul), sans gradient. None sans mémoire (pas de correction)."""
        if not self.memoire:
            return None
        n = y_init.shape[-1] // 2
        masque = y_init[..., n:]
        # Borné comme les entrées : un circuit presque constant a un écart-type minuscule.
        residu = (self.ecarts_init(x_init, y_init).sum(dim=1) / masque.sum(dim=1).clamp(min=1))
        return residu.clamp(-self.borne_niveau, self.borne_niveau)

    @torch.no_grad()
    def ecarts_init(self, x_init, y_init):
        """(B, Ti, C) : power_init normalisé moins la prédiction du simulateur sur ces heures
        (partant d'un état nul), 0 aux heures inconnues."""
        x = x_init[..., self.principales] if self.branche else x_init
        etat = self.simulateur(self.projection(x))[0]
        x_cmd = x_init[..., self.branche["entrees"]] if self.branche else None
        prediction = self.ajouter_commandes(self.sortie_tete(etat)[0], x_cmd)
        n = y_init.shape[-1] // 2
        valeurs, masque = y_init[..., :n], y_init[..., n:]
        return (valeurs - (prediction - self.y_moyenne) / self.y_ecart) * masque

    def simuler(self, x_sim, h, resume=None, debut=0, avec_logits=False):
        """Avancer sur le contexte seul : retourne (courants (B, Ts, C), état final). Avec avec_logits,
        ajoute les logits de régime calculés sur l'état détaché (perte auxiliaire : son gradient
        n'atteint que la tête de régime, jamais le GRU ni les autres circuits)."""
        x_cmd = None
        if self.branche:
            x_cmd = x_sim[..., self.branche["entrees"]]
            x_sim = x_sim[..., self.principales]
        z = self.projection(x_sim)
        if self.memoire:
            etat, h = self.simulateur(z, h)
        else:
            etat = self.simulateur(z)
        sortie = self.ajouter_commandes(self.sortie_tete(etat)[0], x_cmd)
        logits = self.tete_regime(etat.detach()) if avec_logits and self.regime else None
        if not self.memoire:
            return (sortie, None, logits) if avec_logits else (sortie, None)
        # debut : heure de simulation du premier pas de ce morceau (BPTT tronquée).
        t = debut + torch.arange(sortie.shape[1], device=sortie.device, dtype=sortie.dtype)
        attenuation = torch.exp2(-t[:, None] / self.log_demi_vie.exp())
        sortie = sortie + resume[:, None] * self.poids_niveau * attenuation
        lente = torch.exp2(-t[:, None] / self.log_demi_vie_lente.exp())
        sortie = sortie + resume[:, None] * self.poids_lent * lente
        return (sortie, h, logits) if avec_logits else (sortie, h)

    def ajouter_commandes(self, sortie, x_cmd):
        """Ajouter la sortie de la branche de commandes aux circuits qu'elle vise."""
        if x_cmd is None:
            return sortie
        cibles = torch.as_tensor(self.branche["circuits"], device=sortie.device)
        return sortie.index_add(-1, cibles, self.reseau_commandes(x_cmd))

    def forward(self, x_init, y_init, x_sim):
        return self.simuler(x_sim, self.etat_initial(x_init, y_init), self.resume_niveau(x_init, y_init))[0]


def masquer_saaf(x, schema, fraction, graine=0):
    """Copie de la table horaire où des blocs contigus (1 à 720 h, longueur log-uniforme)
    couvrant `fraction` des heures ressemblent à une panne de télémétrie SAAF : valeurs de
    remplissage du schéma, couverture 0, indicateur manquant à 1."""
    x = x.copy()
    rng = np.random.default_rng(graine)
    n, cible = len(x), int(round(fraction * len(x)))
    masque = np.zeros(n, dtype=bool)
    while masque.sum() < cible:
        # Dernier bloc raccourci pour ne pas dépasser la fraction demandée.
        longueur = min(int(round(np.exp(rng.uniform(0.0, np.log(720.0))))), cible - int(masque.sum()), n)
        debut = rng.integers(0, n - longueur + 1)
        masque[debut:debut + longueur] = True
    for colonne in [c for c in x.columns if c.startswith("saaf_")]:
        x.loc[masque, colonne] = schema["fill_values"][colonne]
    x.loc[masque, "saaf_coverage_fraction"] = 0.0
    x.loc[masque, "saaf_missing"] = 1
    return x


def abandonner_etats(x, etats, probabilite):
    """Mettre à 0 (espace normalisé), ensemble et sur tout l'épisode, les variables d'état
    des commandes d'une partie des épisodes du lot x (B, T, F). Entraînement seulement."""
    if not probabilite or not etats:
        return x
    garde = (torch.rand(x.shape[0], 1, 1, device=x.device) >= probabilite).to(x.dtype)
    x = x.clone()
    x[..., etats] = x[..., etats] * garde
    return x


def abandonner_contexte(x, indices, probabilite, debut):
    """Pour une partie des épisodes du lot x (B, T, F), mettre à 0 (espace normalisé) les variables
    `indices` sur un bloc contigu après l'heure `debut` (24 h à toute la simulation, longueur
    log-uniforme). Entraînement seulement ; power_init et les cibles ne passent pas ici."""
    if not probabilite or not indices:
        return x
    x = x.clone()
    n = x.shape[1] - debut
    for b in torch.nonzero(torch.rand(x.shape[0]) < probabilite).flatten().tolist():
        bas = math.log(min(24, n))
        longueur = int(round(math.exp(torch.empty(1).uniform_(bas, math.log(n)).item())))
        s = debut + int(torch.randint(0, n - longueur + 1, (1,)))
        x[b, s:s + longueur, indices] = 0.0
    return x


def stats_regime(y, cibles, k):
    """Régimes appris sur les seules heures d'entraînement y (heures, circuits) : seuils (n, K-1) et
    centres (n, K) de k-moyennes 1D log par circuit visé, poids de classe (K) inverses des fréquences
    de l'étiquette commune (médiane des régimes des circuits, majorité pour K = 2)."""
    centres, seuils = zip(*(kmeans_log(y[:, j], k) for j in cibles))
    seuils = np.array(seuils)
    etiquettes = etiquettes_regime(torch.as_tensor(y[:, cibles]), torch.as_tensor(seuils))
    connues = etiquettes[torch.isfinite(torch.as_tensor(y[:, cibles])).all(-1)].numpy()
    frequences = np.maximum(np.bincount(connues, minlength=k) / max(len(connues), 1), 1e-3)
    return {"regime_seuils": seuils, "regime_centres": np.array(centres),
            "regime_poids": 1.0 / (k * frequences)}


def etiquettes_regime(y, seuils):
    """Étiquette de régime commune (forme de y sans le dernier axe) : médiane, sur les circuits, du
    nombre de seuils dépassés."""
    return (y.unsqueeze(-1) >= seuils.to(y.dtype)).sum(-1).median(dim=-1).values


def perte_regime(logits, y, seuils, poids):
    """Cross-entropie pondérée entre logits (B, T, K) et l'étiquette commune des cibles y (B, T, n) ;
    heures où un circuit est inconnu exclues."""
    connues = torch.isfinite(y).all(-1)
    if not connues.any():
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[connues], etiquettes_regime(y, seuils)[connues], weight=poids)


def normaliser_courants(y, moyenne, ecart):
    """Courants normalisés (NaN -> 0) concaténés à leur masque de présence."""
    connus = np.isfinite(y)
    yn = np.where(connus, (y - moyenne) / ecart, 0.0)
    return np.concatenate([yn, connus.astype(np.float64)], axis=-1)


def preparer_annees(annees, config):
    schema = donnees.charger_schema()
    tables = {annee: donnees.lire_dmop(annee) for annee in annees}
    blocs = []
    for annee in annees:
        heures, x, y = donnees.charger_annee(annee, schema)
        dmop = tables[annee]
        if blocs and blocs[-1][0][-1] + donnees.H == heures[0]:
            # Année qui suit la précédente sans trou : une seule période, les variables
            # glissantes et les épisodes passent la frontière comme dans une vraie simulation.
            avant = blocs.pop()
            heures, x, y = np.concatenate([avant[0], heures]), pd.concat([avant[1], x]), np.concatenate([avant[2], y])
            dmop = pd.concat([avant[3], dmop], ignore_index=True)
        blocs.append((heures, x, y, dmop))
    voc_commandes = None
    if config.get("commandes") is not None:
        voc_commandes = commandes.apprendre([(dmop, heures, y) for heures, _, y, dmop in blocs],
                                            schema["target_columns"], config["commandes"])
    periodes, copies = [], []
    for i, (_, x, y, dmop) in enumerate(blocs):
        f, noms = variables.construire(x, config["variables"], config["sans"], dmop, voc_commandes)
        periodes.append((f, y))
        if config.get("masque_saaf", 0.0) > 0:
            # Copie masquée en plus de l'originale ; les statistiques ne voient que les originales.
            x_masque = masquer_saaf(x, schema, config["masque_saaf"], graine=i)
            copies.append((variables.construire(x_masque, config["variables"], config["sans"], dmop,
                                                voc_commandes)[0], y))
    return schema, periodes, noms, voc_commandes, copies


def entrainer_un(periodes, stats, config, graine, n_circuits):
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    ti, ts = config["jours_init"] * H_JOUR, config["jours_sim"] * H_JOUR
    morceau = config["morceau_h"] or ts
    appareil = APPAREIL
    regime = config.get("_regime")
    modele = MondeGRU(int(stats["garder"].sum()), n_circuits, config["taille"], config["couches"],
                      config["dropout"], (stats["y_moyenne"], stats["y_ecart"]), config["memoire"],
                      config.get("_branche"), regime).to(appareil)
    with torch.no_grad():
        modele.tete.bias.copy_(torch.as_tensor(stats["y_moyenne"], dtype=torch.float32))
    if regime:
        modele.initialiser_regime(stats["regime_centres"])
        seuils = torch.as_tensor(stats["regime_seuils"], dtype=torch.float32, device=appareil)
        poids_classes = torch.as_tensor(stats["regime_poids"], dtype=torch.float32, device=appareil)
    optim = torch.optim.AdamW(modele.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    plan = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config["lr"], total_steps=config["pas"],
                                               pct_start=0.1)
    # Moyenne exponentielle des poids : le modèle rendu est la moyenne des derniers pas,
    # moins sensible au bruit des derniers lots qu'un seul point final.
    moyen = (torch.optim.swa_utils.AveragedModel(
        modele, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config["ema"]))
        if config.get("ema") else None)
    debut, cumul, pas = time.time(), 0.0, 0
    modele.train()
    while pas < config["pas"]:
        if config.get("curriculum_jours"):
            # Curriculum déterministe : parts égales des pas, l'épisode en cours finit sa longueur.
            etapes = config["curriculum_jours"]
            ts = etapes[min(len(etapes) * pas // config["pas"], len(etapes) - 1)] * H_JOUR
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
        x = abandonner_etats(x, config.get("_etats", []), config.get("abandon_etats", 0.0))
        x = abandonner_contexte(x, config.get("_contexte", []), config.get("abandon_regime_contexte", 0.0), ti)
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
                resume = modele.resume_niveau(x[:, :ti], y_init)
            elif h is not None:
                h = h.detach()
            sortie, h, logits = modele.simuler(x[:, ti + a:ti + a + morceau], h, resume, a, avec_logits=True)
            c = cible[:, a:a + morceau]
            masque = torch.isfinite(c)
            if not masque.any():
                continue
            perte = ((sortie - torch.nan_to_num(c)) ** 2)[masque].mean()
            if regime and config["lambda_regime"]:
                perte = perte + config["lambda_regime"] * perte_regime(
                    logits, c[..., regime["circuits"]], seuils, poids_classes)
            optim.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optim.step()
            plan.step()
            if moyen is not None:
                moyen.update_parameters(modele)
            cumul = 0.98 * cumul + 0.02 * perte.item() if pas else perte.item()
            if pas % 100 == 0 or pas == config["pas"] - 1:
                print(f"  graine {graine} pas {pas:5d} RMSE lissée {cumul ** 0.5:.4f}"
                      f" ({time.time() - debut:.0f} s)", flush=True)
            pas += 1
    return (moyen.module if moyen is not None else modele).cpu()


def _entrainer_graine(arguments):
    """Travailleur d'un processus : une graine, un thread (le GRU passe mal à l'échelle en threads)."""
    periodes, stats, config, graine, n_circuits = arguments
    torch.set_num_threads(1)
    modele = entrainer_un(periodes, stats, config, graine, n_circuits)
    return {k: v.half() for k, v in modele.state_dict().items()}


def entrainer(annees, dossier, **options):
    config = {**CONFIG_DEFAUT, **options}
    torch.set_num_threads(int(os.environ.get("MONDE_THREADS", config["threads"])))
    schema, periodes, noms, voc_commandes, copies = preparer_annees(annees, config)
    f_tout = np.concatenate([f for f, _ in periodes])
    y_tout = np.concatenate([y for _, y in periodes])
    stats = calculer_stats(f_tout, y_tout)
    gardees = [n for n, g in zip(noms, stats["garder"]) if g]
    if config.get("branche_commandes") and any(n.startswith("dmopcmd_") for n in gardees):
        config["_branche"] = {
            "entrees": [i for i, n in enumerate(gardees) if n.startswith("dmopcmd_")],
            "circuits": [schema["target_columns"].index(c) for c in config["branche_commandes"]],
        }
    config["_etats"] = [i for i, n in enumerate(gardees) if n.startswith("dmopcmd_etat_")]
    if config.get("abandon_regime_contexte"):
        # Contexte qui peut rendre le régime trop déterministe : éclipses, modes de pointage FTL, DMOP.
        config["_contexte"] = [i for i, n in enumerate(gardees)
                               if n.startswith(variables.GROUPES["eclipses"] + ("dmop_", "dmopcmd_etat_"))
                               or (n.startswith("ftl_") and not n.startswith(("ftl_coverage", "ftl_comms",
                                                                              "ftl_overlap")))]
    if config.get("regime_partage"):
        cibles = [schema["target_columns"].index(c) for c in CIBLES_REGIME]
        config["_regime"] = {"circuits": cibles, "k": config["regime_k"]}
        # Statistiques sérialisées, apprises sur les périodes d'entraînement originales seulement.
        stats.update(stats_regime(y_tout, cibles, config["regime_k"]))
    periodes = periodes + copies
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
        "config": config, "variables": noms, "commandes": voc_commandes, "circuits": schema["target_columns"],
        "annees": list(annees),
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
                          config["couches"], config["dropout"], (stats["y_moyenne"], stats["y_ecart"]),
                          config["memoire"], config.get("_branche"), config.get("_regime"))
        modele.load_state_dict({k: v.float() for k, v in etat.items()})
        if config.get("borne_niveau_prediction") is not None:
            borne = float(config["borne_niveau_prediction"])
            if not np.isfinite(borne) or borne < modele.borne_niveau:
                raise ValueError("borne_niveau_prediction doit être finie et au moins égale à 5")
            # Circuits variables (écart-type appris ≥ 0.01) ; aucune consommation future.
            modele.borne_niveau = torch.where(torch.as_tensor(stats["y_ecart"]) >= .01,
                                               borne, modele.borne_niveau).float()
        modeles.append(modele.eval())
    return paquet, config, stats, modeles


@torch.no_grad()
def simuler(dossier, x_df, y_init, n_init, dmop=None):
    """Retourner (moyenne, écart-type) de l'ensemble sur toute la grille (heures init comprises)."""
    paquet, config, stats, modeles = charger(dossier)
    torch.set_num_threads(config["threads"])
    f, noms = variables.construire(x_df, config["variables"], config["sans"], dmop, paquet.get("commandes"))
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
