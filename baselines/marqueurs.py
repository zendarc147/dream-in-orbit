"""Porte de régime par marqueurs du planning, pour un circuit à modes (NPWD2531).

NPWD2531 reste presque toujours éteint (0.001), avec deux modes : un palier d'impulsions de
moyenne 0.1, que marque une famille de commandes (ATTTF410/420 dans les deux années), et des
campagnes à 0.6, propres à une année. Entraîné sur une année qui contient ces campagnes, un modèle
en moyenne prédit un faux palier de 0.1 partout où rien ne les explique.

Apprentissage (année d'entraînement seule) : liste de décision gloutonne. Chaque famille du
planning vue ≥ 20 fois donne une présence « au moins une commande dans ±24 h ». On ajoute la
famille qui baisse le plus la MSE quand ses heures libres prennent la moyenne du circuit, et on
s'arrête à 3 familles ou quand la baisse passe sous 1 %. Hors marqueur, le niveau est la médiane
du circuit : un mode rare et inexpliqué ne déplace pas le niveau de base.
Prédiction : là où une famille de la liste a une commande dans ±168 h, on garde la prédiction de
l'ensemble, qui suit l'heure des impulsions. Ailleurs, on prend la médiane, recalée sur
power_init avec une demi-vie de 1440 h (même recalage que le boosting).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.boosting import recaler
from dio import commandes, donnees, variables

MIN_HEURES = 200
MAX_MARQUEURS = 3
GAIN_MIN = 0.01
DEMI_FENETRES = (24, 168)


def famille_etendue(code):
    """Famille d'une commande : lettres de variante fusionnées ; MOCS.0000003738 -> MOCS."""
    m = re.match(r"^([A-Z]{4})\.\d+$", code)
    return m.group(1) if m else commandes.famille(code)


def apprendre_planning(tables):
    """Familles vues au moins MIN_FAMILLE fois sur l'entraînement et paires début/fin (XXXS / XXXE)."""
    comptes = pd.concat(tables).subsystem.map(famille_etendue).value_counts()
    familles = sorted(comptes[comptes >= variables.MIN_FAMILLE].index)
    paires = [(f, f[:3] + "E") for f in familles if re.fullmatch(r"[A-Z]{3}S", f) and f[:3] + "E" in familles]
    return {"familles": familles, "paires": paires}


def variables_regimes(dmop, heures, planning):
    """Comptes de chaque famille du planning dans [t - w, t + w] (heures entières)."""
    heures = np.asarray(heures, dtype=np.int64)
    table = dmop.assign(subsystem=dmop.subsystem.map(famille_etendue))
    comptes = commandes.impulsions(table, heures, planning["familles"])
    cumul = np.vstack([np.zeros((1, comptes.shape[1])), np.cumsum(comptes, axis=0)])
    t, n = np.arange(len(heures)), len(heures)
    sortie = {}
    for w in DEMI_FENETRES:
        fenetre = cumul[np.minimum(t + w + 1, n)] - cumul[np.maximum(t - w, 0)]
        sortie.update({f"plan_{f}_pm{w}h": fenetre[:, j] for j, f in enumerate(planning["familles"])})
    return pd.DataFrame(sortie, index=heures)


def presences(dmop, heures, planning, demi_fenetre):
    table = variables_regimes(dmop, heures, planning)
    return {f: table[f"plan_{f}_pm{demi_fenetre}h"].to_numpy() > 0 for f in planning["familles"]}


def niveaux(liste, defaut, presence, n):
    sortie, libre = np.full(n, defaut), np.ones(n, bool)
    for famille, niveau in liste:
        heures = presence[famille] & libre
        sortie[heures], libre = niveau, libre & ~heures
    return sortie


def apprendre_liste(y, presence):
    connus = np.isfinite(y)
    defaut = float(np.nanmedian(y))
    erreur = lambda liste: np.mean((y - niveaux(liste, defaut, presence, len(y)))[connus] ** 2)
    liste, courante = [], erreur([])
    while len(liste) < MAX_MARQUEURS:
        libre = ~np.any([presence[f] for f, _ in liste], axis=0) if liste else np.ones(len(y), bool)
        candidats = []
        for famille, present in presence.items():
            heures = present & libre & connus
            if heures.sum() >= MIN_HEURES:
                essai = liste + [(famille, float(y[heures].mean()))]
                candidats.append((erreur(essai), essai))
        if not candidats:
            break
        meilleure, essai = min(candidats, key=lambda c: c[0])
        if meilleure > courante * (1 - GAIN_MIN):
            break
        liste, courante = essai, meilleure
    return defaut, liste


def entrainer(annees, dossier, circuits=("NPWD2531",)):
    schema = donnees.charger_schema()
    tables = [donnees.lire_dmop(a) for a in annees]
    planning = apprendre_planning(tables)
    ys, presence = [], {}
    for annee, dmop in zip(annees, tables):
        heures, _, y = donnees.charger_annee(annee, schema)
        ys.append(y)
        for f, p in presences(dmop, heures, planning, 24).items():
            presence[f] = np.concatenate([presence.get(f, np.zeros(0, bool)), p])
    y = np.concatenate(ys)
    paquet = {"planning": planning, "circuits": {}}
    for c in circuits:
        defaut, liste = apprendre_liste(y[:, schema["target_columns"].index(c)], presence)
        paquet["circuits"][c] = {"defaut": defaut, "liste": liste}
        print(f"{c} : niveau hors marqueur {defaut:.4f}, marqueurs {liste}")
    Path(dossier).mkdir(parents=True, exist_ok=True)
    (Path(dossier) / "marqueurs.json").write_text(json.dumps(paquet, indent=2) + "\n")


def porte(test, dossier, demi_vie=1440):
    """{circuit: (niveau hors marqueur recalé, masque où garder l'ensemble)} sur la grille du test."""
    paquet = json.loads((Path(dossier) / "marqueurs.json").read_text())
    colonnes = donnees.charger_schema()["target_columns"]
    n = len(test["heures"])
    # Hors de ±168 h d'un marqueur, aucune présence à ±24 h : la liste y vaut son niveau par défaut.
    large = presences(test["dmop"], test["heures"], paquet["planning"], 168)
    sortie = {}
    for c, info in paquet["circuits"].items():
        liste = [tuple(m) for m in info["liste"]]
        j = colonnes.index(c)
        hors = np.full(n, info["defaut"])[:, None]
        hors = recaler(hors, test["y_init"][:, [j]], test["n_init"], 1.0, demi_vie)[:, 0]
        garder = np.any([large[f] for f, _ in liste], axis=0) if liste else np.zeros(n, bool)
        sortie[c] = (np.clip(hors, 0, None), garder)
    return sortie
