"""Variables d'entrée du world model, calculées à partir du X horaire de prepare.py.

v1 : les 65 variables de prepare.py telles quelles.
v2 : v1 + fenêtres glissantes passées ET futures (le planning est connu d'avance)
     sur éclipses, communications, modes ftl et commandes dmop ; heures depuis le
     dernier et jusqu'au prochain événement ; phase orbitale ; cos/sin des angles saaf.
v2r : v2 réduite aux éclipses et communications (fenêtres, depuis/jusqu'à), phase
      orbitale et cos/sin saaf.
Option etats_dmop : état « dernière commande » de chaque sous-système (dmop_etats.py).
Option commandes : impulsions et états décalés des commandes AOOO/ATTT (commandes.py).
Les groupes peuvent être retirés pour les ablations : eclipses, saaf, dmop, ftl, ltdata, orbite.
"""
import numpy as np
import pandas as pd

from . import commandes
from . import dmop_etats

FENETRES = (3, 6, 12, 24)  # la fenêtre de 1 h est la valeur horaire elle-même (déjà dans v1)
PLAFOND_H = 72             # au-delà, « depuis/jusqu'à » est plafonné

GROUPES = {
    "eclipses": ("evtf_OCC", "evtf_MAR_", "ltdata_eclipseduration", "ltdata_occultationduration", "ecl_"),
    "saaf": ("saaf_",),
    "dmop": ("dmop_", "dmopetat_", "dmopcmd_", "dmopatmb_", "dmopmasque_"),
    "ftl": ("ftl_",),
    "ltdata": ("ltdata_",),
    "orbite": ("evtf_PERICENTRE", "evtf_APOCENTRE", "orbite_"),
}


def groupe_de(nom):
    return [g for g, prefixes in GROUPES.items() if nom.startswith(prefixes)]


def sommes_glissantes(serie, w):
    """Moyennes sur les w heures passées (t-w+1..t) et futures (t..t+w-1), bords tronqués."""
    n = len(serie)
    cumul = np.concatenate([[0.0], np.cumsum(serie)])
    t = np.arange(n)
    debut_passe = np.maximum(t + 1 - w, 0)
    passe = (cumul[t + 1] - cumul[debut_passe]) / (t + 1 - debut_passe)
    fin_futur = np.minimum(t + w, n)
    futur = (cumul[fin_futur] - cumul[t]) / (fin_futur - t)
    return passe, futur


def heures_depuis_jusqua(actif):
    """Heures depuis la dernière heure active et jusqu'à la prochaine (0 si active), plafonnées."""
    n = len(actif)
    t = np.arange(n, dtype=float)
    dernier = np.maximum.accumulate(np.where(actif, t, -np.inf))
    prochain = np.minimum.accumulate(np.where(actif, t, np.inf)[::-1])[::-1]
    return np.minimum(t - dernier, PLAFOND_H), np.minimum(prochain - t, PLAFOND_H)


def series_a_fenetrer(x, reduite=False):
    noms = [c for c in x.columns if c.startswith(("evtf_OCC", "evtf_MAR_")) and c.endswith("_minutes")]
    noms += ["ftl_comms_minutes"]
    if reduite:
        return noms
    noms += [c for c in x.columns if c.startswith("ftl_") and c.endswith("_minutes")
             and c not in ("ftl_comms_minutes", "ftl_coverage_minutes", "ftl_overlap_minutes")]
    noms += [c for c in x.columns if c.startswith("dmop_") and c.endswith("_count")]
    return noms


def ajouter_v2(x, reduite=False):
    nouvelles = {}
    for nom in series_a_fenetrer(x, reduite):
        serie = x[nom].to_numpy(dtype=np.float64)
        for w in FENETRES:
            passe, futur = sommes_glissantes(serie, w)
            nouvelles[f"{nom}_passe{w}h"] = passe
            nouvelles[f"{nom}_futur{w}h"] = futur
        depuis, jusqua = heures_depuis_jusqua(serie > 0)
        nouvelles[f"{nom}_depuis_h"] = depuis
        nouvelles[f"{nom}_jusqua_h"] = jusqua
    for code in ("PERICENTRE", "APOCENTRE"):
        depuis, jusqua = heures_depuis_jusqua(x[f"evtf_{code}_PASSAGE_count"].to_numpy() > 0)
        nouvelles[f"orbite_{code.lower()}_depuis_h"] = depuis
        nouvelles[f"orbite_{code.lower()}_jusqua_h"] = jusqua
    # Phase dans l'orbite entre deux péricentres (0 au péricentre), en cos/sin.
    depuis, jusqua = nouvelles["orbite_pericentre_depuis_h"], nouvelles["orbite_pericentre_jusqua_h"]
    phase = 2 * np.pi * depuis / np.maximum(depuis + jusqua, 1)
    nouvelles["orbite_phase_cos"], nouvelles["orbite_phase_sin"] = np.cos(phase), np.sin(phase)
    for angle in ("sa", "sx", "sy", "sz"):
        radians = np.deg2rad(x[f"saaf_{angle}_mean"].to_numpy(dtype=np.float64))
        nouvelles[f"saaf_{angle}_cos"], nouvelles[f"saaf_{angle}_sin"] = np.cos(radians), np.sin(radians)
    return pd.concat([x, pd.DataFrame(nouvelles, index=x.index)], axis=1)


def construire(x, version="v1", sans=(), dmop=None, vocabulaire=None, voc_commandes=None):
    """Retourner (tableau float64 (heures, variables), noms des variables).

    Avec un vocabulaire dmop (appris sur l'entraînement), ajoute l'état des sous-systèmes
    calculé sur la table dmop brute couvrant les mêmes heures.
    """
    if version == "v1":
        table = x.copy()
    elif version in ("v2", "v2r"):
        table = ajouter_v2(x, reduite=version == "v2r")
    else:
        raise ValueError(f"Version de variables inconnue : {version}")
    if vocabulaire:
        etats = dmop_etats.construire(dmop, x.index.to_numpy(), vocabulaire)
        table = pd.concat([table, etats.set_axis(table.index)], axis=1)
    if voc_commandes:
        table = pd.concat([table, commandes.construire(dmop, x.index.to_numpy(), voc_commandes)
                           .set_axis(table.index)], axis=1)
    garder = [c for c in table.columns if not set(groupe_de(c)) & set(sans)]
    table = table[garder]
    valeurs = table.to_numpy(dtype=np.float64)
    if not np.isfinite(valeurs).all():
        raise ValueError("Variables non finies")
    return valeurs, list(table.columns)
