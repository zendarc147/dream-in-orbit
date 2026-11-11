"""Variables d'entrée du world model, calculées à partir du X horaire de prepare.py.

v1 : les 65 variables de prepare.py telles quelles.
v2r : v1 + fenêtres glissantes passées ET futures (le planning est connu d'avance) et
      heures depuis le dernier et jusqu'au prochain événement, sur les éclipses et les
      communications ; phase orbitale ; cos/sin des angles saaf.
Option commandes : impulsions et états décalés des commandes AOOO/ATTT (commandes.py).
Option physique : flux solaire par face (constante solaire × cosinus éclairé), part hors ombre
      et leurs moyennes passées 6/24/72 h (inertie thermique), tirés de saaf, ltdata et evtf.
Option familles : comptes horaires par famille de commande dmop (code sans lettre de variante),
      vocabulaire appris sur l'entraînement.
Les groupes peuvent être retirés pour les ablations : eclipses, saaf, dmop.
"""
import numpy as np
import pandas as pd

from . import commandes

FENETRES = (3, 6, 12, 24)  # la fenêtre de 1 h est la valeur horaire elle-même (déjà dans v1)
PLAFOND_H = 72             # au-delà, « depuis/jusqu'à » est plafonné
FENETRES_THERMIQUES = (6, 24, 72)
MIN_FAMILLE = 20           # une famille de commandes doit être vue au moins 20 fois à l'entraînement
MIN_FAMILLE_VARIANTE = 50  # familles commandes.famille_variante : au moins 50 fois dans une période d'entraînement

GROUPES = {
    "eclipses": ("evtf_OCC", "evtf_MAR_", "ltdata_eclipseduration", "ltdata_occultationduration", "ecl_"),
    "saaf": ("saaf_",),
    "dmop": ("dmop_", "dmopcmd_", "dmopfam_"),
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


def series_a_fenetrer(x):
    noms = [c for c in x.columns if c.startswith(("evtf_OCC", "evtf_MAR_")) and c.endswith("_minutes")]
    return noms + ["ftl_comms_minutes"]


def ajouter_v2r(x):
    nouvelles = {}
    for nom in series_a_fenetrer(x):
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
        if f"saaf_{angle}_cos" not in x:  # déjà fourni (moyenne des cosinus des mesures, contexte.soleil)
            nouvelles[f"saaf_{angle}_cos"] = np.cos(radians)
        nouvelles[f"saaf_{angle}_sin"] = np.sin(radians)
    return pd.concat([x, pd.DataFrame(nouvelles, index=x.index)], axis=1)


def physiques(x):
    """Flux solaire par face (constante solaire × cosinus éclairé), flux hors ombre et moyennes
    passées de ces flux et de l'ombre. Lit seulement saaf, ltdata et evtf."""
    soleil = x["ltdata_solarconstantmars"].to_numpy(dtype=np.float64)
    ombre = np.clip((x["evtf_MAR_UMBRA_minutes"] + 0.5 * x["evtf_MAR_PENUMBRA_minutes"])
                    .to_numpy(dtype=np.float64) / 60, 0, 1)
    cos = {a: x[f"saaf_{a}_cos"].to_numpy(dtype=np.float64) if f"saaf_{a}_cos" in x
           else np.cos(np.deg2rad(x[f"saaf_{a}_mean"].to_numpy(dtype=np.float64))) for a in ("sa", "sx", "sy", "sz")}
    faces = {"sa": np.maximum(cos["sa"], 0)}
    for a in ("sx", "sy", "sz"):
        faces[f"{a}_plus"], faces[f"{a}_moins"] = np.maximum(cos[a], 0), np.maximum(-cos[a], 0)
    sortie = {"ombre": ombre}
    for nom, c in faces.items():
        sortie[f"flux_{nom}"] = soleil * c
        sortie[f"flux_{nom}_eclaire"] = soleil * c * (1 - ombre)
    lentes = ["ombre"] + [f"flux_{nom}_eclaire" for nom in faces]
    for w in FENETRES_THERMIQUES:
        for nom in lentes:
            sortie[f"{nom}_passe{w}h"] = sommes_glissantes(sortie[nom], w)[0]
    return pd.DataFrame(sortie, index=x.index)


def apprendre_familles(tables, variante=False):
    """Familles de commandes (code sans lettre de variante) vues au moins MIN_FAMILLE fois.
    variante : familles commandes.famille_variante vues au moins MIN_FAMILLE_VARIANTE fois dans une
    des périodes, union des vocabulaires des périodes."""
    if variante:
        comptes = [t.subsystem.map(commandes.famille_variante).value_counts() for t in tables]
        return sorted(set().union(*(c[c >= MIN_FAMILLE_VARIANTE].index for c in comptes)))
    comptes = pd.concat(tables).subsystem.map(commandes.famille).value_counts()
    return sorted(comptes[comptes >= MIN_FAMILLE].index)


def comptes_familles(dmop, heures, familles, variante=False):
    table = dmop.assign(subsystem=dmop.subsystem.map(commandes.famille_variante if variante else commandes.famille))
    return pd.DataFrame(commandes.impulsions(table, heures, list(familles)), index=heures,
                        columns=[f"dmopfam_{f}_count" for f in familles])


def construire(x, version="v1", sans=(), dmop=None, voc_commandes=None, physique=False, familles=None,
               variante=False):
    """Retourner (tableau float64 (heures, variables), noms des variables).

    Avec un vocabulaire de commandes (appris sur l'entraînement), ajoute les variables de
    commandes calculées sur la table dmop brute couvrant les mêmes heures. physique et
    familles (vocabulaire appris sur l'entraînement) ajoutent les variables des options ; variante :
    familles commandes.famille_variante.
    """
    if version == "v1":
        table = x.copy()
    elif version == "v2r":
        table = ajouter_v2r(x)
    else:
        raise ValueError(f"Version de variables inconnue : {version}")
    if voc_commandes:
        table = pd.concat([table, commandes.construire(dmop, x.index.to_numpy(), voc_commandes)
                           .set_axis(table.index)], axis=1)
    if physique:
        table = pd.concat([table, physiques(x)], axis=1)
    if familles:
        table = pd.concat([table, comptes_familles(dmop, x.index.to_numpy(), familles, variante)
                           .set_axis(table.index)], axis=1)
    garder = [c for c in table.columns if not set(groupe_de(c)) & set(sans)]
    table = table[garder]
    valeurs = table.to_numpy(dtype=np.float64)
    if not np.isfinite(valeurs).all():
        raise ValueError("Variables non finies")
    return valeurs, list(table.columns)
