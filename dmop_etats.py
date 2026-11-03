"""État des sous-systèmes d'après les commandes dmop.

Une commande dmop (ex. ATTTF310A) = un sous-système (4 premières lettres, ATTT)
+ une commande. Une commande change durablement l'état du sous-système : ce qui
compte pour le chauffage est souvent la dernière commande reçue, pas le nombre de
commandes dans l'heure. Pour chaque sous-système suivi, on produit :
- l'indicatrice de la dernière commande reçue (vocabulaire appris sur l'entraînement) ;
- les heures écoulées depuis cette dernière commande et jusqu'à la prochaine (plafonnées).
"""
import numpy as np
import pandas as pd

H = 3_600_000
PLAFOND_H = 72
MIN_OCCURRENCES = 30   # une commande plus rare est regroupée dans « autre »
MAX_COMMANDES = 12     # par sous-système


def sous_systeme(codes):
    return codes.str[:4]


def apprendre_vocabulaire(tables):
    """Choisir les sous-systèmes « A*** » et leurs commandes fréquentes sur les tables d'entraînement."""
    codes = pd.concat([t.subsystem for t in tables], ignore_index=True)
    codes = codes[codes.str.startswith("A")]
    vocabulaire = {}
    for ss, groupe in codes.groupby(sous_systeme(codes)):
        comptes = groupe.value_counts()
        frequentes = comptes[comptes >= MIN_OCCURRENCES]
        if len(groupe) >= 200 and len(frequentes):
            vocabulaire[ss] = sorted(frequentes.index[:MAX_COMMANDES].tolist())
    return vocabulaire


def construire(table, heures, vocabulaire):
    """Retourner un DataFrame indexé par heure : état des sous-systèmes à la fin de chaque heure."""
    heures = np.asarray(heures, dtype=np.int64)
    fins = heures + H
    sortie = {}
    for ss, commandes in vocabulaire.items():
        lignes = table[sous_systeme(table.subsystem) == ss]
        temps = lignes.ut_ms.to_numpy(dtype=np.int64)
        codes = lignes.subsystem.to_numpy()
        if not len(temps):
            # Sous-système jamais commandé sur cette période : état inconnu, aucun événement.
            temps, codes = np.array([np.iinfo(np.int64).max]), np.array([""])
        # Dernière commande strictement avant la fin de l'heure.
        position = np.searchsorted(temps, fins, side="left") - 1
        connue = position >= 0
        derniere = np.where(connue, codes[np.maximum(position, 0)], "")
        for code in commandes:
            sortie[f"dmopetat_{ss}_{code}"] = (derniere == code).astype(np.float64)
        sortie[f"dmopetat_{ss}_autre"] = (connue & ~np.isin(derniere, commandes)).astype(np.float64)
        depuis = np.where(connue, (fins - temps[np.maximum(position, 0)]) / H, PLAFOND_H)
        suivante = np.searchsorted(temps, fins, side="left")
        existe = suivante < len(temps)
        jusqua = np.where(existe, (temps[np.minimum(suivante, len(temps) - 1)] - fins) / H, PLAFOND_H)
        sortie[f"dmopetat_{ss}_depuis_h"] = np.minimum(depuis, PLAFOND_H)
        sortie[f"dmopetat_{ss}_jusqua_h"] = np.minimum(jusqua, PLAFOND_H)
    return pd.DataFrame(sortie, index=heures)
