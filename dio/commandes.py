"""Variables tirées des commandes dmop des sous-systèmes AOOO et ATTT.

prepare.py ne compte que les 10 commandes les plus fréquentes, toutes AAAA, APSF ou AVVV :
les commandes AOOO et ATTT n'arrivent au modèle que fondues dans dmop_other_count.
Ici, pour chaque commande suivie (vocabulaire appris sur l'entraînement) :
- impulsion : nombre de commandes dans l'heure ;
- état : 1 tant que cette commande est la dernière reçue par sa famille (même code
  sans la dernière lettre variante, ex. ATTTF321A / ATTTF321D -> ATTTF321), sinon 0 ;
- décalage : chaque variable est lue à t + délai, le délai (entre -MAX_DELAI et
  +MAX_DELAI heures) étant celui qui corrèle le mieux avec un circuit cible
  (NPWD2881) sur l'année d'entraînement seulement.
"""
import re

import numpy as np
import pandas as pd

H = 3_600_000
MIN_OCCURRENCES = 20
MAX_DELAI = 12
CIBLE = "NPWD2881"
TYPES = ["impulsion", "etat"]


def famille(code):
    """Code sans sa variante finale : ATTTF321A -> ATTTF321, AOOOF22R1 -> AOOOF22, ATTT305C -> ATTT305."""
    m = re.match(r"^(A[A-Z]{3}F?\d+)", code)
    return m.group(1) if m else code


def _heure(temps, debut, n):
    """Indice d'heure de chaque instant sur une grille horaire commençant à debut."""
    i = (np.asarray(temps, dtype=np.int64) - debut) // H
    return i, (i >= 0) & (i < n)


def impulsions(table, heures, codes):
    """(heures, codes) : nombre de commandes de chaque code dans chaque heure."""
    n, sortie = len(heures), np.zeros((len(heures), len(codes)))
    if not codes:
        return sortie
    position = {c: j for j, c in enumerate(codes)}
    lignes = table[table.subsystem.isin(position)]
    i, ok = _heure(lignes.ut_ms, heures[0], n)
    np.add.at(sortie, (i[ok], lignes.subsystem.map(position).to_numpy()[ok]), 1.0)
    return sortie


def etats(table, heures, codes):
    """(heures, codes) : 1 si le code est la dernière commande de sa famille reçue avant la fin de l'heure."""
    fins = np.asarray(heures, dtype=np.int64) + H
    sortie = np.zeros((len(heures), len(codes)))
    familles = pd.Series(codes).map(famille)
    for fam, indices in familles.groupby(familles).groups.items():
        lignes = table[table.subsystem.map(famille) == fam]
        temps, valeurs = lignes.ut_ms.to_numpy(dtype=np.int64), lignes.subsystem.to_numpy()
        if not len(temps):
            continue
        position = np.searchsorted(temps, fins, side="left") - 1
        derniere = np.where(position >= 0, valeurs[np.maximum(position, 0)], "")
        for j in indices:
            sortie[:, j] = derniere == codes[j]
    return sortie


def decaler(valeurs, delai):
    """Lire chaque colonne à t + délai (bords répétés)."""
    n = len(valeurs)
    indices = np.clip(np.arange(n) + delai, 0, n - 1)
    return valeurs[indices]


def apprendre(periodes, circuits, options):
    """Vocabulaire et délais appris sur l'entraînement.

    periodes : liste de (table dmop, heures, y), circuits : noms des colonnes de y.
    """
    prefixes = options.get("prefixes", ["AOOO", "ATTT"])
    tables = pd.concat([t for t, _, _ in periodes])
    comptes = tables.subsystem[tables.subsystem.str.startswith(tuple(prefixes))].value_counts()
    codes = sorted(comptes[comptes >= MIN_OCCURRENCES].index)
    cible = circuits.index(CIBLE)
    vocabulaire = {"prefixes": prefixes, "codes": codes, "types": TYPES, "delais": {}}
    for genre in vocabulaire["types"]:
        fonction = impulsions if genre == "impulsion" else etats
        blocs = [(fonction(t, h, codes), y[:, cible]) for t, h, y in periodes]
        delais = []
        for j in range(len(codes)):
            meilleur, score = 0, -1.0
            for d in range(-MAX_DELAI, MAX_DELAI + 1):
                x = np.concatenate([decaler(v, d)[:, j] for v, _ in blocs])
                y = np.concatenate([c for _, c in blocs])
                ok = np.isfinite(y)
                if x[ok].std() > 0:
                    s = abs(np.corrcoef(x[ok], y[ok])[0, 1])
                    if s > score:
                        meilleur, score = d, s
            delais.append(meilleur)
        vocabulaire["delais"][genre] = delais
    return vocabulaire


def construire(table, heures, vocabulaire):
    """DataFrame indexé par heure des variables de commandes."""
    heures = np.asarray(heures, dtype=np.int64)
    codes, sortie = vocabulaire["codes"], {}
    for genre in vocabulaire["types"]:
        valeurs = (impulsions if genre == "impulsion" else etats)(table, heures, codes)
        for j, (code, delai) in enumerate(zip(codes, vocabulaire["delais"][genre])):
            sortie[f"dmopcmd_{genre}_{code}"] = decaler(valeurs[:, j:j + 1], delai)[:, 0]
    return pd.DataFrame(sortie, index=heures)
