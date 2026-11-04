"""Variables tirées des commandes dmop des sous-systèmes AOOO et ATTT (et ATMB).

prepare.py ne compte que les 10 commandes les plus fréquentes, toutes AAAA, APSF ou AVVV :
les commandes AOOO et ATTT n'arrivent au modèle que fondues dans dmop_other_count.
Ici, pour chaque commande suivie (vocabulaire appris sur l'entraînement) :
- impulsion : nombre de commandes dans l'heure ;
- état : 1 tant que cette commande est la dernière reçue par sa famille (même code
  sans la dernière lettre variante, ex. ATTTF321A / ATTTF321D -> ATTTF321), sinon 0 ;
- décalage : chaque variable est lue à t + délai, le délai (entre -MAX_DELAI et
  +MAX_DELAI heures) étant celui qui corrèle le mieux avec un circuit cible
  (NPWD2881 par défaut) sur l'année d'entraînement seulement.
Option regroupement : une commande absente de l'entraînement est rattachée à la commande
connue du même sous-système dont les voisines (commandes reçues à moins de VOISINAGE_MS)
lui ressemblent le plus. Ce rattachement ne lit que la table dmop du dossier de test.
Option atmb : ATMBdddK lu comme une consigne numérique (ddd), maintenue jusqu'à la suivante.
Ces deux options (atmb, masque) vont toujours au GRU, même avec une branche de commandes.
Option masque : 1 pendant les K heures qui suivent l'une des commandes listées (ex. ATTT305C
et voisines, qui coupent NPWD2551 quelques heures) ; K (1 à MAX_MASQUE) est la durée qui
corrèle le mieux avec le circuit visé sur l'entraînement.
"""
import re

import numpy as np
import pandas as pd

H = 3_600_000
MIN_OCCURRENCES = 20
MAX_DELAI = 12
VOISINAGE_MS = 30 * 60_000
SIMILARITE_MIN = 0.5
MAX_MASQUE = 8


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


def voisinage(table, codes_connus, codes):
    """Profil de chaque code de codes : fréquence des codes connus reçus à moins de VOISINAGE_MS."""
    table = table.sort_values("ut_ms")
    temps, valeurs = table.ut_ms.to_numpy(dtype=np.int64), table.subsystem.to_numpy()
    position = {c: j for j, c in enumerate(codes_connus)}
    profils = {}
    for code in codes:
        moments = temps[valeurs == code]
        gauche = np.searchsorted(temps, moments - VOISINAGE_MS, side="left")
        droite = np.searchsorted(temps, moments + VOISINAGE_MS, side="right")
        v = np.zeros(len(codes_connus))
        for g, d in zip(gauche, droite):
            for voisin in valeurs[g:d]:
                if voisin != code and voisin in position:
                    v[position[voisin]] += 1
        profils[code] = v / max(len(moments), 1)
    return profils


def regrouper(table, vocabulaire):
    """Renommer les commandes inconnues en la commande connue la plus proche (même sous-système)."""
    connus = vocabulaire["codes"]
    references = {c: np.asarray(p) for c, p in vocabulaire["profils"].items()}
    inconnus = [c for c in pd.unique(table.subsystem)
                if c.startswith(tuple(vocabulaire["prefixes"])) and c not in references]
    if not inconnus:
        return table
    profils = voisinage(table, vocabulaire["voisins"], inconnus)
    correspondance = {}
    for code in inconnus:
        p = profils[code]
        # Variante d'une famille connue (ATTTF321P, AOOOF22R1) : la plus proche de la famille.
        # Sinon : la plus proche du sous-système, si la ressemblance suffit.
        famille_connue = [c for c in connus if famille(c) == famille(code)]
        candidats = famille_connue or [c for c in connus if c[:4] == code[:4]]
        meilleur, score = None, -1.0 if famille_connue else SIMILARITE_MIN
        for c in candidats:
            r = references[c]
            s = p @ r / (np.linalg.norm(p) * np.linalg.norm(r) + 1e-12)
            if s > score:
                meilleur, score = c, s
        if meilleur:
            correspondance[code] = meilleur
    return table.assign(subsystem=table.subsystem.replace(correspondance))


def apprendre(periodes, circuits, options):
    """Vocabulaire et délais appris sur l'entraînement.

    periodes : liste de (table dmop, heures, y), circuits : noms des colonnes de y.
    """
    prefixes = options.get("prefixes", ["AOOO", "ATTT"])
    tables = pd.concat([t for t, _, _ in periodes])
    comptes = tables.subsystem[tables.subsystem.str.startswith(tuple(prefixes))].value_counts()
    codes = sorted(comptes[comptes >= MIN_OCCURRENCES].index)
    cible = circuits.index(options.get("cible", "NPWD2881"))
    vocabulaire = {"prefixes": prefixes, "codes": codes, "types": options.get("types", ["impulsion", "etat"]),
                   "atmb": options.get("atmb", False), "delais": {}}
    for genre in vocabulaire["types"]:
        fonction = impulsions if genre == "impulsion" else etats
        blocs = [(fonction(t, h, codes), y[:, cible]) for t, h, y in periodes]
        delais = []
        for j in range(len(codes)):
            meilleur, score = 0, -1.0
            if options.get("decalage", True):
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
    if options.get("masque"):
        masque = options["masque"]
        colonne = circuits.index(masque["cible"])
        y = np.concatenate([y[:, colonne] for _, _, y in periodes])
        ok, meilleur, score = np.isfinite(y), 1, -1.0
        for k in range(1, MAX_MASQUE + 1):
            x = np.concatenate([fenetre_masque(t, h, masque["codes"], k) for t, h, _ in periodes])
            if x[ok].std() > 0 and abs(np.corrcoef(x[ok], y[ok])[0, 1]) > score:
                meilleur, score = k, abs(np.corrcoef(x[ok], y[ok])[0, 1])
        vocabulaire["masque"] = {"codes": masque["codes"], "heures": meilleur}
    if options.get("regroupement"):
        voisins = sorted(tables.subsystem.value_counts().index[:300])
        vocabulaire["voisins"] = voisins
        profils = voisinage(tables, voisins, codes)
        vocabulaire["profils"] = {c: profils[c].tolist() for c in codes}
    return vocabulaire


def construire(table, heures, vocabulaire):
    """DataFrame indexé par heure des variables de commandes."""
    heures = np.asarray(heures, dtype=np.int64)
    if vocabulaire.get("profils"):
        table = regrouper(table, vocabulaire)
    codes, sortie = vocabulaire["codes"], {}
    for genre in vocabulaire["types"]:
        valeurs = (impulsions if genre == "impulsion" else etats)(table, heures, codes)
        for j, (code, delai) in enumerate(zip(codes, vocabulaire["delais"][genre])):
            sortie[f"dmopcmd_{genre}_{code}"] = decaler(valeurs[:, j:j + 1], delai)[:, 0]
    if vocabulaire.get("atmb"):
        sortie["dmopatmb_consigne"] = consigne_atmb(table, heures)
    if vocabulaire.get("masque"):
        m = vocabulaire["masque"]
        sortie["dmopmasque_attt"] = fenetre_masque(table, heures, m["codes"], m["heures"])
    return pd.DataFrame(sortie, index=heures)


def consigne_atmb(table, heures):
    """Dernière consigne ATMBdddK reçue avant la fin de l'heure (ddd), 0 avant la première."""
    lignes = table[table.subsystem.str.fullmatch(r"ATMB\d{3}K")]
    temps = lignes.ut_ms.to_numpy(dtype=np.int64)
    valeurs = lignes.subsystem.str[4:7].astype(float).to_numpy()
    if not len(temps):
        return np.zeros(len(heures))
    position = np.searchsorted(temps, np.asarray(heures, dtype=np.int64) + H, side="left") - 1
    return np.where(position >= 0, valeurs[np.maximum(position, 0)], 0.0)


def fenetre_masque(table, heures, codes, k):
    """1 si l'une des commandes a été reçue dans l'heure ou les k - 1 heures précédentes."""
    pulses = impulsions(table, np.asarray(heures, dtype=np.int64), list(codes)).sum(axis=1)
    cumul = np.concatenate([[0.0], np.cumsum(pulses)])
    t = np.arange(len(pulses))
    return (cumul[t + 1] - cumul[np.maximum(t + 1 - k, 0)] > 0).astype(np.float64)
