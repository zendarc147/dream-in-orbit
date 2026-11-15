"""NPWD2551 à la résolution brute (programme 5, docs/JOURNAL_2551.md).

Le courant brut (32 s) est une somme de lignes de chauffage tout-ou-rien : A ≈ 0.205 (deux exemplaires),
B ≈ 0.63, C ≈ 1.045 (outils/diag_fin_2551.py). Ce module en tire deux jeux de cibles, écrits une fois en
local dans data/prepared/p5/ (les power.csv bruts ne vont pas sur Kaggle) :
- lignes.csv : moyenne horaire de chaque ligne (A, B, C), dont la somme est la cible horaire à 0.01 près ;
- dix.csv : moyenne de NPWD2551 par case de 10 min.
Et les variables à la minute de la grille de 10 min (aucune consommation lue).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dio import contexte

C = "NPWD2551"
H = 3_600_000
PAS = 600_000
PAR_HEURE = H // PAS
LIGNES = {"A": 0.205, "B": 0.63, "C": 1.045}
PLAFOND_MIN = 600
RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "data" / "prepared" / "p5"


def decoder(v):
    """États par mesure : C (0/1), B (0/1), A (0, 1 ou 2 exemplaires)."""
    c = (v > 0.93).astype(float)
    r = v - LIGNES["C"] * c
    b = (r > 0.52).astype(float)
    a = np.clip(np.round((r - LIGNES["B"] * b) / LIGNES["A"]), 0, 2)
    return {"A": a, "B": b, "C": c}


def cibles(t, v):
    """(lignes horaires, moyennes à 10 min) d'une série brute (t en ms, v courant)."""
    etats = decoder(v)
    heures = t // H * H
    lignes = pd.DataFrame({k: e * LIGNES[k] for k, e in etats.items()}).groupby(heures).mean()
    dix = pd.Series(v).groupby(t // PAS * PAS).mean().rename(C)
    return lignes.rename_axis("ut_ms"), dix.rename_axis("ut_ms")


def preparer():
    DOSSIER.mkdir(parents=True, exist_ok=True)
    lignes, dix = [], []
    for annee in ("annee1", "annee2"):
        d = pd.read_csv(RACINE / "data" / annee / "power.csv", usecols=["ut_ms", C]).dropna()
        a, b = cibles(d.ut_ms.to_numpy(dtype=np.int64), d[C].to_numpy())
        lignes.append(a)
        dix.append(b)
    pd.concat(lignes).round(5).to_csv(DOSSIER / "lignes.csv")
    pd.concat(dix).round(5).to_csv(DOSSIER / "dix.csv")


def grille(heures):
    """Débuts des cases de 10 min de chaque heure (6 par heure, dans l'ordre)."""
    return (np.asarray(heures, dtype=np.int64)[:, None] + np.arange(PAR_HEURE) * PAS).ravel()


def vers_heures(valeurs):
    """Moyenne des 6 cases de chaque heure (NaN ignorés ; heure sans case connue : NaN)."""
    v = np.asarray(valeurs, dtype=np.float64).reshape(-1, PAR_HEURE)
    n = np.isfinite(v).sum(axis=1)
    return np.where(n > 0, np.nansum(v, axis=1) / np.maximum(n, 1), np.nan)


def _depuis_jusqua(instants, t):
    """Minutes depuis le dernier instant et jusqu'au prochain (plafond PLAFOND_MIN)."""
    instants = np.sort(np.asarray(instants, dtype=np.int64))
    depuis, jusqua = np.full(len(t), float(PLAFOND_MIN)), np.full(len(t), float(PLAFOND_MIN))
    if len(instants):
        i = np.searchsorted(instants, t, side="right") - 1
        depuis[i >= 0] = np.minimum((t[i >= 0] - instants[i[i >= 0]]) / 60_000, PLAFOND_MIN)
        j = i + 1
        jusqua[j < len(instants)] = np.minimum((instants[j[j < len(instants)]] - t[j < len(instants)]) / 60_000,
                                               PLAFOND_MIN)
    return depuis, jusqua


def variables_minute(tables, t10):
    """Au milieu de chaque case de 10 min : minutes depuis / jusqu'aux entrées et sorties d'ombre et de
    pénombre de Mars, au péricentre et à l'apocentre, à la commande dmop la plus proche ; dans l'ombre (0/1)."""
    t = np.asarray(t10, dtype=np.int64) + PAS // 2
    evtf, dmop = tables["evtf"], tables["dmop"]
    c, temps = contexte.codes(evtf), evtf.ut_ms.to_numpy(dtype=np.int64)
    sortie = {}
    for nom in ("MAR_UMBRA_START", "MAR_UMBRA_END", "MAR_PENUMBRA_START", "MAR_PENUMBRA_END", "PERICENTRE_PASSAGE",
                "APOCENTRE_PASSAGE"):
        d, j = _depuis_jusqua(temps[(c == nom).to_numpy()], t)
        sortie[f"min_{nom.lower()}_depuis"], sortie[f"min_{nom.lower()}_jusqua"] = d, j
    sortie["min_dans_ombre"] = (sortie["min_mar_umbra_start_depuis"] < sortie["min_mar_umbra_end_depuis"]).astype(float)
    sortie["min_dmop_depuis"], sortie["min_dmop_jusqua"] = _depuis_jusqua(dmop.ut_ms.to_numpy(), t)
    return pd.DataFrame(sortie, index=t10)
