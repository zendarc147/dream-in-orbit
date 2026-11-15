"""Thermostat à faible capacité pour les lignes de chauffage de NPWD2551 (programme 5, docs/JOURNAL_2551.md).

Par ligne : température latente T = w · Σ_j π_j F_j(u), où F_j(u) sont les filtres du premier ordre (constantes
de temps gbm_sacs.CONSTANTES_THERMIQUES) de l'ombre et du flux éclairé par face, π un mélange appris des
constantes de temps (softmax) et w un poids par série ; taux d'allumage = sigmoïde(b − T), courant = amplitude ×
taux. Une cinquantaine de paramètres pour les trois lignes : pensé pour se transférer d'une année à l'autre.
Aucune consommation en entrée. Ne décrit pas les bascules commandées du sol.
"""
import re

import numpy as np
import torch

PAS, LR = 1500, 0.03


def decouper(colonnes):
    """Indices (série, constante de temps) des colonnes th_<série>_tau<τ> : tableau [séries, constantes]."""
    cles = [re.fullmatch(r"th_(.+)_tau(\d+)", c).groups() for c in colonnes]
    series, taus = sorted({s for s, _ in cles}), sorted({int(t) for _, t in cles})
    return np.array([[cles.index((s, str(t))) for t in taus] for s in series]), series, taus


def ajuster(f, y, colonnes, graine=0):
    """f : colonnes th_* (heures × colonnes) ; y : (heures × lignes) moyennes horaires, NaN ignorés."""
    idx, series, taus = decouper(colonnes)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    moyenne, ecart = np.nanmean(f, axis=0), np.nanstd(f, axis=0) + 1e-9
    x = torch.as_tensor(((f - moyenne) / ecart)[:, idx], dtype=torch.float32, device=appareil)  # heures, séries, τ
    connus = torch.as_tensor(np.isfinite(y), device=appareil)
    cible = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32, device=appareil)
    torch.manual_seed(graine)
    k = y.shape[1]
    p = {"w": torch.zeros(k, len(series)), "pi": torch.zeros(k, len(taus)), "b": torch.zeros(k),
         "amp": torch.as_tensor(np.nanmax(y, axis=0), dtype=torch.float32).clamp(min=1e-3).log()}
    p = {n: v.to(appareil).requires_grad_() for n, v in p.items()}
    optim = torch.optim.Adam(p.values(), lr=LR)
    for _ in range(PAS):
        optim.zero_grad()
        perte = ((_sortie(x, p) - cible)[connus] ** 2).mean()
        perte.backward()
        optim.step()
    return {"params": {n: v.detach().cpu().numpy() for n, v in p.items()}, "moyenne": moyenne, "ecart": ecart,
            "idx": idx, "series": series, "taus": taus, "perte": float(perte.detach())}


def _sortie(x, p):
    lisse = torch.einsum("hst,kt->hks", x, torch.softmax(p["pi"], dim=1))  # heures, lignes, séries
    return p["amp"].exp() * torch.sigmoid(p["b"] - (lisse * p["w"]).sum(-1))


def predire(modele, f):
    x = torch.as_tensor(((f - modele["moyenne"]) / modele["ecart"])[:, modele["idx"]], dtype=torch.float32)
    with torch.no_grad():
        return _sortie(x, {n: torch.as_tensor(v) for n, v in modele["params"].items()}).numpy()
