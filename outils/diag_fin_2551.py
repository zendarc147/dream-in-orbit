"""Diagnostics de NPWD2551 à la résolution brute (32 s), programme 5 (docs/JOURNAL_2551.md).

    python outils/diag_fin_2551.py            # écrit runs/p5_diag/diag.json

Le courant brut prend des niveaux discrets : somme de lignes de chauffage tout-ou-rien (A ≈ 0.205, deux
exemplaires ; B ≈ 0.63 ; C ≈ 1.045). Chaque mesure est décodée en états de ligne, puis pour chaque ligne :
- D6 : part du temps allumée, durées des créneaux allumés / éteints, part de la moyenne et de la variance
  horaires de 2551 ;
- D7 : part de la variance intra-journalière des moyennes horaires expliquée par la phase orbitale (temps
  depuis le péricentre exact), repliée par cases de 10 min ou de 60 min (oracle : cases apprises sur la
  même année, jours hauts seulement) ;
- D9 : variance horaire = niveau journalier + intra-jour, autocorrélation intra-jour à 1, 2, 7, 24 h.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import contexte, donnees  # noqa: E402

C = "NPWD2551"
H = 3_600_000
LIGNES = {"A": 0.205, "B": 0.63, "C": 1.045}
PAS = 600_000  # 10 min


def lire(annee):
    cache = RACINE / "runs" / "p5_diag" / f"{annee}_2551.npy"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, pd.read_csv(RACINE / "data" / annee / "power.csv", usecols=["ut_ms", C]).to_numpy())
    d = np.load(cache)
    return d[:, 0].astype(np.int64), d[:, 1]


def decoder(v):
    """États par mesure : C (0/1), B (0/1), A (0, 1 ou 2 exemplaires), et reste non expliqué."""
    c = (v > 0.93).astype(float)
    r = v - LIGNES["C"] * c
    b = (r > 0.52).astype(float)
    r = r - LIGNES["B"] * b
    a = np.clip(np.round(r / LIGNES["A"]), 0, 2)
    return {"A": a, "B": b, "C": c}, r - LIGNES["A"] * a


def moyennes(t, x, pas):
    """Moyenne de x par case de pas ms (index = début de case), cases vides absentes."""
    return pd.Series(x).groupby(t // pas * pas).mean()


def creneaux(t, etat):
    """Durées (min) des séquences allumées et éteintes, hors trous de mesure de plus de 5 min."""
    change = np.flatnonzero((np.diff(etat) != 0) | (np.diff(t) > 300_000)) + 1
    bornes = np.r_[0, change, len(t)]
    duree = (t[bornes[1:] - 1] - t[bornes[:-1]]) / 60_000 + 32 / 60
    val = etat[bornes[:-1]]
    q = lambda d: [round(float(x), 1) for x in np.quantile(d, [0.1, 0.5, 0.9])] if len(d) else None  # noqa: E731
    return {"allume_min_q10_50_90": q(duree[val > 0]), "eteint_min_q10_50_90": q(duree[val == 0]),
            "creneaux": int((val > 0).sum())}


def autocorr(x, k):
    a, b = x[:-k], x[k:]
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(np.corrcoef(a[m], b[m])[0, 1]), 3) if m.sum() > 100 else None


def replie(t10, x10, peri, cases_min, masque_jour):
    """Part de la variance intra-jour des moyennes horaires expliquée par le repli sur le temps depuis le
    péricentre, par cases de cases_min minutes (moyenne par case apprise sur les mêmes données)."""
    k = np.clip(np.searchsorted(peri, t10 + PAS // 2, side="right") - 1, 0, len(peri) - 1)
    depuis = (t10 + PAS // 2 - peri[k]) / 60_000
    case = (depuis // cases_min).astype(int)
    d = pd.DataFrame({"x": x10, "case": case, "h": t10 // H, "j": t10 // (24 * H)})
    d = d[d.j.isin(masque_jour)]
    d["p"] = d.groupby("case").x.transform("mean")
    h = d.groupby("h").agg(x=("x", "mean"), p=("p", "mean"), j=("j", "first"))
    xi, pi = h.x - h.groupby("j").x.transform("mean"), h.p - h.groupby("j").p.transform("mean")
    return round(float(1 - np.mean((xi - pi) ** 2) / np.mean(xi ** 2)), 4), int(len(h))


def annee(nom):
    t, v = lire(nom)
    etats, reste = decoder(v)
    evtf = donnees.lire_contextes(nom)["evtf"]
    peri = np.sort(evtf.ut_ms[contexte.codes(evtf) == "PERICENTRE_PASSAGE"].to_numpy(dtype=np.int64))
    y = moyennes(t, v, H)
    jour = y.index // (24 * H)
    niveau = y.groupby(jour).transform("mean")
    hauts = set(niveau.index[niveau > 0.15] // (24 * H))
    sortie = {"mesures": len(v), "reste_rmse": round(float(np.sqrt(np.mean(reste ** 2))), 4),
              "moyenne": round(float(y.mean()), 4), "variance_horaire": round(float(y.var()), 5),
              "part_variance_niveau_jour": round(float(niveau.var() / y.var()), 3),
              "autocorr_intra_jour": {k: autocorr((y - niveau).reindex(range(y.index[0], y.index[-1] + H, H)).to_numpy(), k)
                                      for k in (1, 2, 7, 24)}, "lignes": {}}
    series = {**{k: e * LIGNES[k] for k, e in etats.items()}, "total": v}
    for k, x in series.items():
        m = moyennes(t, x, H)
        nj = m.groupby(m.index // (24 * H)).transform("mean")
        x10 = moyennes(t, x, PAS)
        r = {"part_moyenne": round(float(m.mean() / y.mean()), 3),
             "part_variance_horaire": round(float(np.cov(m, y)[0, 1] / y.var()), 3),
             "part_variance_intra_jour": round(float(np.cov(m - nj, y - niveau)[0, 1] / (y - niveau).var()), 3),
             "autocorr_intra_jour_1h": autocorr((m - nj).reindex(range(m.index[0], m.index[-1] + H, H)).to_numpy(), 1)}
        for cases in (10, 60):
            r[f"replie_{cases}min_jours_hauts"] = replie(x10.index.to_numpy(), x10.to_numpy(), peri, cases, hauts)[0]
        if k != "total":
            r.update(creneaux(t, etats[k]))
            r["part_temps_allume"] = round(float((etats[k] > 0).mean()), 3)
        sortie["lignes"][k] = r
    return sortie


def main():
    sortie = {a: annee(a) for a in ("annee1", "annee2")}
    (RACINE / "runs" / "p5_diag").mkdir(parents=True, exist_ok=True)
    (RACINE / "runs" / "p5_diag" / "diag.json").write_text(json.dumps(sortie, indent=1) + "\n")
    print(json.dumps(sortie, indent=1))


if __name__ == "__main__":
    main()
