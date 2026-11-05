"""Figures du rapport d'avancement (rapport/figures/*.pdf) et chiffres utilisés dans le texte.

    python rapport/figures.py --preds DOSSIER

DOSSIER contient boost_{directe,inverse}.csv, final_{directe,inverse}.csv et
final_{directe,inverse}_ecart.csv, produits par baselines/boosting.py et mixte.py
(poids runs/c2_niveau_600/) sur validation/ et validation_inverse/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "rapport" / "figures"
BLEU, ORANGE, VERT, GRIS, ENCRE, ENCRE2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
DIRECTIONS = {"directe": ("annee1", "validation"), "inverse": ("annee2", "validation_inverse")}
MOIS_H = 24 * 30

plt.rcParams.update({
    "font.family": ["C059", "DejaVu Serif"], "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": GRIS, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "xtick.color": ENCRE2,
    "ytick.color": ENCRE2, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.4, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def lire(chemin):
    return pd.read_csv(chemin, index_col="ut_ms")


def rmse(p, v):
    e = (p - v).to_numpy()
    return float(np.sqrt(np.nanmean(e ** 2)))


def constante(annee, verite):
    moyenne = pd.read_csv(RACINE / "data/prepared" / annee / "y.csv", index_col=0).mean()
    return pd.DataFrame(np.tile(moyenne[verite.columns].to_numpy(), (len(verite), 1)),
                        index=verite.index, columns=verite.columns)


def rmse_par_mois(p, v):
    e2 = ((p - v) ** 2).to_numpy()
    blocs = np.arange(len(e2)) // MOIS_H
    return np.array([np.sqrt(np.nanmean(e2[blocs == b])) for b in np.unique(blocs)])


def fig_progression(chiffres):
    etapes = [
        ("Constante (moyenne par circuit)", 0.1058, 0.1252),
        ("Boosting brut", 0.0981, 0.1095),
        ("Boosting régularisé + recalage", 0.0951, 0.0999),
        ("GRU, 1 graine (E2)", 0.0901, 0.0989),
        ("GRU, ensemble 5 graines (E3)", 0.0876, 0.0975),
        ("+ bruit sur les entrées (it. 3)", 0.0872, 0.0971),
        ("+ second membre v2r (it. 5)", 0.0863, 0.0964),
        ("+ EMA, 600 pas, correction de niveau", 0.0853, 0.0940),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y = np.arange(len(etapes))[::-1]
    for yi, (_, d, i) in zip(y, etapes):
        ax.plot([d, i], [yi, yi], color="#d6d5d0", lw=1.2, zorder=1)
    ax.scatter([e[1] for e in etapes], y, s=34, color=BLEU, zorder=3, label="directe (annee1 → annee2)",
               edgecolor="white", linewidth=1)
    ax.scatter([e[2] for e in etapes], y, s=34, color=ORANGE, zorder=3, marker="s",
               label="inverse (annee2 → annee1)", edgecolor="white", linewidth=1)
    for yi, (_, d, i) in zip(y, etapes):
        # Moyennes de RESULTATS.md, calculées sur les RMSE non arrondies (it. 3 : 0.0921).
        moyenne = 0.0921 if (d, i) == (0.0872, 0.0971) else (d + i) / 2
        ax.text(max(d, i) + 0.0012, yi, f"{moyenne:.4f}", va="center", fontsize=7.5, color=ENCRE2)
    ax.set_yticks(y, [e[0] for e in etapes])
    ax.axhline(4.5, color=GRIS, lw=0.6, ls=(0, (3, 3)))
    ax.text(0.1305, 4.62, "modèles classiques ↑", fontsize=7, color=ENCRE2, va="bottom", ha="right")
    ax.text(0.1305, 4.38, "world models (réseaux de neurones) ↓", fontsize=7, color=ENCRE2, va="top", ha="right")
    ax.set_xlim(0.083, 0.131)
    ax.set_xlabel("RMSE (plus bas = mieux) ; chiffre à droite : moyenne des deux directions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(SORTIE / "progression.pdf")
    plt.close(fig)


def fig_derive(donnees, chiffres):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    for ax, (direction, d) in zip(axes, donnees.items()):
        v = d["verite"]
        for nom, cle, couleur, style in [("constante", "constante", GRIS, (0, (3, 2))),
                                         ("boosting + recalage", "boost", ORANGE, "-"),
                                         ("world model final", "final", BLEU, "-")]:
            courbe = rmse_par_mois(d[cle], v)
            ax.plot(np.arange(1, len(courbe) + 1), courbe, color=couleur, ls=style, label=nom,
                    marker="o", ms=2.5)
            chiffres[f"derive_{direction}_{cle}"] = [round(float(c), 4) for c in courbe]
        ax.set_title(f"Validation {direction}", loc="left")
        ax.set_xlabel("mois de simulation (blocs de 30 jours)")
    axes[0].set_ylabel("RMSE du mois")
    axes[0].legend(loc="upper left")
    fig.savefig(SORTIE / "derive.pdf")
    plt.close(fig)


def fig_reve(d):
    v, p, e = d["verite"], d["final"], d["ecart"]
    dates = pd.to_datetime(v.index, unit="ms")

    def jour(s):
        return s.rolling(24, min_periods=1, center=True).mean()

    series = [("Consommation totale des 33 circuits", v.sum(axis=1, min_count=1), p.sum(axis=1), e.sum(axis=1)),
              ("NPWD2551 (circuit le plus difficile)", v["NPWD2551"], p["NPWD2551"], e["NPWD2551"]),
              ("NPWD2532 (circuit le plus fort en moyenne)", v["NPWD2532"], p["NPWD2532"], e["NPWD2532"])]
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 5.2), sharex=True)
    fig.subplots_adjust(hspace=0.38)
    for ax, (titre, sv, sp, se) in zip(axes, series):
        ax.fill_between(dates, jour(sp - 2 * se).clip(lower=0), jour(sp + 2 * se), color=BLEU, alpha=0.18, lw=0,
                        label="± 2 écarts-types de l'ensemble")
        ax.plot(dates, jour(sv), color=ENCRE, lw=0.7, label="réalité")
        ax.plot(dates, jour(sp), color=BLEU, lw=0.9, label="simulation")
        ax.set_title(f"{titre}, RMSE horaire {rmse(sp, sv):.3f}", loc="left")
        ax.set_ylabel("courant")
    axes[0].legend(loc="upper right", ncol=3)
    fig.savefig(SORTIE / "reve.pdf")
    plt.close(fig)


def fig_circuits(d, chiffres):
    v, p, c = d["verite"], d["final"], d["constante"]
    e2 = ((p - v) ** 2).sum()
    part = e2 / e2.sum()
    haut = part.sort_values(ascending=False).index[:10]
    r_final = np.sqrt(((p - v) ** 2).mean())
    r_const = np.sqrt(((c - v) ** 2).mean())
    r_ideal = v.std()  # constante égale à la vraie moyenne de la période
    chiffres["part_erreur_directe"] = {k: round(float(part[k]), 3) for k in haut}
    chiffres["rmse_circuit_directe"] = {k: [round(float(r_final[k]), 3), round(float(r_const[k]), 3),
                                            round(float(r_ideal[k]), 3)] for k in haut}
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    y = np.arange(len(haut))[::-1]
    h = 0.26
    ax.barh(y + h, r_const[haut], h, color="#c9c8c2", label="constante apprise sur annee1")
    ax.barh(y, r_final[haut], h, color=BLEU, label="world model final")
    ax.barh(y - h, r_ideal[haut], h, color=VERT, label="constante à la vraie moyenne d'annee2 (inaccessible)")
    ax.set_yticks(y, [f"{k}  ({100 * part[k]:.0f} %)" for k in haut])
    ax.set_xlabel("RMSE du circuit sur la validation directe ; entre parenthèses : part de l'erreur totale")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(SORTIE / "circuits.pdf")
    plt.close(fig)


def fig_niveau(chiffres):
    """NPWD2531 en validation inverse : sans et avec la correction de niveau."""
    v = lire(RACINE / "validation_inverse/verite_horaire.csv")["NPWD2531"]
    init = pd.read_csv(RACINE / "validation_inverse/test/power_init.csv", index_col="ut_ms")["NPWD2531"]
    sans = lire(RACINE / "runs/ref_mixte/inverse/pred.csv")["NPWD2531"].reindex(v.index)
    avec = lire(RACINE / "runs/c2_niveau_600/inverse/pred.csv")["NPWD2531"].reindex(v.index)
    dates = pd.to_datetime(v.index, unit="ms")
    debut = dates[0]
    jours = (dates - debut).days + (dates - debut).seconds / 86400
    fin = jours <= 240
    fig, ax = plt.subplots(figsize=(6.3, 2.2))

    def jour(s):
        return s.rolling(24, min_periods=1, center=True).mean()

    ax.plot(jours[fin], jour(v)[fin], color=ENCRE, lw=0.7, label="réalité")
    ax.plot(jours[fin], jour(sans)[fin], color=ORANGE, lw=1.0, label="sans correction (ensemble mixte, it. 5)")
    ax.plot(jours[fin], jour(avec)[fin], color=BLEU, lw=1.0, label="avec correction de niveau (final)")
    ax.axhline(float(init.mean()), color=GRIS, lw=0.6, ls=(0, (3, 2)))
    ax.text(236, float(init.mean()) + 0.01, f"moyenne sur power_init : {init.mean():.2f}", ha="right",
            fontsize=7, color=ENCRE2)
    ax.set_xlabel("jours depuis la fin de power_init")
    ax.set_ylabel("courant NPWD2531")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.52), handlelength=2.5)
    for nom, s in [("sans", sans), ("avec", avec)]:
        chiffres[f"niveau_2531_{nom}_90j"] = round(rmse(s[jours <= 90], v[jours <= 90]), 3)
        chiffres[f"niveau_2531_{nom}_annee"] = round(rmse(s, v), 3)
    fig.savefig(SORTIE / "niveau.pdf")
    plt.close(fig)


def fig_ablations():
    # Écarts moyens recopiés d'ABLATIONS.md (calculés sur les RMSE non arrondies).
    variantes = [("sans mémoire\n(ni encodeur ni GRU)", 0.0891, 0.1055, 0.0071),
                 ("sans saaf\n(angles du Soleil)", 0.0870, 0.0989, 0.0027),
                 ("sans dmop\n(commandes)", 0.0885, 0.0958, 0.0019), ("sans éclipses", 0.0879, 0.0957, 0.0016)]
    complet = (0.0864, 0.0941)
    fig, ax = plt.subplots(figsize=(6.3, 2.1))
    y = np.arange(len(variantes))[::-1]
    h = 0.34
    ax.barh(y + h / 2, [v[1] - complet[0] for v in variantes], h, color=BLEU, label="directe")
    ax.barh(y - h / 2, [v[2] - complet[1] for v in variantes], h, color=ORANGE, label="inverse")
    for yi, v in zip(y, variantes):
        ax.text(max(v[1] - complet[0], v[2] - complet[1]) + 0.0002, yi, f"moyenne +{v[3]:.4f}",
                va="center", fontsize=7.5, color=ENCRE2)
    ax.set_yticks(y, [v[0] for v in variantes])
    ax.set_xlim(0, 0.0145)
    ax.set_xlabel("hausse de la RMSE par rapport au modèle complet (directe 0.0864, inverse 0.0941)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(SORTIE / "ablations.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, required=True)
    args = ap.parse_args()
    SORTIE.mkdir(parents=True, exist_ok=True)
    chiffres, donnees = {}, {}
    for direction, (annee, dossier) in DIRECTIONS.items():
        v = lire(RACINE / dossier / "verite_horaire.csv")
        d = {"verite": v, "constante": constante(annee, v)}
        for cle in ("boost", "final"):
            d[cle] = lire(args.preds / f"{cle}_{direction}.csv").reindex(v.index)[v.columns]
        d["ecart"] = lire(args.preds / f"final_{direction}_ecart.csv").reindex(v.index)[v.columns]
        donnees[direction] = d
        chiffres[f"rmse_{direction}"] = {k: round(rmse(d[k], v), 4) for k in ("constante", "boost", "final")}
        # Couverture : part des valeurs réelles dans la bande ± 2 écarts-types.
        dedans = ((v - d["final"]).abs() <= 2 * d["ecart"]).to_numpy()[v.notna().to_numpy()]
        chiffres[f"couverture_2sigma_{direction}"] = round(float(dedans.mean()), 3)
    fig_progression(chiffres)
    fig_derive(donnees, chiffres)
    fig_reve(donnees["directe"])
    fig_circuits(donnees["directe"], chiffres)
    fig_niveau(chiffres)
    fig_ablations()
    (SORTIE / "chiffres.json").write_text(json.dumps(chiffres, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(chiffres, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
