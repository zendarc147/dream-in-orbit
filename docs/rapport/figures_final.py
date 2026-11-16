"""Figures du rapport final (docs/rapport/figures_final/*.pdf) et chiffres du texte (chiffres.json).

    python docs/rapport/figures_final.py [--ecart DOSSIER]

Modèle final S2 évalué sur les deux validations : runs/gb_S2_{directe,inverse}/<direction>/pred.csv
(modèles entraînés sur une seule période), avec ses parts dans pred.csv.parts/ (gbm, gru, gru_recette,
gru_recette2, melange). Les scores des coupures glissantes viennent de docs/journaux/JOURNAL_GRU_BOUCLE.md.
DOSSIER (facultatif) contient ecart_{directe,inverse}.csv, écrits par
predict.py --poids runs/gb_S2_<d>/<d> --options '{"sortie_incertitude": ...}' : couverture de la bande.
"""
import argparse
import json
from pathlib import Path

from figures import (BLEU, ORANGE, VERT, GRIS, ENCRE, ENCRE2, RACINE, DIRECTIONS,  # règle aussi le style
                     lire, rmse, constante, rmse_par_mois)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SORTIE = Path(__file__).resolve().parent / "figures_final"
VIOLET = "#8a5cc2"


def charger():
    donnees = {}
    for direction, (annee, dossier) in DIRECTIONS.items():
        v = lire(RACINE / dossier / "verite_horaire.csv")
        base = RACINE / "runs" / f"gb_S2_{direction}" / direction / "pred.csv"
        d = {"verite": v, "constante": constante(annee, v),
             "final": lire(base).reindex(v.index)[v.columns]}
        for part in ("gbm", "gru", "gru_recette", "gru_recette2", "melange"):
            d[part] = lire(base.parent / "pred.csv.parts" / f"{part}.csv").reindex(v.index)[v.columns]
        donnees[direction] = d
    return donnees


def fig_progression():
    # README.md (tableau des résultats) : RMSE directe / inverse du modèle installé à chaque étape.
    etapes = [
        ("24/10", "Constante (moyenne par circuit)", 0.1058, 0.1252, 0.1155),
        ("03/11", "Boosting régularisé + recalage", 0.0951, 0.0999, 0.0975),
        ("03/11", "GRU, 1 graine, entrées filtrées", 0.0901, 0.0989, 0.0945),
        ("03/11", "Ensemble de 5 GRU, bruit d'entrée", 0.0872, 0.0971, 0.0921),
        ("04/11", "+ EMA, correction de niveau", 0.0853, 0.0940, 0.0897),
        ("05/11", "+ branche AOOO/ATTT, résidu", 0.0827, 0.0892, 0.0859),
        ("10/11", "+ familles dmop, facteur 1,5", 0.0791, 0.0836, 0.0813),
        ("11/11", "+ boosting à sacs (poids 0,3)", 0.0780, 0.0810, 0.0795),
        ("11/11", "+ spécialistes 2531 et 2532", 0.0767, 0.0791, 0.0779),
        ("12/11", "+ recettes, boosting 0,5, spé. 2551 (L)", 0.0750, 0.0785, 0.0767),
        ("16/11", "C1 (meilleur score interne)", 0.0744, 0.0782, 0.0763),
        ("16/11", "T3 (structure sobre)", 0.0746, 0.0804, 0.0775),
        ("16/11", "S2 = T3 + GRU à boucle (livré)", 0.0746, 0.0805, 0.0775),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(etapes))[::-1]
    for yi, (_, _, d, i, m) in zip(y, etapes):
        ax.plot([d, i], [yi, yi], color="#d6d5d0", lw=1.2, zorder=1)
        ax.text(max(d, i) + 0.0010, yi, f"{m:.4f}", va="center", fontsize=7.2, color=ENCRE2)
    ax.scatter([e[2] for e in etapes], y, s=30, color=BLEU, zorder=3, label="directe (annee1 \u2192 annee2)",
               edgecolor="white", linewidth=1)
    ax.scatter([e[3] for e in etapes], y, s=30, color=ORANGE, zorder=3, marker="s",
               label="inverse (annee2 \u2192 annee1)", edgecolor="white", linewidth=1)
    ax.axhspan(-0.5, 0.5, color="#f3f2ee", zorder=0)
    for sep, texte in [(10.5, "modèles classiques \u2191   world models GRU seuls \u2193"),
                       (6.5, "GRU seuls \u2191   réseaux + boosting \u2193"),
                       (2.5, "structures sobres \u2193")]:
        ax.axhline(sep, color=GRIS, lw=0.6, ls=(0, (3, 3)))
        ax.text(0.1300, sep + 0.08, texte, fontsize=6.5, color=ENCRE2, va="bottom", ha="right")
    ax.set_yticks(y, [f"{e[0]}  {e[1]}" for e in etapes])
    ax.set_xlim(0.072, 0.1305)
    ax.set_xlabel("RMSE (plus bas = mieux) ; chiffre à droite : moyenne des deux directions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.45))
    fig.savefig(SORTIE / "progression.pdf")
    plt.close(fig)


def fig_validations(chiffres):
    # JOURNAL_GRU_BOUCLE.md, tableaux « structures » (modèle S2 et ses parts) et « GRU seuls ».
    noms = ["directe", "inverse", "c50", "c62", "c75", "c87"]
    series = [
        ("GRU S borné seul (G0)", [0.078496, 0.085615, 0.065348, 0.062202, 0.098213, 0.094507], "#9fc3ec"),
        ("GRU à boucle seul (GB)", [0.079570, 0.093860, 0.067564, 0.061160, 0.101570, 0.089988], VIOLET),
        ("part réseau (G0 + GB)/2", [0.078621, 0.088144, 0.065879, 0.061352, 0.099458, 0.089982], BLEU),
        ("part boosting", [0.078902, 0.082585, 0.072776, 0.076242, 0.089708, 0.071504], ORANGE),
        ("S2 complet", [0.074572, 0.080497, 0.064867, 0.063204, 0.092547, 0.074218], ENCRE),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    x = np.arange(len(noms))
    w = 0.16
    for k, (nom, vals, couleur) in enumerate(series):
        ax.bar(x + (k - 2) * w, vals, w, color=couleur, label=nom)
    ax.set_xticks(x, ["directe", "inverse", "coupure 50 %", "62,5 %", "75 %", "87,5 %"])
    ax.set_ylim(0.055, 0.105)
    ax.set_ylabel("RMSE")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    chiffres["validations"] = {n: {s[0]: s[1][i] for s in series} for i, n in enumerate(noms)}
    fig.savefig(SORTIE / "validations.pdf")
    plt.close(fig)


def fig_derive(donnees, chiffres):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    for ax, (direction, d) in zip(axes, donnees.items()):
        v = d["verite"]
        for nom, cle, couleur, style in [("constante", "constante", GRIS, (0, (3, 2))),
                                         ("part boosting", "gbm", ORANGE, "-"),
                                         ("part réseau", "gru", BLEU, "-"),
                                         ("S2 complet", "final", ENCRE, "-")]:
            courbe = rmse_par_mois(d[cle], v)
            ax.plot(np.arange(1, len(courbe) + 1), courbe, color=couleur, ls=style, label=nom,
                    marker="o", ms=2.2, lw=1.1)
            chiffres[f"derive_{direction}_{cle}"] = [round(float(c), 4) for c in courbe]
        ax.set_title(f"Validation {direction}", loc="left")
        ax.set_xlabel("mois de simulation (blocs de 30 jours)")
    axes[0].set_ylabel("RMSE du mois")
    axes[0].legend(loc="upper left", fontsize=6.8, ncol=2)
    fig.savefig(SORTIE / "derive.pdf")
    plt.close(fig)
    for direction, d in donnees.items():
        c = np.array(chiffres[f"derive_{direction}_constante"])
        f = np.array(chiffres[f"derive_{direction}_final"])
        chiffres[f"blocs_battus_{direction}"] = f"{int((f < c).sum())}/{len(f)}"
        n = len(f) // 4
        chiffres[f"derive_{direction}_premier_dernier_quart"] = [round(float(f[:n].mean()), 4),
                                                                 round(float(f[-n:].mean()), 4)]


def fig_circuits(donnees, chiffres):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    v0 = donnees["directe"]
    e2 = ((v0["final"] - v0["verite"]) ** 2).sum()
    haut = (e2 / e2.sum()).sort_values(ascending=False).index[:8]
    y = np.arange(len(haut))[::-1]
    h = 0.38
    for ax, (direction, d) in zip(axes, donnees.items()):
        v = d["verite"]
        e2d = ((d["final"] - v) ** 2).sum()
        part = e2d / e2d.sum()
        r_f = np.sqrt(((d["final"] - v) ** 2).mean())
        r_c = np.sqrt(((d["constante"] - v) ** 2).mean())
        ax.barh(y + h / 2, r_c[haut], h, color="#c9c8c2", label="constante")
        ax.barh(y - h / 2, r_f[haut], h, color=BLEU, label="S2")
        for yi, k in zip(y, haut):
            ax.text(max(r_c[k], r_f[k]) + 0.006, yi, f"{100 * part[k]:.0f} %", va="center", fontsize=6.8,
                    color=ENCRE2)
        ax.set_title(f"Validation {direction}", loc="left")
        ax.set_xlabel("RMSE du circuit")
        ax.grid(axis="y", visible=False)
        chiffres[f"circuits_{direction}"] = {k: {"s2": round(float(r_f[k]), 4), "constante": round(float(r_c[k]), 4),
                                                 "part": round(float(part[k]), 3)} for k in haut}
    axes[0].set_yticks(y, [k.replace("NPWD", "") for k in haut])
    axes[0].legend(loc="lower right", fontsize=7)
    fig.savefig(SORTIE / "circuits.pdf")
    plt.close(fig)


def fig_zoom(donnees, chiffres):
    d = donnees["directe"]
    v = d["verite"]
    jours = np.arange(len(v)) / 24
    f = (jours >= 100) & (jours < 105)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.6), sharex=True)
    fig.subplots_adjust(hspace=0.32)
    for ax, c in zip(axes, ["NPWD2551", "NPWD2451"]):
        ax.plot(jours[f], v[c][f], color=ENCRE, lw=0.8, label="réalité (moyenne horaire)")
        ax.plot(jours[f], d["gru"][c][f], color=VIOLET, lw=0.9, label="part réseau (2 GRU)")
        ax.plot(jours[f], d["gbm"][c][f], color=ORANGE, lw=0.9, label="part boosting")
        ax.plot(jours[f], d["final"][c][f], color=BLEU, lw=1.4, label="S2")
        r = rmse(d["final"][c][f], v[c][f])
        ax.set_title(f"{c}, validation directe, jours 100 à 105 ; RMSE S2 sur la fenêtre {r:.3f}", loc="left")
        ax.set_ylabel("courant")
        chiffres[f"zoom_{c}"] = round(r, 4)
    axes[0].legend(loc="upper left", ncol=4, fontsize=6.8, frameon=True, facecolor="white", edgecolor="none",
                   bbox_to_anchor=(0, 1.02))
    axes[1].set_xlabel("jours depuis la fin de power_init")
    fig.savefig(SORTIE / "zoom.pdf")
    plt.close(fig)


def chiffres_globaux(donnees, chiffres, ecart):
    for direction, d in donnees.items():
        v = d["verite"]
        chiffres[f"rmse_{direction}"] = {k: round(rmse(d[k], v), 6)
                                         for k in ("constante", "gbm", "gru", "gru_recette", "gru_recette2",
                                                   "melange", "final")}
        e = d["final"] - v
        # Bandes d'erreur : biais constant, lent (> 30 j), moyen (1-30 j), rapide (< 1 j), par moyennes glissantes.
        lent = e.rolling(720, center=True, min_periods=1).mean()
        moyen = e.rolling(24, center=True, min_periods=1).mean()
        biais = e.mean()
        bandes = {"biais": biais.pow(2).mean(), "lent": (lent - biais).pow(2).mean().mean(),
                  "moyen": (moyen - lent).pow(2).mean().mean(), "rapide": (e - moyen).pow(2).mean().mean()}
        tot = sum(bandes.values())
        chiffres[f"bandes_{direction}"] = {k: round(float(b / tot), 3) for k, b in bandes.items()}
        cor = np.corrcoef(*[(d[p] - v).to_numpy().ravel()[np.isfinite(v.to_numpy().ravel())]
                            for p in ("gru", "gbm")])[0, 1]
        chiffres[f"correlation_reseau_boosting_{direction}"] = round(float(cor), 3)
        if ecart:
            s = lire(Path(ecart) / f"ecart_{direction}.csv").reindex(v.index)[v.columns]
            p = lire(Path(ecart) / f"pred_{direction}.csv").reindex(v.index)[v.columns]
            ok = v.notna().to_numpy()
            z = (np.abs(v - p) / s.replace(0, np.nan)).to_numpy()[ok]
            chiffres[f"couverture_{direction}"] = {"1sigma": round(float(np.nanmean(z <= 1)), 3),
                                                   "2sigma": round(float(np.nanmean(z <= 2)), 3),
                                                   "rmse_pred": round(rmse(p, v), 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecart", default=None)
    args = ap.parse_args()
    SORTIE.mkdir(exist_ok=True)
    donnees = charger()
    chiffres = {}
    fig_progression()
    fig_validations(chiffres)
    fig_derive(donnees, chiffres)
    fig_circuits(donnees, chiffres)
    fig_zoom(donnees, chiffres)
    chiffres_globaux(donnees, chiffres, args.ecart)
    (SORTIE / "chiffres.json").write_text(json.dumps(chiffres, indent=1, ensure_ascii=False))
    print(json.dumps(chiffres, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
