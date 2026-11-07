"""Figures du rapport du 8 novembre 2026 (docs/rapport/figures_8_novembre/*.pdf) et chiffres du texte.

    python docs/rapport/figures_8_novembre.py --preds DOSSIER

Modèle du 4 novembre : runs/c2_niveau_600/. Modèle actuel : runs/rob_cand_g04/ (configuration de
poids/, entraînée sur une seule période par direction). DOSSIER contient final_{directe,inverse}_ecart.csv,
écrits par dio/mixte.predire avec les poids de runs/rob_cand_g04/ (écart-type de l'ensemble).
Les scores du journal restent ceux des pred.csv calculés sur GPU.
"""
import argparse
import json
from pathlib import Path

from figures import (BLEU, ORANGE, VERT, GRIS, ENCRE, ENCRE2, RACINE, DIRECTIONS,  # règle aussi le style
                     lire, rmse, constante, rmse_par_mois)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SORTIE = Path(__file__).resolve().parent / "figures_8_novembre"
AVANT, APRES = "c2_niveau_600", "rob_cand_g04"


def pred(run, direction, index, colonnes):
    return lire(RACINE / "runs" / run / direction / "pred.csv").reindex(index)[colonnes]


def fig_progression():
    # RESULTATS.md, sessions « commandes », « robustesse temporelle » et « régularisations » (graines 0 à 4).
    etapes = [
        ("Modèle du 4 novembre (CPU)", 0.0853, 0.0940, 0.0897),
        ("Même modèle relancé sur GPU T4", 0.0851, 0.0941, 0.0896),
        ("+ commandes AOOO/ATTT vers NPWD2881", 0.0832, 0.0914, 0.0873),
        ("+ correction sur le résidu", 0.0829, 0.0901, 0.0865),
        ("+ second terme lent", 0.0827, 0.0892, 0.0859),
        ("+ borne du résidu à 10 (prédiction)", 0.08265, 0.08806, 0.08535),
        ("+ masquage SAAF, abandon des états", 0.08255, 0.08847, 0.08551),
        ("tête de régime partagée (non retenue)", 0.08349, 0.08865, 0.08607),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    y = np.arange(len(etapes))[::-1]
    for yi, (_, d, i, m) in zip(y, etapes):
        ax.plot([d, i], [yi, yi], color="#d6d5d0", lw=1.2, zorder=1)
        ax.text(max(d, i) + 0.0004, yi, f"{m:.4f}", va="center", fontsize=7.5, color=ENCRE2)
    ax.scatter([e[1] for e in etapes], y, s=34, color=BLEU, zorder=3, label="directe (annee1 → annee2)",
               edgecolor="white", linewidth=1)
    ax.scatter([e[2] for e in etapes], y, s=34, color=ORANGE, zorder=3, marker="s",
               label="inverse (annee2 → annee1)", edgecolor="white", linewidth=1)
    ax.scatter([etapes[-1][1], etapes[-1][2]], [y[-1], y[-1]], s=34, color="white", zorder=4,
               edgecolor=GRIS, linewidth=1)
    ax.axhline(0.5, color=GRIS, lw=0.6, ls=(0, (3, 3)))
    ax.set_yticks(y, [e[0] for e in etapes])
    ax.set_xlim(0.0815, 0.0965)
    ax.set_xlabel("RMSE (plus bas = mieux) ; chiffre à droite : moyenne des deux directions")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(SORTIE / "progression.pdf")
    plt.close(fig)


def fig_npwd2881(chiffres):
    v = lire(RACINE / "validation/verite_horaire.csv")["NPWD2881"]
    avant = pred(AVANT, "directe", v.index, ["NPWD2881"])["NPWD2881"]
    apres = pred(APRES, "directe", v.index, ["NPWD2881"])["NPWD2881"]
    jours = np.arange(len(v)) / 24
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.9), gridspec_kw={"height_ratios": [1.3, 1]})
    fig.subplots_adjust(hspace=0.45)
    ax = axes[0]
    f = (jours >= 294) & (jours < 315)
    ax.plot(jours[f], v[f], color=ENCRE, lw=0.8, label="réalité")
    ax.plot(jours[f], avant[f], color=ORANGE, lw=0.9, label="4 novembre (sans branche)")
    ax.plot(jours[f], apres[f], color=BLEU, lw=0.9, label="actuel (branche de commandes)")
    ax.set_title("NPWD2881, trois semaines de la validation directe, valeurs horaires", loc="left")
    ax.set_ylabel("courant")
    ax.set_ylim(-0.05, 1.05 * float(v[f].max()) + 0.15)
    ax.legend(loc="upper right", ncol=3, fontsize=7, frameon=True, facecolor="white", edgecolor="none")
    ax = axes[1]
    for nom, s, couleur in [("4 novembre", avant, ORANGE), ("actuel", apres, BLEU)]:
        courbe = rmse_par_mois(s, v)
        ax.plot(np.arange(1, len(courbe) + 1), courbe, color=couleur, marker="o", ms=2.5, label=nom)
        chiffres[f"npwd2881_mois_{nom}"] = [round(float(c), 3) for c in courbe]
        chiffres[f"npwd2881_rmse_{nom}"] = round(rmse(s, v), 3)
    ax.set_title("NPWD2881, RMSE par bloc de 30 jours", loc="left")
    ax.set_xlabel("mois de simulation")
    ax.set_ylabel("RMSE du mois")
    ax.legend(loc="upper left")
    fig.savefig(SORTIE / "npwd2881.pdf")
    plt.close(fig)


def fig_derive(donnees, chiffres):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    for ax, (direction, d) in zip(axes, donnees.items()):
        for nom, cle, couleur, style in [("constante", "constante", GRIS, (0, (3, 2))),
                                         ("4 novembre", "avant", ORANGE, "-"),
                                         ("actuel", "apres", BLEU, "-")]:
            courbe = rmse_par_mois(d[cle], d["verite"])
            ax.plot(np.arange(1, len(courbe) + 1), courbe, color=couleur, ls=style, label=nom, marker="o", ms=2.5)
            chiffres[f"derive_{direction}_{cle}"] = [round(float(c), 4) for c in courbe]
        ax.set_title(f"Validation {direction}", loc="left")
        ax.set_xlabel("mois de simulation (blocs de 30 jours)")
    axes[0].set_ylabel("RMSE du mois")
    axes[0].legend(loc="upper left")
    fig.savefig(SORTIE / "derive.pdf")
    plt.close(fig)


def fig_circuits(donnees, chiffres):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharex=True)
    for ax, (direction, d) in zip(axes, donnees.items()):
        v = d["verite"]
        r = {cle: np.sqrt(((d[cle] - v) ** 2).mean()) for cle in ("avant", "apres")}
        e2 = ((d["apres"] - v) ** 2).sum()
        haut = (e2 / e2.sum()).sort_values(ascending=False).index[:8]
        chiffres[f"circuits_{direction}"] = {k: [round(float(r["avant"][k]), 3), round(float(r["apres"][k]), 3),
                                                 round(float(e2[k] / e2.sum()), 3)] for k in haut}
        y = np.arange(len(haut))[::-1]
        h = 0.36
        ax.barh(y + h / 2, r["avant"][haut], h, color="#c9c8c2", label="4 novembre")
        ax.barh(y - h / 2, r["apres"][haut], h, color=BLEU, label="actuel")
        ax.set_yticks(y, [k.replace("NPWD", "") for k in haut])
        ax.set_title(f"Validation {direction}", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RMSE du circuit")
    axes[0].legend(loc="lower right")
    fig.savefig(SORTIE / "circuits.pdf")
    plt.close(fig)


def fig_niveau(chiffres):
    v = lire(RACINE / "validation_inverse/verite_horaire.csv")["NPWD2531"]
    series = [("ensemble mixte (it. 5), sans correction", "ref_mixte", GRIS),
              ("4 novembre : correction sur l'écart à la moyenne", AVANT, ORANGE),
              ("actuel : correction sur le résidu, borne 10", APRES, BLEU)]
    dates = pd.to_datetime(v.index, unit="ms")
    jours = (dates - dates[0]).total_seconds() / 86400
    fin = jours <= 240
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.plot(jours[fin], v.rolling(24, min_periods=1, center=True).mean()[fin], color=ENCRE, lw=0.7, label="réalité")
    for nom, run, couleur in series:
        s = pred(run, "inverse", v.index, ["NPWD2531"])["NPWD2531"]
        ax.plot(jours[fin], s.rolling(24, min_periods=1, center=True).mean()[fin], color=couleur, lw=1.0, label=nom)
        chiffres[f"niveau_2531_{run}_90j"] = round(rmse(s[jours <= 90], v[jours <= 90]), 3)
        chiffres[f"niveau_2531_{run}_tout"] = round(rmse(s, v), 3)
    ax.set_xlabel("jours depuis la fin de power_init")
    ax.set_ylabel("courant NPWD2531")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.5), fontsize=7, frameon=True, facecolor="white",
              edgecolor="none")
    fig.savefig(SORTIE / "niveau.pdf")
    plt.close(fig)


def fig_regimes(chiffres):
    """Calibration de la tête de régime et RMSE par tranche d'horizon (runs/regimes/bilan.json)."""
    bilan = json.loads((RACINE / "runs/regimes/bilan.json").read_text())
    calib = json.loads((RACINE / "runs/rg_l05ctx_g04/calibration_regime.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRIS, lw=0.6, ls=(0, (3, 2)))
    for direction, couleur, marque in [("directe", BLEU, "o"), ("inverse", ORANGE, "s")]:
        dec = calib[direction]["deciles"]
        ax.plot([x["p_predite"] for x in dec], [x["frequence"] for x in dec], color=couleur, marker=marque,
                ms=3, label=direction)
        chiffres[f"brier_{direction}"] = [round(calib[direction]["brier"], 3),
                                          round(calib[direction]["brier_frequence_constante"], 3)]
    ax.set_xlabel("probabilité prédite")
    ax.set_ylabel("fréquence observée")
    ax.set_title("Calibration (déciles)", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_aspect("equal")
    tranches = list(bilan["rg_ref_g04"]["tranches"]["directe"])
    x = np.arange(len(tranches))
    for ax, direction in zip(axes[1:], ("directe", "inverse")):
        for run, nom, couleur in [("rg_ref_g04", "référence", ENCRE2), ("rg_l05ctx_g04", "tête de régime", VERT),
                                  ("rg_h540_g04", "épisodes de 540 j", ORANGE)]:
            t = bilan[run]["tranches"][direction]
            ax.plot(x, [t[k]["globale"] for k in tranches], color=couleur, marker="o", ms=2.5, label=nom)
        ax.set_xticks(x, [k.replace("-", "–").replace("365–680", "365+") for k in tranches], fontsize=6.5,
                      rotation=30)
        ax.set_title(f"RMSE par tranche, {direction}", loc="left")
        ax.set_xlabel("jours après power_init")
    axes[1].legend(loc="upper left", fontsize=6.5)
    fig.savefig(SORTIE / "regimes.pdf")
    plt.close(fig)


def fig_ablations():
    # docs/ABLATIONS.md (v1 seul, 2 graines, configuration du 5 novembre).
    complet = (0.0840, 0.0918)
    variantes = [("sans mémoire", 0.0884, 0.1032, 0.0079), ("sans dmop\n(comptes + branche)", 0.0885, 0.0944, 0.0035),
                 ("sans saaf", 0.0865, 0.0950, 0.0028), ("sans branche\nAOOO/ATTT", 0.0857, 0.0939, 0.0019),
                 ("sans éclipses", 0.0844, 0.0910, -0.0002)]
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    y = np.arange(len(variantes))[::-1]
    h = 0.34
    ax.barh(y + h / 2, [v[1] - complet[0] for v in variantes], h, color=BLEU, label="directe")
    ax.barh(y - h / 2, [v[2] - complet[1] for v in variantes], h, color=ORANGE, label="inverse")
    for yi, v in zip(y, variantes):
        ax.text(max(v[1] - complet[0], v[2] - complet[1], 0) + 0.0002, yi, f"moyenne {v[3]:+.4f}",
                va="center", fontsize=7.5, color=ENCRE2)
    ax.axvline(0, color=GRIS, lw=0.6)
    ax.set_yticks(y, [v[0] for v in variantes])
    ax.set_xlim(-0.0012, 0.0145)
    ax.set_xlabel("hausse de la RMSE par rapport au modèle complet (directe 0.0840, inverse 0.0918)")
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
        d = {"verite": v, "constante": constante(annee, v),
             "avant": pred(AVANT, direction, v.index, v.columns), "apres": pred(APRES, direction, v.index, v.columns)}
        donnees[direction] = d
        chiffres[f"rmse_{direction}"] = {k: round(rmse(d[k], v), 4) for k in ("constante", "avant", "apres")}
        ecart = lire(args.preds / f"final_{direction}_ecart.csv").reindex(v.index)[v.columns]
        connu = v.notna().to_numpy()
        for k in (1, 2):
            dedans = ((v - d["apres"]).abs() <= k * ecart).to_numpy()[connu]
            chiffres[f"couverture_{k}sigma_{direction}"] = round(float(dedans.mean()), 3)
    fig_progression()
    fig_npwd2881(chiffres)
    fig_derive(donnees, chiffres)
    fig_circuits(donnees, chiffres)
    fig_niveau(chiffres)
    fig_regimes(chiffres)
    fig_ablations()
    (SORTIE / "chiffres.json").write_text(json.dumps(chiffres, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(chiffres, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
