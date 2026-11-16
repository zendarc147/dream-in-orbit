"""Démo « rêve contre réalité » sur la validation directe, avec la bande d'incertitude.

    python outils/valider.py --nom final            # entraîne runs/final/directe sur annee1 seule
    python outils/demo.py --poids runs/final/directe --out docs/demo/reve_contre_realite.png

Le modèle doit avoir été entraîné sans la période affichée (validation directe :
entraînement sur annee1 seul, affichage d'annee2). La bande = ± 2 écarts-types des 20 GRU
de la part réseau (dispersion de l'ensemble). Produit aussi la courbe de dérive (RMSE glissante sur 7 jours).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import modeles  # noqa: E402
CIRCUITS = ["NPWD2551", "NPWD2451", "NPWD2532", "NPWD2851"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poids", type=Path, required=True)
    ap.add_argument("--validation", type=Path, default=RACINE / "validation")
    ap.add_argument("--out", type=Path, default=RACINE / "docs/demo/reve_contre_realite.png")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporaire:
        pred_csv, ecart_csv = Path(temporaire) / "pred.csv", Path(temporaire) / "ecart.csv"
        meta = json.loads((args.poids / "modele.json").read_text())
        modeles.module(meta["modele"]).predire(args.validation / "test", pred_csv, args.poids,
                                               sortie_incertitude=ecart_csv)
        pred = pd.read_csv(pred_csv, index_col="ut_ms")
        ecart = pd.read_csv(ecart_csv, index_col="ut_ms")
    verite = pd.read_csv(args.validation / "verite_horaire.csv", index_col="ut_ms").reindex(pred.index)
    dates = pd.to_datetime(pred.index, unit="ms")

    # Lissage journalier pour la lisibilité sur ~2 ans ; la RMSE reste horaire.
    def jour(s):
        return s.rolling(24, min_periods=1, center=True).mean()

    total_v = verite.sum(axis=1, min_count=1)
    total_p = pred.sum(axis=1)
    # Écart-type du total : borne haute (corrélation parfaite entre circuits).
    total_e = ecart.sum(axis=1)
    fig, axes = plt.subplots(len(CIRCUITS) + 2, 1, figsize=(14, 3 * (len(CIRCUITS) + 2)), sharex=True)
    series = [("Consommation totale (33 circuits)", total_v, total_p, total_e)]
    series += [(c, verite[c], pred[c], ecart[c]) for c in CIRCUITS]
    for ax, (titre, v, p, e) in zip(axes, series):
        ax.plot(dates, jour(v), color="#1f2937", lw=0.8, label="réalité")
        ax.plot(dates, jour(p), color="#2563eb", lw=0.9, label="rêve (simulation)")
        ax.fill_between(dates, jour(p - 2 * e), jour(p + 2 * e), color="#2563eb", alpha=0.2,
                        label="± 2 σ de l'ensemble")
        rmse = np.sqrt(np.nanmean((p - v) ** 2))
        ax.set_title(f"{titre} — RMSE horaire {rmse:.4f}", loc="left", fontsize=10)
        ax.set_ylabel("courant")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    erreur2 = ((pred - verite) ** 2).mean(axis=1)
    derive = np.sqrt(erreur2.rolling(24 * 7, min_periods=24).mean())
    axes[-1].plot(dates, derive, color="#dc2626", lw=1)
    axes[-1].set_title("Dérive : RMSE glissante sur 7 jours (33 circuits)", loc="left", fontsize=10)
    axes[-1].set_ylabel("RMSE")
    fig.suptitle("Dream in Orbit — rêve contre réalité (validation directe : entraîné sur annee1, simulé sur annee2)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(args.out, dpi=110)
    masque = verite.notna()
    print(f"Démo écrite dans {args.out} ; RMSE globale "
          f"{np.sqrt(np.nanmean(((pred - verite) ** 2)[masque])):.4f}")


if __name__ == "__main__":
    main()
