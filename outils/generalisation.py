"""Enquête sur la généralisation d'une année à l'autre (session du 10 novembre 2026).

    python outils/generalisation.py decomposer runs/rob_cand_g04
    python outils/generalisation.py comparer runs/rob_cand_g04 runs/gen_h1b_g04
    python outils/generalisation.py facteur directe          # poids de runs/gru_hors_sac

decomposer : erreur par circuit, bloc de 30 jours et moitié, découpée en bandes : biais constant,
  lent (moyenne glissante de 30 jours), moyen (1 à 30 jours), rapide (moins d'un jour).
comparer : candidat contre référence sur les mêmes cellules qu'evaluate.py, avec les critères de
  sélection : (1) les deux directions baissent ; (2) gain non concentré : la RMSE baisse encore sans
  le circuit qui gagne le plus et sans le 1 % d'heures qui gagnent le plus ; (3) au moins une moitié
  gagnante par direction, aucune moitié perdante de plus de 0.0005.
facteur : facteur de la correction de niveau estimé hors bloc, sur l'année d'entraînement seule.
  Modèles entraînés sans un tiers de l'année (outils/gru_hors_sac.py), départs tous les 60 jours
  dans ce tiers, 7 jours réels d'initialisation. La correction est linéaire en k : sortie = base +
  k × correction, donc une simulation avec et une sans correction suffisent pour toute la grille.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import donnees, monde  # noqa: E402
from outils.valider import DIRECTIONS  # noqa: E402


def bandes(e):
    """e : DataFrame heures × circuits (NaN = inconnu) -> biais, lent (> 30 j), moyen, rapide (< 1 j)."""
    biais = e.mean()
    m30 = e.rolling(720, center=True, min_periods=240).mean()
    m24 = e.rolling(24, center=True, min_periods=8).mean()
    return biais, m30 - biais, m24 - m30, e - m24


def erreurs(dossier, direction):
    """(verite, erreur) alignées sur heures_a_predire, NaN là où la vérité manque."""
    validation = DIRECTIONS[direction][1]
    verite = pd.read_csv(validation / "verite_horaire.csv").set_index("ut_ms")
    heures = pd.read_csv(validation / "test" / "heures_a_predire.csv")["ut_ms"]
    verite = verite.reindex(heures)
    pred = pd.read_csv(Path(dossier) / direction / "pred.csv").set_index("ut_ms").reindex(heures)[verite.columns]
    return verite, pred - verite


def decomposer(dossier):
    for direction in DIRECTIONS:
        verite, e = erreurs(dossier, direction)
        masque, total = e.notna(), (e ** 2).sum().sum()
        biais, lent, moyen, rapide = bandes(e)
        parts = pd.DataFrame({
            "rmse": np.sqrt((e ** 2).sum() / masque.sum()), "part_mse": (e ** 2).sum() / total,
            "biais": biais, "biais2": biais ** 2 * masque.sum() / total,
            "lent": (lent.where(masque) ** 2).sum() / total, "moyen": (moyen.where(masque) ** 2).sum() / total,
            "rapide": (rapide.where(masque) ** 2).sum() / total})
        h = np.arange(len(e)) // 720
        blocs = np.sqrt((e ** 2).sum(axis=1).groupby(h).sum() / masque.sum(axis=1).groupby(h).sum())
        moitie = np.arange(len(e)) >= len(e) // 2
        demi = [np.sqrt((e[moitie == k] ** 2).sum().sum() / masque[moitie == k].sum().sum()) for k in (False, True)]
        print(f"\n=== {direction} : RMSE {np.sqrt(total / masque.sum().sum()):.6f} ; moitiés {demi[0]:.4f} / {demi[1]:.4f}")
        print("parts de la MSE :", {k: round(float(parts[k].sum()), 3) for k in ("biais2", "lent", "moyen", "rapide")})
        print("RMSE par bloc de 30 jours :", blocs.round(3).tolist())
        print(parts.sort_values("part_mse", ascending=False).head(10).round(4).to_string())


def comparer(ref, cand, montrer=8):
    bilan = {}
    for direction in DIRECTIONS:
        verite, e_ref = erreurs(ref, direction)
        e_cand = erreurs(cand, direction)[1]
        m = e_ref.notna().to_numpy()
        e_r, e_c = (e_ref ** 2).fillna(0).to_numpy(), (e_cand ** 2).fillna(0).to_numpy()
        rmse = lambda e, lignes=slice(None), cols=slice(None): np.sqrt(e[lignes][:, cols].sum() / m[lignes][:, cols].sum())
        moitie = np.arange(len(m)) >= len(m) // 2
        gain_c, gain_h = (e_r - e_c).sum(0), (e_r - e_c).sum(1)
        heures = np.ones(len(m), bool)
        heures[np.argsort(gain_h)[::-1][:len(m) // 100]] = False
        autres = np.arange(m.shape[1]) != gain_c.argmax()
        b = {"ref": rmse(e_r), "cand": rmse(e_c),
             "moities": [(rmse(e_r, moitie == k), rmse(e_c, moitie == k)) for k in (False, True)],
             "meilleur": verite.columns[gain_c.argmax()],
             "d_sans_meilleur": rmse(e_c, cols=autres) - rmse(e_r, cols=autres),
             "d_sans_1pc": rmse(e_c, heures) - rmse(e_r, heures)}
        bilan[direction] = b
        print(f"\n{direction} : {b['ref']:.6f} -> {b['cand']:.6f} ({b['cand'] - b['ref']:+.6f}) ; moitiés "
              + " ; ".join(f"{a:.4f} -> {c:.4f}" for a, c in b["moities"])
              + f" ; sans {b['meilleur']} {b['d_sans_meilleur']:+.6f} ; sans 1 % d'heures {b['d_sans_1pc']:+.6f}")
        par_c = pd.DataFrame({"ref": np.sqrt(e_r.sum(0) / m.sum(0)), "cand": np.sqrt(e_c.sum(0) / m.sum(0)),
                              "part_du_gain": gain_c / max(abs(gain_c.sum()), 1e-12)}, index=verite.columns)
        print(par_c.reindex(par_c.part_du_gain.abs().sort_values(ascending=False).index).head(montrer).round(4).to_string())
    deux = all(b["cand"] < b["ref"] for b in bilan.values())
    diffus = all(b["d_sans_meilleur"] < 0 and b["d_sans_1pc"] < 0 for b in bilan.values())
    moities = all(max(a - c for a, c in b["moities"]) > 0 and min(a - c for a, c in b["moities"]) > -0.0005
                  for b in bilan.values())
    print(f"\ndeux directions {deux} ; non concentré {diffus} ; moitiés {moities}")
    return bilan


def facteur(direction, grille=(0, 0.5, 1, 1.5, 2, 2.5, 3, 4), sacs=3, init=168, pas=60 * 24):
    annee = DIRECTIONS[direction][0]
    _, x, y = donnees.charger_annee(annee)
    dmop = donnees.lire_dmop(annee)
    grille = np.asarray(grille, dtype=float)
    sse, n = np.zeros(len(grille)), 0
    bornes = np.linspace(0, len(y), sacs + 1).astype(int)
    for s, (a, b) in enumerate(zip(bornes[:-1], bornes[1:])):
        dossier = RACINE / "runs" / "gru_hors_sac" / direction / f"sac{s}"
        for depart in range(a, b - init - 720, pas):
            base, correction, total = 0.0, 0.0, 0
            for membre in sorted(dossier.glob("membre*")):
                paquet, config, stats, modeles = monde.charger(membre)
                f, _ = monde.construire_variables(x.iloc[depart:b], config, dmop, paquet.get("commandes"))
                xt = torch.as_tensor(monde.normaliser_entrees(f, stats)[None], dtype=torch.float32)
                yi = torch.as_tensor(monde.normaliser_courants(y[None, depart:depart + init], stats["y_moyenne"],
                                                              stats["y_ecart"]), dtype=torch.float32)
                with torch.no_grad():
                    for modele in modeles:
                        avec = modele(xt[:, :init], yi, xt[:, init:])[0].numpy()
                        poids = modele.poids_niveau.clone(), modele.poids_lent.clone()
                        modele.poids_niveau.zero_()
                        modele.poids_lent.zero_()
                        sans = modele(xt[:, :init], yi, xt[:, init:])[0].numpy()
                        modele.poids_niveau.copy_(poids[0])
                        modele.poids_lent.copy_(poids[1])
                        base, correction, total = base + sans, correction + avec - sans, total + 1
            # Membres de même taille : moyenne simple de tous les modèles, comme mixte.py.
            base, correction = base / total, correction / total
            verite = y[depart + init:b]
            connue = np.isfinite(verite)
            for i, k in enumerate(grille):
                sse[i] += ((np.clip(base + k * correction, 0, None) - np.nan_to_num(verite)) ** 2)[connue].sum()
            n += connue.sum()
    rmse = np.sqrt(sse / n)
    print(f"{direction} ({annee}) : k {grille.tolist()}")
    print(f"RMSE hors bloc {np.round(rmse, 5).tolist()} -> k* = {grille[rmse.argmin()]}")
    return grille, rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="action", required=True)
    sous.add_parser("decomposer").add_argument("dossier")
    c = sous.add_parser("comparer")
    c.add_argument("ref")
    c.add_argument("cand")
    sous.add_parser("facteur").add_argument("direction", choices=sorted(DIRECTIONS))
    a = ap.parse_args()
    torch.set_num_threads(16)
    if a.action == "decomposer":
        decomposer(a.dossier)
    elif a.action == "comparer":
        comparer(a.ref, a.cand)
    else:
        facteur(a.direction)


if __name__ == "__main__":
    main()
