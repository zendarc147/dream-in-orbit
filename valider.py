"""Valider un modèle dans les deux directions, toujours via train.py, predict.py et evaluate.py.

    python valider.py --nom e1_boosting --modele boosting
    python valider.py --nom e1_boosting --sans-entrainement --options-prediction '{"recalage": 1.0}'

Directe : entraînement sur annee1, test sur validation/ (annee2).
Inverse : entraînement sur annee2, test sur validation_inverse/ (annee1),
dossier créé par : python make_validation.py --annee data/annee1 --out validation_inverse
Score de référence = moyenne des deux RMSE.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent
DIRECTIONS = {
    "directe": ("annee1", RACINE / "validation"),
    "inverse": ("annee2", RACINE / "validation_inverse"),
}


def lancer(commande):
    resultat = subprocess.run([sys.executable, *map(str, commande)], cwd=RACINE,
                              capture_output=True, text=True)
    if resultat.returncode:
        raise RuntimeError(f"Échec de {commande[0]} :\n{resultat.stdout}\n{resultat.stderr}")
    return resultat.stdout


def valider(nom, modele, options, options_prediction, entrainer=True, directions=DIRECTIONS):
    scores = {}
    for direction in directions:
        annee, validation = DIRECTIONS[direction]
        poids = RACINE / "runs" / nom / direction
        debut = time.time()
        if entrainer:
            lancer(["train.py", "--annees", annee, "--out", poids, "--modele", modele,
                    "--options", json.dumps(options)])
        duree_train = time.time() - debut
        pred = poids / "pred.csv"
        debut = time.time()
        lancer(["predict.py", "--test_dir", validation / "test", "--out", pred, "--poids", poids,
                "--options", json.dumps(options_prediction)])
        duree_pred = time.time() - debut
        sortie = lancer(["evaluate.py", "--pred", pred, "--verite", validation / "verite_horaire.csv",
                         "--heures", validation / "test" / "heures_a_predire.csv"])
        rmse = float(re.search(r"RMSE = ([0-9.]+)", sortie).group(1))
        scores[direction] = {"rmse": rmse, "train_s": round(duree_train), "pred_s": round(duree_pred)}
        print(f"{nom} {direction} : RMSE {rmse:.6f} (train {duree_train:.0f} s, predict {duree_pred:.0f} s)",
              flush=True)
    if len(scores) == 2:
        moyenne = sum(s["rmse"] for s in scores.values()) / 2
        print(f"{nom} moyenne : {moyenne:.6f}")
        scores["moyenne"] = moyenne
    journal = RACINE / "runs" / nom / f"scores{'_' + json.dumps(options_prediction) if options_prediction else ''}.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(scores, indent=2) + "\n")
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--modele", default=None)
    ap.add_argument("--options", default="{}")
    ap.add_argument("--options-prediction", default="{}")
    ap.add_argument("--sans-entrainement", action="store_true")
    ap.add_argument("--direction", choices=sorted(DIRECTIONS), nargs="+", default=list(DIRECTIONS))
    args = ap.parse_args()
    import modeles
    valider(args.nom, args.modele or modeles.FINAL, json.loads(args.options),
            json.loads(args.options_prediction), not args.sans_entrainement, args.direction)


if __name__ == "__main__":
    main()
