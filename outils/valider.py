"""Valider un modèle dans les deux directions, toujours via train.py, predict.py et evaluate.py.

    python outils/valider.py --nom e1_boosting --modele boosting
    python outils/valider.py --nom e1_boosting --sans-entrainement --options-prediction '{"recalage": 1.0}'

Directe : entraînement sur annee1, test sur validation/ (annee2).
Inverse : entraînement sur annee2, test sur validation_inverse/ (annee1).
Dossiers créés par : python make_validation.py --annee data/annee2 --out validation
                     python make_validation.py --annee data/annee1 --out validation_inverse
Score de référence = moyenne des deux RMSE.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
DIRECTIONS = {
    "directe": ("annee1", RACINE / "validation"),
    "inverse": ("annee2", RACINE / "validation_inverse"),
}
# Validation chronologique (outils/faire_chrono.py) : annee1 + première moitié d'annee2 -> seconde moitié.
# Jamais incluse par défaut ; --direction chrono.
TOUTES = {**DIRECTIONS, "chrono": (["annee1", "validation_chrono/entrainement"], RACINE / "validation_chrono")}


def lancer(commande):
    resultat = subprocess.run([sys.executable, *map(str, commande)], cwd=RACINE,
                              capture_output=True, text=True)
    if resultat.returncode:
        raise RuntimeError(f"Échec de {commande[0]} :\n{resultat.stdout}\n{resultat.stderr}")
    return resultat.stdout


def entrainer_en_parallele(nom, modele, options, directions):
    """Lancer les entraînements des directions en même temps, en partageant les cœurs."""
    coeurs = os.cpu_count() or 1
    env = {**os.environ, "MONDE_THREADS": str(max(1, coeurs // len(directions))),
           "MONDE_PROCESSUS": str(max(1, coeurs // len(directions)))}
    debut, actifs = time.time(), {}
    for direction in directions:
        annee, _ = TOUTES[direction]
        poids = RACINE / "runs" / nom / direction
        poids.mkdir(parents=True, exist_ok=True)
        journal = open(poids / "train.log", "w")
        actifs[direction] = (subprocess.Popen(
            [sys.executable, "train.py", "--annees", *([annee] if isinstance(annee, str) else annee), "--out", str(poids), "--modele", modele,
             "--options", json.dumps(options)], cwd=RACINE, env=env, stdout=journal,
            stderr=subprocess.STDOUT), journal)
    for direction, (processus, journal) in actifs.items():
        if processus.wait():
            raise RuntimeError(f"Échec de l'entraînement {direction}, voir runs/{nom}/{direction}/train.log")
        journal.close()
    return time.time() - debut


def valider(nom, modele, options, options_prediction, entrainer=True, directions=DIRECTIONS):
    scores = {}
    duree_train = entrainer_en_parallele(nom, modele, options, list(directions)) if entrainer else 0.0
    for direction in directions:
        annee, validation = TOUTES[direction]
        poids = RACINE / "runs" / nom / direction
        # Une variante de prédiction n'écrase pas pred.csv (même suffixe que son fichier de scores).
        pred = poids / (f"pred_{json.dumps(options_prediction)}.csv" if options_prediction else "pred.csv")
        debut = time.time()
        lancer(["predict.py", "--test_dir", validation / "test", "--out", pred, "--poids", poids,
                "--options", json.dumps(options_prediction)])
        duree_pred = time.time() - debut
        sortie = lancer(["evaluate.py", "--pred", pred, "--verite", validation / "verite_horaire.csv",
                         "--heures", validation / "test" / "heures_a_predire.csv"])
        rmse = float(re.search(r"RMSE = ([0-9.]+)", sortie).group(1))
        scores[direction] = {"rmse": rmse, "train_s": round(duree_train), "pred_s": round(duree_pred)}
        print(f"{nom} {direction} : RMSE {rmse:.6f} (train {duree_train:.0f} s pour les deux directions"
              f" en parallèle, predict {duree_pred:.0f} s)",
              flush=True)
    if set(scores) == set(DIRECTIONS):
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
    ap.add_argument("--direction", choices=sorted(TOUTES), nargs="+", default=list(DIRECTIONS))
    args = ap.parse_args()
    from dio import modeles
    valider(args.nom, args.modele or modeles.FINAL, json.loads(args.options),
            json.loads(args.options_prediction), not args.sans_entrainement, args.direction)


if __name__ == "__main__":
    main()
