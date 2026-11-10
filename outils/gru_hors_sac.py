"""Prédictions hors bloc du world model GRU sur son année d'entraînement (apprentissage du résidu).

    python outils/gru_hors_sac.py --nom gru_hors_sac --direction directe

Pour chaque bloc k (un tiers contigu de l'année) : ensemble mixte final (mêmes options que les
poids livrés) entraîné sans ce bloc ni 7 jours de marge de chaque côté, puis simulation du bloc à
partir de ses 7 premiers jours réels, comme à l'évaluation. Les heures d'initialisation restent
NaN dans hors_sac.csv. Aucune donnée de l'année de validation n'intervient.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from dio import donnees, mixte, monde  # noqa: E402
from outils.valider import DIRECTIONS  # noqa: E402

OPTIONS = {"masque_saaf": 0.1, "abandon_etats": 0.5, "borne_niveau_prediction": 10.0, "graines": [0, 1, 2, 3, 4]}
INIT = 168


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    ap.add_argument("--sacs", type=int, default=3)
    args = ap.parse_args()
    annee = DIRECTIONS[args.direction][0]
    heures, x, y = donnees.charger_annee(annee)
    dmop = donnees.lire_dmop(annee)
    n = len(heures)
    hors_sac = np.full(y.shape, np.nan)
    bornes = np.linspace(0, n, args.sacs + 1).astype(int)
    racine = RACINE / "runs" / args.nom / args.direction
    for k, (a, b) in enumerate(zip(bornes[:-1], bornes[1:])):
        dossier = racine / f"sac{k}"
        # Flottants Python : la config est relue par torch.load(weights_only=True).
        mixte.entrainer([annee], dossier, exclure=[float(a / n), float(b / n)], **OPTIONS)
        y_init = np.full((b - a, y.shape[1]), np.nan)
        y_init[:INIT] = y[a:a + INIT]
        membres = json.loads((dossier / "membres.json").read_text())
        simulations = [monde.simuler(dossier / f"membre{i}", x.iloc[a:b], y_init, INIT, dmop)[0]
                       for i in range(len(membres))]
        hors_sac[a + INIT:b] = np.mean(simulations, axis=0)[INIT:]
        print(f"{args.direction} sac {k} : heures {a}-{b} simulées", flush=True)
    schema = donnees.charger_schema()
    pd.DataFrame(hors_sac, index=pd.Index(heures, name="ut_ms"),
                 columns=schema["target_columns"]).to_csv(racine / "hors_sac.csv")


if __name__ == "__main__":
    main()
