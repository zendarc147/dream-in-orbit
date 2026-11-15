"""Fusionner deux paquets du GRU recette (graines différentes, tout le reste identique) en un seul.

    python outils/fusionner_recette.py runs/rc3_gruA_norm/directe runs/rc4_gruA_mm_g59/directe runs/st_gruS10/directe

La prédiction moyenne les graines (dio/recette.predire) : le paquet fusionné est le GRU à 10 graines
(docs/JOURNAL_STRUCTURE.md). Refus si configuration (hors graines), variables, familles, circuits ou statistiques
diffèrent.
"""
import sys
from pathlib import Path

import torch


def fusionner(a, b, sortie):
    pa, pb = (torch.load(Path(d) / "recette.pt", weights_only=True) for d in (a, b))
    ca, cb = ({k: v for k, v in p["config"].items() if k != "graines"} for p in (pa, pb))
    if ca != cb or any(pa[k] != pb[k] for k in ("variables", "familles", "circuits")):
        raise ValueError("configurations, variables, familles ou circuits différents")
    for k in pa["stats"]:
        if not torch.equal(pa["stats"][k], pb["stats"][k]):
            raise ValueError(f"statistiques différentes : {k}")
    if set(pa["config"]["graines"]) & set(pb["config"]["graines"]):
        raise ValueError("graines communes")
    paquet = {**pa, "config": {**pa["config"], "graines": pa["config"]["graines"] + pb["config"]["graines"]},
              "etats": pa["etats"] + pb["etats"]}
    Path(sortie).mkdir(parents=True, exist_ok=True)
    torch.save(paquet, Path(sortie) / "recette.pt")
    return len(paquet["etats"])


if __name__ == "__main__":
    print(fusionner(*sys.argv[1:4]), "graines")
