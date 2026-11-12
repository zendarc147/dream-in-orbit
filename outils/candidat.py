"""Construire un dossier de validation de bout en bout d'un candidat à partir du livré (runs/s51_val_*).

    python outils/candidat.py --nom rc6_val_g04 --base s51_val_g04 --recette rc3_gruA_norm --part 0.5 [--gbm RUN] [--poids 0.5]

Copie runs/<base>/<direction>/ dans runs/<nom>/<direction>/, remplace recette/recette.pt (GRU recette) et/ou
gbm/ (boosting principal), écrit ensemble.json ; puis : python outils/valider.py --nom <nom> --sans-entrainement.
"""
import argparse
import json
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--recette")
    ap.add_argument("--gbm")
    ap.add_argument("--part", type=float)
    ap.add_argument("--poids", type=float)
    ap.add_argument("--sans-specialiste", nargs="*", default=[], help="circuits dont la règle spécialiste est retirée")
    a = ap.parse_args()
    for d in ("directe", "inverse"):
        cible = RACINE / "runs" / a.nom / d
        if cible.exists():
            shutil.rmtree(cible)
        shutil.copytree(RACINE / "runs" / a.base / d, cible, ignore=shutil.ignore_patterns("pred*.csv"))
        if a.recette:
            shutil.copy(RACINE / "runs" / a.recette / d / "recette.pt", cible / "recette" / "recette.pt")
        if a.gbm:
            shutil.rmtree(cible / "gbm")
            (cible / "gbm").mkdir()
            for f in (RACINE / "runs" / a.gbm / d).glob("gbm_sacs.pkl*"):
                shutil.copy(f, cible / "gbm" / f.name)
        reglages = json.loads((cible / "ensemble.json").read_text())
        if a.part is not None:
            reglages["part_recette"] = a.part
        if a.poids is not None:
            reglages["poids_gbm"] = a.poids
        (cible / "ensemble.json").write_text(json.dumps(reglages, indent=2) + "\n")
        if a.sans_specialiste:
            chemin = cible / "specialiste" / "specialistes.json"
            regles = {c: r for c, r in json.loads(chemin.read_text()).items() if c not in a.sans_specialiste}
            chemin.write_text(json.dumps(regles, indent=2) + "\n")
    print(f"runs/{a.nom} prêt")


if __name__ == "__main__":
    main()
