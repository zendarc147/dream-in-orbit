"""Construire un dossier de validation de bout en bout d'un candidat à partir du livré (runs/s51_val_*).

    python outils/candidat.py --nom rc6_val_g04 --base s51_val_g04 --recette rc3_gruA_norm --part 0.5 [--gbm RUN] [--poids 0.5]
    python outils/candidat.py --nom st_T2_c62_g04 --base st_c62_L_g04 --directions c62 --recette st_c62_gruS_g04 \
        --part 1 --gbm2 st_c62_gbmS --poids-gbm2 0.5 --b st_c62_B --part-b 0.5

Copie (liens durs) runs/<base>/<direction>/ dans runs/<nom>/<direction>/, remplace recette/recette.pt (GRU recette),
membre*/ (--mixte), gbm/ (boosting principal), ajoute gbm2/ (second boosting, poids --poids-gbm2 dans la part
boosting) et specialiste/b_NPWD2551/ (recette B, part --part-b de la part boosting de NPWD2551 ; 1 = B seule), écrit
ensemble.json ; puis : python outils/valider.py --nom <nom> --sans-entrainement.
Un RUN est lu dans runs/RUN/<direction>/ s'il existe, sinon dans runs/RUN/ (modèle entraîné par train.py --out).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def source(run, d):
    r = RACINE / "runs" / run
    return r / d if (r / d).is_dir() else r


def lier(a, b):
    shutil.copytree(a, b, copy_function=os.link, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("pred*.csv", "pred*.parts", "hors_sac.csv", "train.log"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nom", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--directions", nargs="+", default=["directe", "inverse"])
    ap.add_argument("--recette")
    ap.add_argument("--mixte", help="GRU mixte (membre*/, membres.json)")
    ap.add_argument("--gbm")
    ap.add_argument("--gbm2")
    ap.add_argument("--poids-gbm2", type=float, default=0.5)
    ap.add_argument("--b", help="recette B de NPWD2551 (modele_b.pkl.xz)")
    ap.add_argument("--part-b", type=float, default=1.0)
    ap.add_argument("--part", type=float)
    ap.add_argument("--poids", type=float)
    ap.add_argument("--sans-specialiste", nargs="*", default=[], help="circuits dont la règle spécialiste est retirée")
    a = ap.parse_args()
    for d in a.directions:
        cible = RACINE / "runs" / a.nom / d
        if cible.exists():
            shutil.rmtree(cible)
        lier(source(a.base, d), cible)
        if a.recette:
            (cible / "recette").mkdir(exist_ok=True)
            (cible / "recette" / "recette.pt").unlink(missing_ok=True)
            os.link(source(a.recette, d) / "recette.pt", cible / "recette" / "recette.pt")
        if a.mixte:
            for m in list(cible.glob("membre[0-9]*")):
                shutil.rmtree(m)
            (cible / "membres.json").unlink(missing_ok=True)
            os.link(source(a.mixte, d) / "membres.json", cible / "membres.json")
            for m in sorted(source(a.mixte, d).glob("membre[0-9]*")):
                lier(m, cible / m.name)
        for option, nom in ((a.gbm, "gbm"), (a.gbm2, "gbm2")):
            if option:
                shutil.rmtree(cible / nom, ignore_errors=True)
                (cible / nom).mkdir()
                for f in source(option, d).glob("gbm_sacs.pkl*"):
                    os.link(f, cible / nom / f.name)
        reglages = json.loads((cible / "ensemble.json").read_text())
        if a.part is not None:
            reglages["part_recette"] = a.part
        if a.poids is not None:
            reglages["poids_gbm"] = a.poids
        if a.gbm2:
            reglages["gbm2"] = a.poids_gbm2
        (cible / "ensemble.json").unlink()  # lien dur : ne jamais écrire dans le fichier de la base
        (cible / "ensemble.json").write_text(json.dumps(reglages, indent=2) + "\n")
        if a.part == 1:  # GRU mixte non lu : retiré du dossier pour que la structure se voie
            for m in list(cible.glob("membre[0-9]*")):
                shutil.rmtree(m)
            (cible / "membres.json").unlink(missing_ok=True)
        chemin = cible / "specialiste" / "specialistes.json"
        regles = {c: r for c, r in json.loads(chemin.read_text()).items() if c not in a.sans_specialiste}
        if a.b:
            lier(source(a.b, d), cible / "specialiste" / "b_NPWD2551")
            regles["NPWD2551"] = {**regles.get("NPWD2551", {}), "recette_b": a.part_b}
            if a.part_b == 1:
                regles["NPWD2551"].pop("gbm_specialiste", None)
                shutil.rmtree(cible / "specialiste" / "gbm_NPWD2551", ignore_errors=True)
        chemin.unlink()
        chemin.write_text(json.dumps(regles, indent=2) + "\n")
    print(f"runs/{a.nom} prêt")


if __name__ == "__main__":
    main()
