"""Exécuté sur Kaggle par lancer_kaggle.py, ne pas lancer à la main.

Retrouve la copie du dépôt (dataset attaché), installe les dépendances, recrée
validation_inverse/, lance la commande de kaggle_commande.json et copie les dossiers
demandés dans /kaggle/working (onglet Output, téléchargé par lancer_kaggle.py).
"""
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEPOT = Path("/tmp/depot")
SORTIE = Path("/kaggle/working")


def lancer(*commande):
    print("$", *commande, flush=True)
    subprocess.run(list(commande), cwd=DEPOT, check=True)


def copier_depot():
    # Kaggle décompresse parfois les archives d'un dataset : accepter les deux formes.
    trouves = glob.glob("/kaggle/input/**/kaggle_commande.json", recursive=True)
    if trouves:
        shutil.copytree(Path(trouves[0]).parent, DEPOT)
    else:
        shutil.unpack_archive(glob.glob("/kaggle/input/**/depot.tar.gz", recursive=True)[0], DEPOT)


def main():
    copier_depot()
    commande = json.loads((DEPOT / "kaggle_commande.json").read_text())
    # ipykernel retiré : inutile ici et lourd à réinstaller.
    exigences = [l for l in (DEPOT / "requirements.txt").read_text().split() if not l.startswith("ipykernel")]
    lancer(sys.executable, "-m", "pip", "install", "-q", *exigences)
    lancer(sys.executable, "-c", "import sys, torch; print(sys.version, 'torch', torch.__version__,"
           " 'cuda', torch.cuda.is_available() and torch.cuda.get_device_name(0))")
    lancer(sys.executable, "reconstruire_power.py")
    lancer(sys.executable, "make_validation.py", "--annee", "data/annee1", "--out", "validation_inverse")
    try:
        lancer(sys.executable, *commande["args"])
    finally:
        # Copier même en cas d'échec, pour garder les train.log.
        for dossier in commande["recuperer"]:
            if (DEPOT / dossier).exists():
                shutil.copytree(DEPOT / dossier, SORTIE / dossier, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
