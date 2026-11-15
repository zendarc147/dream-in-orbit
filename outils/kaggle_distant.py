"""Exécuté sur Kaggle par outils/lancer_kaggle.py, ne pas lancer à la main.

Retrouve les données (dataset attaché), y dépose le code embarqué (CODE, ajouté en tête de
ce fichier par lancer_kaggle.py), installe les dépendances, recrée
validation/ et validation_inverse/, lance la commande de kaggle_commande.json et copie les dossiers
demandés dans /kaggle/working (onglet Output, téléchargé par lancer_kaggle.py).
"""
import base64
import glob
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEPOT = Path("/tmp/depot")
SORTIE = Path("/kaggle/working")


def lancer(*commande):
    print("$", *commande, flush=True)
    subprocess.run(list(commande), cwd=DEPOT, check=True)


def copier_depot():
    # Kaggle décompresse parfois les archives d'un dataset : accepter les deux formes.
    trouves = glob.glob("/kaggle/input/**/data/manifest.json", recursive=True)
    if trouves:
        shutil.copytree(Path(trouves[0]).parent.parent, DEPOT)
    else:
        archive = glob.glob("/kaggle/input/**/*.tar.gz", recursive=True)[0]
        shutil.unpack_archive(archive, DEPOT)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz") as tar:
        tar.extractall(DEPOT, filter="data")


def main():
    copier_depot()
    commande = json.loads((DEPOT / "kaggle_commande.json").read_text())
    # ipykernel retiré : inutile ici et lourd à réinstaller.
    exigences = [l for l in (DEPOT / "requirements.txt").read_text().split() if not l.startswith("ipykernel")]
    # Compte sans internet (non vérifié) : pip échoue, on garde alors les paquets préinstallés de Kaggle.
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--retries", "1", "--timeout", "15",
                       *exigences], cwd=DEPOT).returncode:
        print("pip indisponible : paquets préinstallés de Kaggle", flush=True)
    lancer(sys.executable, "-c", "import numpy, pandas, sklearn; print('numpy', numpy.__version__, 'pandas',"
           " pandas.__version__, 'sklearn', sklearn.__version__)")
    lancer(sys.executable, "-c", "import sys, torch; print(sys.version, 'torch', torch.__version__,"
           " 'cuda', torch.cuda.is_available() and torch.cuda.get_device_name(0))")
    if commande.get("gpu"):  # session GPU demandée mais servie sans GPU (compte non vérifié) : ne rien calculer
        import torch
        if not torch.cuda.is_available():
            sys.exit("GPU demandé mais absent : arrêt sans calcul sur CPU")
    lancer(sys.executable, "outils/reconstruire_power.py")
    lancer(sys.executable, "make_validation.py", "--annee", "data/annee2", "--out", "validation")
    lancer(sys.executable, "make_validation.py", "--annee", "data/annee1", "--out", "validation_inverse")
    lancer(sys.executable, "outils/faire_chrono.py")
    try:
        for args in commande.get("commandes", [commande.get("args")]):
            print("$", *args, flush=True)
            subprocess.run([sys.executable, *args], cwd=DEPOT)  # un essai raté n'arrête pas les suivants
    finally:
        # Copier même en cas d'échec, pour garder les train.log.
        for dossier in commande["recuperer"]:
            if (DEPOT / dossier).is_dir():
                shutil.copytree(DEPOT / dossier, SORTIE / dossier, dirs_exist_ok=True)
            elif (DEPOT / dossier).is_file():
                (SORTIE / dossier).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(DEPOT / dossier, SORTIE / dossier)


if __name__ == "__main__":
    main()
