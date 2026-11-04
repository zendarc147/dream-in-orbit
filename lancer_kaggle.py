"""Lancer une commande du projet sur un GPU T4 Kaggle et rapatrier ses résultats.

    python lancer_kaggle.py -- valider.py --nom essai --modele gru_mixte
    python lancer_kaggle.py --recuperer poids -- train.py
    python lancer_kaggle.py --sans-attendre -- valider.py --nom essai   # rapatrier plus tard :
    python lancer_kaggle.py --rapatrier

Le dépôt tel qu'il est sur le disque (fichiers suivis et non suivis, hors .gitignore)
part dans le dataset privé UTILISATEUR/dream-in-orbit-code ; kaggle/distant.py le lance
sur Kaggle puis les dossiers demandés (runs/ par défaut) reviennent ici, fusionnés dans
le dépôt comme après un lancement local. Le journal Kaggle est gardé dans runs/_kaggle/.
Prérequis : CLI kaggle (uv tool install kaggle --with six) et ~/.kaggle/access_token.
"""
import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent
UTILISATEUR = "zendarc147"
DATASET = f"{UTILISATEUR}/dream-in-orbit-code"
NOYAU = f"{UTILISATEUR}/dream-in-orbit"


def kaggle(*args, verifier=True):
    resultat = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if verifier and resultat.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} :\n{resultat.stdout}\n{resultat.stderr}")
    return resultat


def attendre(args, fini, pause=20):
    """Relancer `kaggle args` jusqu'à ce que fini(sortie en minuscules) donne un statut final."""
    while True:
        sortie = kaggle(*args, verifier=False).stdout.strip().lower()
        statut = fini(sortie)
        if statut:
            return statut, sortie
        time.sleep(pause)


def envoyer_depot(dossier, commande):
    fichiers = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=RACINE,
                              capture_output=True, text=True, check=True).stdout.split("\0")
    with tarfile.open(dossier / "depot.tar.gz", "w:gz") as tar:
        for f in filter(None, fichiers):
            if (RACINE / f).is_file():  # un fichier supprimé mais encore suivi n'existe plus
                tar.add(RACINE / f, arcname=f)
        contenu = json.dumps(commande).encode()
        info = tarfile.TarInfo("kaggle_commande.json")
        info.size = len(contenu)
        tar.addfile(info, io.BytesIO(contenu))
    print(f"Dépôt empaqueté : {(dossier / 'depot.tar.gz').stat().st_size / 1e6:.0f} Mo", flush=True)
    (dossier / "dataset-metadata.json").write_text(json.dumps(
        {"title": "dream-in-orbit-code", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}))
    if kaggle("datasets", "status", DATASET, verifier=False).returncode:
        kaggle("datasets", "create", "-p", str(dossier), "-t", "-q")  # privé par défaut
    else:
        kaggle("datasets", "version", "-p", str(dossier), "-m", " ".join(commande["args"]), "-t", "-q")
    statut, sortie = attendre(["datasets", "status", DATASET],
                              lambda s: "ready" if "ready" in s else "error" if "error" in s else None, 10)
    if statut != "ready":
        raise RuntimeError(f"Dataset non prêt : {sortie}")


def pousser_noyau(dossier):
    shutil.copy(RACINE / "kaggle" / "distant.py", dossier)
    (dossier / "kernel-metadata.json").write_text(json.dumps({
        "id": NOYAU, "title": "dream-in-orbit", "code_file": "distant.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
    }))
    print(kaggle("kernels", "push", "-p", str(dossier)).stdout.strip(), flush=True)


def rapatrier():
    statut, sortie = attendre(["kernels", "status", NOYAU],
                              lambda s: next((m for m in ("complete", "error", "cancel") if m in s), None))
    dossier = RACINE / "runs" / "_kaggle" / time.strftime("%Y%m%d_%H%M%S")
    kaggle("kernels", "output", NOYAU, "-p", str(dossier), "-o", "-q")
    for sous in dossier.iterdir():
        if sous.is_dir():
            shutil.copytree(sous, RACINE / sous.name, dirs_exist_ok=True)
    journaux = list(dossier.glob("*.log"))
    print(f"Statut Kaggle : {sortie}\nSorties : {dossier}")
    if statut != "complete":
        if journaux:  # journal Kaggle : JSON de lignes {"data": ...}
            lignes = [l.get("data", "") for l in json.loads(journaux[0].read_text() or "[]")]
            print("".join(lignes)[-3000:])
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recuperer", nargs="+", default=["runs"], help="dossiers rapatriés (défaut : runs)")
    ap.add_argument("--sans-attendre", action="store_true", help="lancer sans attendre la fin")
    ap.add_argument("--rapatrier", action="store_true", help="attendre le dernier lancement et rapatrier")
    ap.add_argument("args", nargs="*", help="script et arguments, après --")
    a = ap.parse_args()
    if not a.rapatrier:
        if not a.args:
            ap.error("commande manquante, ex. : -- valider.py --nom essai")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dataset").mkdir()
            (Path(tmp) / "noyau").mkdir()
            envoyer_depot(Path(tmp) / "dataset", {"args": a.args, "recuperer": a.recuperer})
            pousser_noyau(Path(tmp) / "noyau")
        print(f"Lancé : https://www.kaggle.com/code/{NOYAU}", flush=True)
        if a.sans_attendre:
            return
        time.sleep(30)  # laisser Kaggle enregistrer la nouvelle version avant de lire son statut
    rapatrier()


if __name__ == "__main__":
    main()
