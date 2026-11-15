"""Lancer une commande du projet sur un GPU T4 Kaggle et rapatrier ses résultats.

    python outils/lancer_kaggle.py -- outils/valider.py --nom essai --modele gru_mixte
    python outils/lancer_kaggle.py --recuperer poids -- train.py
    python outils/lancer_kaggle.py --sans-attendre -- outils/valider.py --nom essai   # rapatrier plus tard :
    python outils/lancer_kaggle.py --rapatrier
    python outils/lancer_kaggle.py --fente 2 -- outils/valider.py --nom essai2   # 2e lancement simultané (2 au plus)
    python outils/lancer_kaggle.py -- outils/valider.py --nom a ";" outils/valider.py --nom b   # à la suite, une session
    python outils/lancer_kaggle.py --cpu -- outils/valider.py --nom gbm --modele gbm_sacs   # sans GPU

Le dépôt tel qu'il est sur le disque (fichiers suivis et non suivis, hors .gitignore) part
sur Kaggle : les données (data/) dans le dataset privé
UTILISATEUR/dream-in-orbit-code, envoyé une fois (--donnees pour le renvoyer ; Kaggle peut
mettre quelques minutes à servir la nouvelle version), et le code dans le script du noyau
lui-même, donc toujours à jour. outils/kaggle_distant.py lance la ou les commandes, puis les
dossiers demandés (runs/ par défaut) reviennent ici, fusionnés dans le dépôt comme après
un lancement local. Le journal Kaggle est gardé dans runs/_kaggle/.
Prérequis : CLI kaggle (uv tool install kaggle --with six) et ~/.kaggle/access_token.
"""
import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
UTILISATEUR = "zendarc148"
DATASET = f"{UTILISATEUR}/dream-in-orbit-code"
NOYAU = f"{UTILISATEUR}/dream-in-orbit"
INUTILES = ("docs/",)
DONNEES = ("data/",)


def fente(numero):
    """Un noyau par fente : plusieurs lancements peuvent tourner en même temps."""
    global NOYAU
    if numero:
        NOYAU = f"{NOYAU}-{numero}"


def kaggle(*args, verifier=True):
    try:  # le téléchargement des sorties (poids du boosting : plusieurs centaines de Mo) peut dépasser 10 min
        resultat = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                                  timeout=3600 if args[:2] == ("kernels", "output") else 600)
    except subprocess.TimeoutExpired:  # appel réseau bloqué : rendre un échec, attendre() réessaie
        resultat = subprocess.CompletedProcess(args, 1, "", "délai dépassé")
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


def fichiers_depot():
    """Fichiers suivis et non suivis (hors .gitignore), sans les sorties et documents locaux."""
    fichiers = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=RACINE,
                              capture_output=True, text=True, check=True).stdout.split("\0")
    return [f for f in fichiers if f and (RACINE / f).is_file()  # un fichier supprimé peut rester suivi
            and not f.startswith(INUTILES) and not f.endswith((".ipynb", ".pdf")) and f != "pred.csv"]


def envoyer_donnees(dossier, forcer):
    """Dataset privé des données (data/) : envoyé une fois, ou avec --donnees."""
    if not forcer and not kaggle("datasets", "status", DATASET, verifier=False).returncode:
        return
    with tarfile.open(dossier / "donnees.tar.gz", "w:gz") as tar:
        for f in fichiers_depot():
            if f.startswith(DONNEES):
                tar.add(RACINE / f, arcname=f)
    (dossier / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET.split("/")[1], "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}))
    if kaggle("datasets", "status", DATASET, verifier=False).returncode:
        kaggle("datasets", "create", "-p", str(dossier), "-t", "-q")  # privé par défaut
    else:
        kaggle("datasets", "version", "-p", str(dossier), "-m", "données", "-t", "-q")
    statut, sortie = attendre(["datasets", "status", DATASET],
                              lambda s: "ready" if "ready" in s else "error" if "error" in s else None, 10)
    if statut != "ready":
        raise RuntimeError(f"Dataset non prêt : {sortie}")


def pousser_noyau(dossier, commande, gpu=True):
    """Le code (hors données) part dans le script du noyau, encodé en base64 : il est toujours
    celui du disque au moment du lancement, sans attendre une nouvelle version du dataset."""
    paquet = io.BytesIO()
    with tarfile.open(fileobj=paquet, mode="w:gz") as tar:
        for f in fichiers_depot():
            if not f.startswith(DONNEES + ("poids/", "poids_candidats/")):
                tar.add(RACINE / f, arcname=f)
        contenu = json.dumps(commande).encode()
        info = tarfile.TarInfo("kaggle_commande.json")
        info.size = len(contenu)
        tar.addfile(info, io.BytesIO(contenu))
    print(f"Code empaqueté : {paquet.tell() / 1e3:.0f} ko", flush=True)
    script = (RACINE / "outils" / "kaggle_distant.py").read_text()
    (dossier / "distant.py").write_text(f'CODE = "{base64.b64encode(paquet.getvalue()).decode()}"\n' + script)
    metadonnees = {
        "id": NOYAU, "title": NOYAU.split("/")[1], "code_file": "distant.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [DATASET],
    }
    if gpu:
        metadonnees["machine_shape"] = "NvidiaTeslaT4"
    (dossier / "kernel-metadata.json").write_text(json.dumps(metadonnees))
    sortie = kaggle("kernels", "push", "-p", str(dossier)).stdout.strip()
    print(sortie, flush=True)
    if "error" in sortie.lower():  # la CLI rend 0 même quand Kaggle refuse (ex. 2 sessions GPU au plus)
        raise RuntimeError(sortie)


def rapatrier(motif=None):
    statut, sortie = attendre(["kernels", "status", NOYAU],
                              lambda s: next((m for m in ("complete", "error", "cancel") if m in s), None))
    dossier = RACINE / "runs" / "_kaggle" / (time.strftime("%Y%m%d_%H%M%S") + NOYAU.removeprefix(f"{UTILISATEUR}/dream-in-orbit"))
    filtre = ["--file-pattern", motif] if motif else []
    kaggle("kernels", "output", NOYAU, "-p", str(dossier), "-o", "-q", *filtre)
    for sous in dossier.iterdir():
        if sous.is_dir():  # runs/, poids/, ou docs/ pour un fichier comme docs/ABLATIONS.md
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
    ap.add_argument("--recuperer", nargs="+", default=["runs"], help="dossiers ou fichiers rapatriés (défaut : runs)")
    ap.add_argument("--sans-attendre", action="store_true", help="lancer sans attendre la fin")
    ap.add_argument("--rapatrier", action="store_true", help="attendre le dernier lancement et rapatrier")
    ap.add_argument("--donnees", action="store_true", help="renvoyer data/ (après un changement)")
    ap.add_argument("--fente", default="", help="numéro de fente pour lancer en parallèle (défaut : aucune)")
    ap.add_argument("--cpu", action="store_true", help="session CPU sans GPU (boosting, diagnostics)")
    ap.add_argument("--legers", action="store_true",
                    help="rapatrier seulement csv, json et journaux (pas les poids, longs à télécharger)")
    ap.add_argument("args", nargs="*", help="script et arguments, après --")
    a = ap.parse_args()
    fente(a.fente)
    if not a.rapatrier:
        if not a.args:
            ap.error("commande manquante, ex. : -- outils/valider.py --nom essai")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dataset").mkdir()
            (Path(tmp) / "noyau").mkdir()
            commandes, courante = [], []
            for mot in a.args + [";"]:
                if mot == ";":
                    commandes, courante = commandes + [courante] * bool(courante), []
                else:
                    courante.append(mot)
            envoyer_donnees(Path(tmp) / "dataset", a.donnees)
            pousser_noyau(Path(tmp) / "noyau", {"commandes": commandes, "recuperer": a.recuperer, "gpu": not a.cpu},
                          not a.cpu)
        print(f"Lancé : https://www.kaggle.com/code/{NOYAU}", flush=True)
        if a.sans_attendre:
            return
        time.sleep(30)  # laisser Kaggle enregistrer la nouvelle version avant de lire son statut
    rapatrier(r"\.(csv|json|log)$" if a.legers else None)


if __name__ == "__main__":
    main()
