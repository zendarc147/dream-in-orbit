"""Ablations du world model : que valent les éclipses, le Soleil, les commandes, la mémoire ?

    python outils/ablations.py                         # direction directe, configuration par défaut de dio/monde.py
    python outils/ablations.py --direction directe inverse --graines 0 1

Ablations du world model GRU à encodeur (dio/monde.py, modèle « gru »). Chaque variante est entraînée puis
notée via train.py -> predict.py -> evaluate.py (voir valider.py). Le tableau est écrit dans docs/ABLATIONS.md.
"""
import argparse
import json
from pathlib import Path

from valider import valider

RACINE = Path(__file__).resolve().parent.parent
VARIANTES = {
    "complet": {},
    "sans éclipses": {"sans": ["eclipses"]},
    "sans saaf (Soleil)": {"sans": ["saaf"]},
    "sans commandes AOOO/ATTT (branche vers NPWD2881)": {"commandes": None},
    "sans dmop (commandes)": {"sans": ["dmop"]},
    "sans mémoire (ni encodeur ni GRU)": {"memoire": False},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", nargs="+", default=["directe"], choices=["directe", "inverse"])
    ap.add_argument("--graines", nargs="+", type=int, default=None)
    ap.add_argument("--options", default="{}", help="options JSON communes du world model (dio/monde.py)")
    args = ap.parse_args()
    base = json.loads(args.options)
    if args.graines is not None:
        base["graines"] = args.graines
    lignes = []
    for nom, changement in VARIANTES.items():
        options = {**base, **changement}
        if "sans" in changement:
            options["sans"] = sorted(set(base.get("sans", [])) | set(changement["sans"]))
        dossier = "ablation_" + nom.split(" (")[0].replace(" ", "_").replace("é", "e")
        scores = valider(dossier, "gru", options, {}, True, args.direction)
        lignes.append((nom, scores))
    reference = lignes[0][1]
    texte = ["# Ablations du world model", "",
             f"Configuration de base : `{json.dumps(base)}`", "",
             "| Variante | " + " | ".join(f"RMSE {d}" for d in args.direction) + " | Écart vs complet |",
             "|---|" + "---:|" * (len(args.direction) + 1)]
    for nom, scores in lignes:
        ecart = sum(scores[d]["rmse"] - reference[d]["rmse"] for d in args.direction) / len(args.direction)
        texte.append(f"| {nom} | " + " | ".join(f"{scores[d]['rmse']:.4f}" for d in args.direction)
                     + f" | {ecart:+.4f} |")
    (RACINE / "docs").mkdir(exist_ok=True)
    (RACINE / "docs/ABLATIONS.md").write_text("\n".join(texte) + "\n")
    print("\n".join(texte))


if __name__ == "__main__":
    main()
