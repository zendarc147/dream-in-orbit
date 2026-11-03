# Dream in Orbit — world model de la consommation électrique d'un satellite

Un réseau de neurones récurrent apprend comment le satellite réagit à son planning
(commandes dmop, modes de pointage ftl) et à son contexte (Soleil saaf, événements evtf,
tendances ltdata), puis simule seul ses 33 circuits heure par heure sur toute une période,
à partir de ses 7 premiers jours réels (`power_init.csv`). Sujet : `sujet.md`.
Scores et historique des essais : `RESULTATS.md`. Ablations : `ABLATIONS.md`.

## Résultats (RMSE, `evaluate.py`)

| Modèle | Directe (annee1 → annee2) | Inverse (annee2 → annee1) | Moyenne |
|---|---:|---:|---:|
| Constante | 0.1058 | 0.1252 | 0.1155 |
| Gradient boosting + recalage sur power_init (`baselines/`) | 0.0951 | 0.0999 | 0.0975 |
| World model GRU, ensemble de 5 graines (v1 + bruit) | 0.0872 | 0.0971 | 0.0921 |
| **Ensemble mixte de world models GRU (v1 + bruit, v2r ; 2 × 5 graines), modèle final** | **0.0863** | **0.0964** | **0.0914** |

Démo : `demo/reve_contre_realite.png` (rêve contre réalité sur annee2, bande ± 2σ de
l'ensemble, courbe de dérive).

## Installation

Python 3.13 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

Sans uv : `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
Les commandes ci-dessous supposent l'environnement activé (`source .venv/bin/activate`)
ou préfixées par `uv run`.

## Prédire (interface d'évaluation)

```bash
python predict.py --test_dir DOSSIER --out predictions.csv
```

`DOSSIER` contient `dmop.csv`, `evtf.csv`, `ftl.csv`, `saaf.csv`, `ltdata.csv`,
`power_init.csv` et `heures_a_predire.csv`. Les poids commités sont dans `poids/`
(`modele.json`, `membres.json`, `membre*/monde.pt`). Environ 20 s sur CPU.

Avec l'incertitude (écart-type de l'ensemble) :

```bash
python predict.py --test_dir DOSSIER --out pred.csv --options '{"sortie_incertitude": "ecart.csv"}'
```

## Entraîner (une commande)

```bash
python train.py
```

Entraîne l'ensemble final sur annee1 + annee2 et écrit `poids/`. Les données horaires
sont lues dans `data/prepared/` ; si elles manquent, `prepare.py` les recrée à partir
de `data/annee*/` (qui doit alors contenir `power.csv`).

Sur CPU l'entraînement complet est long ; sur un GPU T4 (Colab), il tient en moins
d'une heure : le code utilise automatiquement CUDA s'il est disponible.

```bash
# Colab : après avoir cloné le dépôt
pip install -r requirements.txt
python train.py
```

## Valider

Deux directions, toujours via `train.py` → `predict.py` → `evaluate.py` :

```bash
python reconstruire_power.py      # recrée data/annee*/power.csv depuis data/prepared si absent
python make_validation.py --annee data/annee1 --out validation_inverse
python valider.py --nom essai --modele gru
```

- directe : entraînement sur annee1, test sur `validation/` (annee2) ;
- inverse : entraînement sur annee2, test sur `validation_inverse/` (annee1) ;
- score de référence : moyenne des deux RMSE.

## Ablations et démo

```bash
python ablations.py               # sans éclipses, sans saaf, sans dmop, sans mémoire
python demo.py --poids runs/NOM/directe --out demo/reve_contre_realite.png
```

## Tests

```bash
python -m unittest test_prepare baselines.test_ridge test_predict
```

## Organisation

| Fichier | Rôle |
|---|---|
| `prepare.py` | contextes bruts → 65 variables horaires (schéma appris sur annee1) |
| `variables.py` | variables v2 : fenêtres passées/futures, temps depuis/jusqu'à, phase orbitale, cos/sin |
| `monde.py` | world model GRU (encodeur sur power_init → h0, simulateur 2 couches, tête 33 circuits) |
| `mixte.py` | modèle final : moyenne de deux ensembles de world models (variables v1 et v2r) |
| `dmop_etats.py` | état « dernière commande » des sous-systèmes dmop (testé, non retenu) |
| `donnees.py` | chargement des années préparées et des dossiers de test |
| `train.py`, `predict.py` | entraînement et prédiction (modèle choisi dans `modeles.py`) |
| `valider.py`, `ablations.py`, `demo.py` | validation, ablations, figure rêve contre réalité |
| `baselines/` | constante, Ridge, gradient boosting (+ recalage sur power_init) |
