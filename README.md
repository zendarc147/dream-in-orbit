# Dream in Orbit — world model de la consommation électrique d'un satellite

Un réseau de neurones récurrent apprend comment le satellite réagit à son planning
(commandes dmop, dont les commandes AOOO et ATTT, modes de pointage ftl) et à son contexte (Soleil saaf, événements evtf,
tendances ltdata), puis simule seul ses 33 circuits heure par heure sur toute une période,
à partir de ses 7 premiers jours réels (`power_init.csv`). Sujet : [`docs/sujet.md`](docs/sujet.md).
Scores et historique des essais : [`docs/RESULTATS.md`](docs/RESULTATS.md). Ablations : [`docs/ABLATIONS.md`](docs/ABLATIONS.md). Ensemble GRU + boosting : [`docs/JOURNAL_ENSEMBLE.md`](docs/JOURNAL_ENSEMBLE.md).

## Résultats (RMSE, `evaluate.py`)

| Modèle | Directe (annee1 → annee2) | Inverse (annee2 → annee1) | Moyenne |
|---|---:|---:|---:|
| Constante | 0.1058 | 0.1252 | 0.1155 |
| Gradient boosting + recalage sur power_init (`baselines/`) | 0.0951 | 0.0999 | 0.0975 |
| World model GRU, ensemble de 5 graines (v1 + bruit) | 0.0872 | 0.0971 | 0.0921 |
| Ensemble mixte de world models GRU (v1 + bruit, v2r ; 2 × 5 graines) | 0.0863 | 0.0964 | 0.0914 |
| + 600 pas avec moyenne exponentielle des poids + correction de niveau de power_init à demi-vie apprise | 0.0853 | 0.0940 | 0.0897 |
| + commandes AOOO/ATTT (impulsions et états décalés) dans une branche vers NPWD2881 + correction de niveau sur le résidu (terme rapide + terme lent) | 0.0827 | 0.0892 | 0.0859 |
| + borne du résidu initial élargie seulement à la prédiction pour les circuits variables à l'entraînement | 0.0827 | 0.0881 | 0.0854 |
| + régularisations de robustesse (masquage SAAF par blocs, abandon groupé des états de commandes) | 0.0826 | 0.0885 | 0.0855 |
| + familles de commandes dmop en entrée du GRU + correction de niveau × 1.5 à la prédiction | 0.0791 | 0.0836 | 0.0813 |
| **+ mélange avec un boosting à sacs temporels enrichi (commandes planifiées, moyennes exponentielles), poids 0.3, modèle final** | **0.0780** | **0.0810** | **0.0795** |

Les deux derniers changements viennent d'une enquête sur l'erreur d'une année à l'autre.
Les familles de commandes sont des comptes horaires par code dmop sans lettre de variante
(vocabulaire appris sur l'entraînement). La correction de niveau issue de power_init est
multipliée par 1.5 à la prédiction : estimé hors bloc sur l'année d'entraînement seule, ce facteur
corrige une correction apprise trop faible. Graines 5–9 : 0.0787 / 0.0843. Gain dans les deux
directions et sur plusieurs circuits ; NPWD2532 se dégrade. Détail, critères et limites dans
`docs/RESULTATS.md`, section « Généralisation d'une année à l'autre ».

Le modèle final est un ensemble : 0.7 × world model GRU + 0.3 × boosting à sacs temporels
(un HistGradientBoosting par circuit, 5 sacs). Le boosting, sans mémoire, reçoit en plus des
variables du GRU les commandes planifiées par famille (heures depuis la dernière, comptes dans
l'heure et les six heures suivantes, périodes début/fin appariées) et des moyennes
exponentielles du flux solaire, de l'ombre, du pointage et des communications. Ensemble, graines
0–4 : 0.0780 / 0.0810 ; graines 5–9 : 0.0778 / 0.0814 (GRU seul : 0.0791 / 0.0836 et
0.0787 / 0.0842, recalculés en local). Le GRU reste majoritaire et le retour au GRU seul est
une option de prédiction. Ablations, critères et limites : `docs/JOURNAL_ENSEMBLE.md`.

Démo : [`docs/demo/reve_contre_realite.png`](docs/demo/reve_contre_realite.png) (rêve contre réalité sur annee2, bande ± 2σ de
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
(`modele.json`, `ensemble.json`, `membres.json`, `membre*/monde.pt`, `gbm/gbm_sacs.pkl.xz`).
Environ 20 s et 1.5 Go de mémoire sur CPU.

GRU seul (sans le boosting), avec les mêmes poids :

```bash
python predict.py --test_dir DOSSIER --out predictions.csv --options '{"poids_gbm": 0}'
```

Avec l'incertitude (écart-type de l'ensemble) :

```bash
python predict.py --test_dir DOSSIER --out pred.csv --options '{"sortie_incertitude": "ecart.csv"}'
```

## Entraîner (une commande)

```bash
python train.py
```

Entraîne l'ensemble final (world models GRU puis boosting) sur annee1 + annee2 et écrit
`poids/`. Pour ne réentraîner que le boosting (environ 6 min sur CPU) en gardant les world
models déjà installés, ce qui reproduit exactement les poids livrés :

```bash
python train.py --options '{"gru_depuis": "poids"}'
```

Les données horaires
sont lues dans `data/prepared/` ; si elles manquent, `dio/prepare.py` les recrée à partir
de `data/annee*/` (qui doit alors contenir `power.csv`). Les deux années se suivent
sans trou : elles forment une seule période, et les épisodes d'entraînement peuvent
traverser la frontière annee1/annee2.

Sur CPU l'entraînement complet est long ; sur un GPU T4 (Colab), il tient en moins
d'une heure : le code utilise automatiquement CUDA s'il est disponible.

```bash
# Colab : après avoir cloné le dépôt
pip install -r requirements.txt
python train.py
```

### Sur Kaggle depuis le terminal

`outils/lancer_kaggle.py` envoie le dépôt tel qu'il est sur le disque, lance la
commande sur un GPU T4 Kaggle et rapatrie `runs/` dans le dépôt, comme un lancement local.
Prérequis : `uv tool install kaggle --with six` et un token API Kaggle dans
`~/.kaggle/access_token` (kaggle.com > Settings > API).

```bash
python outils/lancer_kaggle.py -- outils/valider.py --nom essai --modele gru_mixte
python outils/lancer_kaggle.py --recuperer poids -- train.py
python outils/lancer_kaggle.py --sans-attendre -- outils/valider.py --nom essai   # puis : --rapatrier
python outils/lancer_kaggle.py --fente 2 -- outils/valider.py --nom a ";" outils/valider.py --nom b   # 2e session, 2 essais à la suite
```

Le code part dans le script du noyau (quelques centaines de ko) ; le dataset privé ne porte
que `data/`, envoyé une fois (`--donnees` pour les renvoyer). Kaggle
accepte deux sessions GPU à la fois (`--fente`).

## Valider

Deux directions, toujours via `train.py` → `predict.py` → `evaluate.py` :

```bash
python outils/reconstruire_power.py   # recrée data/annee*/power.csv depuis data/prepared si absent
python make_validation.py --annee data/annee2 --out validation
python make_validation.py --annee data/annee1 --out validation_inverse
python outils/valider.py --nom essai --modele gru
```

- directe : entraînement sur annee1, test sur `validation/` (annee2) ;
- inverse : entraînement sur annee2, test sur `validation_inverse/` (annee1) ;
- score de référence : moyenne des deux RMSE.

## Ablations et démo

```bash
python outils/ablations.py        # sans éclipses, sans saaf, sans dmop, sans mémoire
python outils/demo.py --poids runs/NOM/directe --out docs/demo/reve_contre_realite.png
```

## Tests

```bash
python -m unittest discover -s tests
```

## Organisation

```
train.py, predict.py             entraînement et prédiction (interface d'évaluation)
evaluate.py, make_validation.py  scripts du kit, inchangés
dio/                             le modèle
  prepare.py                     contextes bruts -> 65 variables horaires (schéma appris sur annee1)
  variables.py                   variables v2r : fenêtres passées/futures, temps depuis/jusqu'à, phase orbitale, cos/sin ;
                                 options physique (flux solaire par face) et familles de commandes dmop
  commandes.py                   commandes AOOO et ATTT : impulsions et états décalés (délai appris sur NPWD2881)
  donnees.py                     chargement des années préparées et des dossiers de test
  monde.py                       world model GRU (encodeur sur power_init, simulateur 2 couches, tête 33 circuits,
                                 correction de niveau, branche des commandes AOOO/ATTT vers NPWD2881)
  mixte.py                       moyenne de deux ensembles de world models (variables v1 et v2r)
  ensemble.py                    modèle final : 0.7 × world models (mixte.py) + 0.3 × boosting à sacs (gbm_sacs.py)
  autoregressif.py               étude non livrée : GRU autorégressif, flux continus, scheduled sampling
  modeles.py                     registre des modèles utilisables par train.py et predict.py
baselines/                       constante, Ridge, gradient boosting (+ recalage sur power_init) ;
                                 boosting à sacs temporels (composant du modèle final) et Ridge physique
outils/                          validation, ablations, démo, diagnostics, lancement sur Kaggle,
                                 mélange GRU + classique, empilement, résidu hors bloc, généralisation
tests/                           tests unitaires
poids/                           poids du modèle final
data/                            kit (annee1, annee2) et données horaires préparées
docs/                            sujet, journal des résultats, ablations, rapport, exploration, démo
```

Le code ne garde que ce que sert le modèle final et les ablations. Les variantes testées puis
écartées (décrites dans `docs/RESULTATS.md`) restent dans l'historique git.
