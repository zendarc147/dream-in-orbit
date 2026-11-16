# Dream in Orbit — world model de la consommation électrique d'un satellite

Des réseaux de neurones récurrents apprennent comment le satellite réagit à son planning (commandes dmop,
modes de pointage ftl) et à son contexte (Soleil saaf, événements evtf, tendances ltdata). À partir de ses
7 premiers jours réels (`power_init.csv`), le modèle simule seul ses 33 circuits heure par heure sur toute
la période. Sujet : [`docs/sujet.md`](docs/sujet.md). Rapport final :
[`docs/rapport/rapport_final_2026.pdf`](docs/rapport/rapport_final_2026.pdf).

## Modèle final (structure S2)

```
prédiction = 0.5 × part réseau + 0.5 × part boosting

part réseau   = 0.5 × GRU S borné   (dio/recette.py, 10 graines, recalé sur power_init)
              + 0.5 × GRU à boucle  (dio/recette.py, 10 graines, relit sa propre prédiction, sans recalage)
part boosting = boosting à sacs temporels sur les variables S (baselines/gbm_sacs.py, recalé sur power_init)

spécialistes  NPWD2551 : part boosting = recette B, boosting propre au circuit (baselines/recette_b.py)
              NPWD2532 : 0.6 × boosting + 0.4 × world model GRU à variables physiques (dio/monde.py)
              NPWD2531 : hors des marqueurs de régime du planning, niveau médian recalé (baselines/marqueurs.py)
```

- Les deux GRU sont le world model : eux seuls simulent la période pas à pas. Le boosting n'a pas de
  mémoire : il corrige heure par heure.
- Les deux GRU voient les mêmes 165 variables « S » (`dio/caracteristiques.py`), calculées à partir des
  seuls contextes bruts.
- Le GRU à boucle réinjecte sa propre sortie à chaque heure. Il est appris avec un scheduled sampling qui
  descend jusqu'à p = 0. Les 7 jours réels ne servent qu'à réchauffer son état.
- Incertitude : écart-type des 20 GRU de la part réseau.
- Choix, mesures et limites : [`docs/journaux/JOURNAL_GRU_BOUCLE.md`](docs/journaux/JOURNAL_GRU_BOUCLE.md)
  et [`docs/journaux/JOURNAL_STRUCTURE.md`](docs/journaux/JOURNAL_STRUCTURE.md).

## Résultats (RMSE, `evaluate.py`)

Directe : entraînement sur annee1, test sur annee2. Inverse : entraînement sur annee2, test sur annee1.

| Modèle | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| Constante | 0.1058 | 0.1252 | 0.1155 |
| Gradient boosting + recalage sur power_init (`baselines/boosting.py`) | 0.0951 | 0.0999 | 0.0975 |
| World model GRU, ensemble de 5 graines | 0.0872 | 0.0971 | 0.0921 |
| + moyenne exponentielle des poids, correction de niveau apprise sur power_init | 0.0853 | 0.0940 | 0.0897 |
| + commandes AOOO/ATTT dans une branche vers NPWD2881, correction de niveau sur le résidu | 0.0827 | 0.0892 | 0.0859 |
| + familles de commandes dmop, correction de niveau × 1.5 à la prédiction | 0.0791 | 0.0836 | 0.0813 |
| + mélange avec un boosting à sacs temporels (poids 0.3) | 0.0780 | 0.0810 | 0.0795 |
| + spécialistes NPWD2531 et NPWD2532 | 0.0767 | 0.0791 | 0.0779 |
| + boosting « recette » et GRU recette ; boosting à 0.5 | 0.0751 | 0.0785 | 0.0768 |
| + boosting propre à NPWD2551 (livré L, jusqu'au 15 novembre) | 0.0750 | 0.0785 | 0.0767 |
| Structure C1 : GRU S borné à la place du GRU recette | 0.0744 | 0.0782 | 0.0763 |
| Structure T3 : GRU S borné seul, boosting S, recette B pour NPWD2551 | 0.0746 | 0.0804 | 0.0775 |
| **Structure S2 : T3 + GRU à boucle à moitié dans la part réseau (modèle final)** | **0.0746** | **0.0805** | **0.0775** |

L'historique complet de ces essais est dans [`docs/RESULTATS.md`](docs/RESULTATS.md) et `docs/journaux/`.

S2 n'est pas le meilleur score interne : L et C1 font mieux en directe / inverse. S2 est un choix de
l'utilisateur, qui voulait livrer un modèle plus sobre, avec un world model qui relit son propre état. Sur la
validation à origine glissante (quatre coupures dans annee2, réentraînement à chaque coupure), la moyenne
de S2 vaut 0.0737, contre 0.0742 pour T3. Les poids des candidats non retenus restent sur le disque dans
`poids_candidats/` ; git ne suit que leurs `INDEX.md` (scores et verdicts).

Ablations du world model : [`docs/ABLATIONS.md`](docs/ABLATIONS.md). Démo rêve contre réalité, avec la bande
d'incertitude et la courbe de dérive : [`docs/demo/reve_contre_realite.png`](docs/demo/reve_contre_realite.png).

## Installation

Python 3.13 et [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
```

Sans uv : `python -m venv .venv && .venv/bin/pip install -r requirements.txt`. Les commandes ci-dessous
supposent l'environnement activé (`source .venv/bin/activate`) ou sont préfixées par `uv run`.

## Prédire (interface d'évaluation)

```bash
python predict.py --test_dir DOSSIER --out predictions.csv
```

`DOSSIER` contient `dmop.csv`, `evtf.csv`, `ftl.csv`, `saaf.csv`, `ltdata.csv`, `power_init.csv` et
`heures_a_predire.csv`. Les poids sont dans `poids/`. Environ 1 min sur CPU.

Options (`--options` en JSON) :

```bash
python predict.py --test_dir DOSSIER --out pred.csv --options '{"poids_gbm": 0}'                       # part réseau seule
python predict.py --test_dir DOSSIER --out pred.csv --options '{"specialistes": false}'                # sans spécialistes
python predict.py --test_dir DOSSIER --out pred.csv --options '{"sortie_incertitude": "ecart.csv"}'     # + écart-type
```

## Entraîner (une commande)

```bash
python train.py
```

Entraîne le modèle final sur annee1 + annee2 et écrit `poids/` : les deux GRU, le world model physique, les
boostings et la porte de NPWD2531. Le code utilise CUDA s'il est disponible. Sur CPU, l'entraînement des
réseaux prend des heures. Pour ne refaire que les boostings, en gardant les réseaux déjà entraînés :

```bash
python train.py --options '{"reseaux_depuis": "poids"}'
```

Les données horaires sont lues dans `data/prepared/`. Si elles manquent, `dio/prepare.py` les recrée à
partir de `data/annee*/`, qui doit alors contenir `power.csv` (`python outils/reconstruire_power.py`).
Les deux années se suivent sans trou : elles forment une seule période.

Reproductibilité : l'arrêt anticipé des boostings et l'entraînement des GRU dépendent des calculs en virgule
flottante du matériel. Un réentraînement sur un autre matériel donne des poids proches, pas identiques au bit près.

## Valider

```bash
python outils/reconstruire_power.py   # recrée data/annee*/power.csv depuis data/prepared si absent
python make_validation.py --annee data/annee2 --out validation
python make_validation.py --annee data/annee1 --out validation_inverse
python outils/valider.py --nom essai  # train.py -> predict.py -> evaluate.py, dans les deux directions
```

Le score de référence est la moyenne des deux RMSE. `python outils/faire_chrono.py --glissant` crée les
coupures glissantes (`--direction c50 c62 c75 c87` dans `valider.py`).

Sur un GPU T4 Kaggle depuis le terminal (CLI `kaggle` et `~/.kaggle/access_token` requis) :

```bash
python outils/lancer_kaggle.py -- outils/valider.py --nom essai
```

## Ablations et démo

```bash
python outils/ablations.py --direction directe inverse --graines 0 1
python outils/demo.py --poids runs/essai/directe --out docs/demo/reve_contre_realite.png
```

## Tests

```bash
python -m unittest discover -s tests
```

## Organisation

```
train.py, predict.py         entraînement et prédiction (interface d'évaluation)
evaluate.py, make_validation.py
                             scripts du kit, inchangés
dio/                         les réseaux et les variables
  ensemble.py                modèle final S2 : assemblage des composants et des spécialistes
  recette.py                 GRU S borné et GRU à boucle
  monde.py, mixte.py         world model GRU à encodeur (spécialiste NPWD2532, ablations)
  caracteristiques.py        variables « S » tirées des contextes bruts
  prepare.py, variables.py,  grille horaire, variables v1/v2r, commandes AOOO/ATTT, géométrie orbitale
  commandes.py, contexte.py
  donnees.py                 chargement des années et des dossiers de test
  modeles.py                 registre des modèles de train.py et predict.py
baselines/                   constante, Ridge, gradient boosting ; boosting à sacs (gbm_sacs.py),
                             recette B (NPWD2551) et porte par marqueurs (NPWD2531) du modèle final
outils/                      validation, ablations, démo, coupures chronologiques, lancement sur Kaggle
tests/                       tests unitaires (tests/controles/ : empreintes de contrôle de la recette B)
poids/                       poids du modèle final
data/                        kit (annee1, annee2) et données horaires préparées
docs/                        sujet, rapport final et rapports d'étape (rapport/), résultats, ablations,
                             journaux, exploration, démo
```

Les variantes testées puis écartées sont décrites dans `docs/RESULTATS.md` et `docs/journaux/`. Leur code
reste dans l'historique git.
