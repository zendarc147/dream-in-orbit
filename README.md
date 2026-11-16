# Dream in Orbit — world model de la consommation électrique d'un satellite

Un réseau de neurones récurrent apprend comment le satellite réagit à son planning
(commandes dmop, dont les commandes AOOO et ATTT, modes de pointage ftl) et à son contexte (Soleil saaf, événements evtf,
tendances ltdata), puis simule seul ses 33 circuits heure par heure sur toute une période,
à partir de ses 7 premiers jours réels (`power_init.csv`). Sujet : [`docs/sujet.md`](docs/sujet.md).
Scores et historique des essais : [`docs/RESULTATS.md`](docs/RESULTATS.md). Ablations : [`docs/ABLATIONS.md`](docs/ABLATIONS.md). Ensemble GRU + boosting : [`docs/JOURNAL_ENSEMBLE.md`](docs/JOURNAL_ENSEMBLE.md). Spécialistes NPWD2551 / NPWD2451 (plafond, essais non retenus) : [`docs/JOURNAL_SPECIALISTES.md`](docs/JOURNAL_SPECIALISTES.md).

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
| + mélange avec un boosting à sacs temporels enrichi (commandes planifiées, moyennes exponentielles), poids 0.3 | 0.0780 | 0.0810 | 0.0795 |
| + spécialistes : porte de régime par marqueurs pour NPWD2531, boosting + GRU physique pour NPWD2532 | 0.0767 | 0.0791 | 0.0779 |
| + boosting « recette » (familles corrigées, événements de commande, géométrie exacte, décalages, réglage avec arrêt anticipé) et GRU recette à 0.3 dans la part GRU | 0.0753 | 0.0787 | 0.0770 |
| + poids du boosting 0.5 au lieu de 0.3 ; GRU principal sans encodages par lettre de variante | 0.0751 | 0.0785 | 0.0768 |
| + boosting propre à NPWD2551 (D3L, variables d'orbite), livré jusqu'au 15 novembre (`poids_repli/`) | 0.0750 | 0.0785 | 0.0767 |
| Structure C1 : GRU recette remplacé par un GRU S borné (165 entrées au lieu de 236), moitié de la part GRU, installée le 16 novembre au matin (`poids_candidats/structure/C1_g5/`) | 0.0744 | 0.0782 | 0.0763 |
| Structure T3 : GRU S borné seul (10 graines), boosting S, recette B pour NPWD2551, sans GRU mixte (décision de l'utilisateur, 16 novembre) | 0.0746 | 0.0804 | 0.0775 |
| **Structure S2 : T3 + GRU à boucle (réinjection de sa sortie, normalisation standard, sans recalage) à moitié dans la part réseau, modèle final (décision de l'utilisateur)** | **0.0746** | **0.0805** | **0.0775** |

**Modèle final : structure S2** (`docs/JOURNAL_GRU_BOUCLE.md`). Second GRU à boucle ajouté sur décision de
l'utilisateur ; ce choix ne maximise pas la validation interne. Part réseau = 0.5 × GRU S borné (G0, 10 graines,
recalé sur power_init) + 0.5 × GRU à boucle (GB, 10 graines : il relit sa propre prédiction, scheduled sampling
jusqu'à p = 0, premier segment avec la vérité, normalisation standard, aucun recalage : les 7 jours réels ne font
que réchauffer son état). Part boosting de T3 : boosting S (recalé), recette B seule pour NPWD2551. Mêmes spécialistes
NPWD2532 et NPWD2531. **Part du réseau : 0.5 du mélange, partagée entre deux GRU** (NPWD2532 : 0.4).
S2 a été installé par une règle écrite avant mesure, qui ne le refusait qu'en cas d'échec technique. Contre T3 :
moyenne directe / inverse 0.077534 (+0.000009, égal au bruit près), moyenne des coupures glissantes 0.073709
(−0.000449), c87 −0.0021, c62 −0.0006, c50 +0.0007, c75 +0.0003. Le livré L (`poids_repli/`, 0.076735) et C1
restent meilleurs sur directe / inverse. T3 est gardé dans `poids_candidats/structure/T3_g10/`, S1 et S2 avec un
poids boosting de 0.6 dans `poids_candidats/gru_boucle/`.

Structure T3 (16 novembre, installée à la demande de l'utilisateur alors que la règle écrite avant mesure la
refusait, `docs/JOURNAL_STRUCTURE.md`) : 0.5 × GRU S borné (10 graines) + 0.5 × boosting S (recalé) ; NPWD2551 :
recette B seule dans la part boosting. Moyenne des 4 RMSE : 0.077525 (+0.0008 sur le livré L, +0.0012 sur C1) ;
moyenne des coupures glissantes 0.074158 (+0.0023 sur L), c87 +0.0084. Plus sobre (aucun GRU mixte, 3 000 arbres
pour NPWD2551 au lieu de 26 198) et environ deux fois plus rapide en prédiction.

Structure C1 (choisie par la règle, remplacée par T3) : sobriété sur décision de l'utilisateur ; elle ne maximise pas la validation interne.
C1 (16 novembre, `docs/JOURNAL_STRUCTURE.md`) : 0.5 × (0.5 GRU mixte + 0.5 GRU S borné) + 0.5 × boosting R,
mêmes spécialistes. **Part du réseau : 0.5 du mélange** (NPWD2532 : 0.4). Moyenne des 4 RMSE (directe, inverse,
graines 0–4 et 5–9) : 0.076288 contre 0.076735 pour le livré précédent. Sur la validation à origine glissante (quatre
coupures, réentraînement à chaque coupure), C1 fait mieux sur les deux premières coupures et moins bien sur les deux
plus récentes (+0.0015 et +0.0031) ; moyenne des coupures 0.072482 contre 0.071885. Il passe les garde-fous fixés
avant mesure ; les structures plus sobres (GRU S seul, boosting S, recette B) cèdent toutes sur la dernière coupure
(+0.008 à +0.009, NPWD2551 et NPWD2451 surestimés). Le livré précédent reste dans `poids_repli/`, les poids de tous
les candidats dans `poids_candidats/structure/`.

Les deux derniers changements viennent d'une enquête sur l'erreur d'une année à l'autre.
Les familles de commandes sont des comptes horaires par code dmop sans lettre de variante
(vocabulaire appris sur l'entraînement). La correction de niveau issue de power_init est
multipliée par 1.5 à la prédiction : estimé hors bloc sur l'année d'entraînement seule, ce facteur
corrige une correction apprise trop faible. Graines 5–9 : 0.0787 / 0.0843. Gain dans les deux
directions et sur plusieurs circuits ; NPWD2532 se dégrade. Détail, critères et limites dans
`docs/RESULTATS.md`, section « Généralisation d'une année à l'autre ».

Le modèle du 11 novembre était un ensemble : 0.7 × world model GRU + 0.3 × boosting à sacs temporels (depuis le 12 novembre : 0.5 × GRU, dont 30 % de GRU recette, + 0.5 × boosting recette ; depuis le 16 novembre, le GRU recette est un GRU S borné et pèse 50 % de la part GRU ; structure S2 : la part GRU est faite de deux GRU recette, le GRU S borné et un GRU à boucle, à parts égales)
(un HistGradientBoosting par circuit, 5 sacs). Le boosting, sans mémoire, reçoit en plus des
variables du GRU les commandes planifiées par famille (heures depuis la dernière, comptes dans
l'heure et les six heures suivantes, périodes début/fin appariées) et des moyennes
exponentielles du flux solaire, de l'ombre, du pointage et des communications. Ensemble, graines
0–4 : 0.0780 / 0.0810 ; graines 5–9 : 0.0778 / 0.0814 (GRU seul : 0.0791 / 0.0836 et
0.0787 / 0.0842, recalculés en local). Le GRU n'est plus majoritaire : depuis le 12 novembre,
réseaux de neurones et boosting pèsent chacun 0.5 du mélange (NPWD2532 : 0.6 boosting + 0.4 réseau).
Les réseaux restent le world model : eux seuls simulent la période pas à pas à partir de
power_init ; le boosting, sans état, corrige heure par heure. Le retour au GRU seul est une option
de prédiction. Ablations, critères et limites : `docs/JOURNAL_ENSEMBLE.md`.

Trois colonnes ont un spécialiste (`poids/specialiste/`, `docs/JOURNAL_SPECIALISTES.md`, `docs/JOURNAL_2551.md`).
NPWD2531 garde l'ensemble près des familles de commandes qui marquent ses modes (apprises sur
l'entraînement : ATMB045, SCMN, ATTTF420) ; ailleurs, il prend son niveau médian recalé sur
power_init, au lieu du faux palier qu'apprend un modèle en moyenne. NPWD2532 vaut 0.6 × boosting +
0.4 × world model avec variables physiques. Pour NPWD2551, la part boosting vient d'un boosting appris
sur ce seul circuit, plus long (taux 0.01, jusqu'à 6000 arbres) et avec les variables d'orbite :
2551 passe de 0.2592 / 0.2063 à 0.2583 / 0.2054. Les 30 autres colonnes sont inchangées. Graines 5–9 :
0.0765 / 0.0795. Retour à l'ensemble sans spécialistes : `--options '{"specialistes": false}'`.

Recette du 12 novembre (`docs/JOURNAL_RECETTE.md`) : le boosting change de variables et de réglage
(familles de commandes sans lettre de variante, changements de variante, préfixes, péricentres aux instants
exacts, décalages ±1–3 h, moyenne des cosinus solaires, réglage avec arrêt anticipé, embargo nul). Seul, il passe
de 0.0858 / 0.0828 à 0.0778 / 0.0810 et bat le GRU seul. Un second GRU, plus petit et sans retour des sorties
(`dio/recette.py`), prend 30 % de la part GRU. Le boosting passe ensuite de 0.3 à 0.5 du mélange. Le GRU principal
n'utilise plus les impulsions et états par commande exacte ni les comptes avec lettre de variante (familles corrigées seulement).
Graines 5–9 : 0.0752 / 0.0784.

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
(`modele.json`, `ensemble.json`, `membres.json`, `membre*/monde.pt`, `gbm/gbm_sacs.pkl.xz`,
`specialiste/`). Environ 55 s et 1.5 Go de mémoire sur CPU.

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
python train.py --options '{"gru_depuis": "poids", "physique_depuis": "poids/specialiste/physique"}'
```

Reproductibilité selon le matériel : l'arrêt anticipé des boostings dépend des calculs en
virgule flottante. Le boosting de NPWD2551 livré a été entraîné en session CPU (Kaggle) ; le même
réglage réentraîné sur une session T4 (Python 3.13.16, même scikit-learn 1.9.1) s'arrête à un autre
nombre d'arbres dans 2 sacs sur 5, d'où un écart de prédiction moyen de 0.0025 (max 0.025) sur ce
circuit (`docs/JOURNAL_2551.md`, complément du 15 novembre). Un réentraînement sur un autre
matériel donne donc des poids proches, pas identiques au bit près.

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
