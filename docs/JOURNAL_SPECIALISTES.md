# Journal — spécialistes NPWD2551 / NPWD2451 (11 novembre 2026)

Demande : diviser le RMSE par deux en réglant NPWD2551 et NPWD2451, sans dégrader les 31
autres circuits. Référence : modèle livré `gru_gbm` (0.7 GRU + 0.3 boosting), prédictions de
validation `runs/p3_val_g04` et `runs/p3_val_g59`. Branche locale `ensemble-gbm`, rien n'est
poussé. Entraînements de GRU sur T4 Kaggle ; boostings sur 2 circuits en local (environ 70 s).

## Plafond (`outils/plafond_circuits.py`)

```
python outils/plafond_circuits.py --run runs/p3_val_g04
```

| | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| Modèle livré (graines 0–4) | 0.0780 | 0.0810 | 0.0795 |
| 2551 et 2451 parfaits (erreur 0) | 0.0501 | 0.0620 | 0.0560 |

Une prédiction parfaite des deux circuits donne −30 %, pas −50 %. Diviser le RMSE par deux
demanderait aussi de gagner sur les autres circuits.

Les deux circuits sont des réchauffeurs à thermostat tout-ou-rien : créneaux allumés de 5 à
20 min dans `power.csv` (pas de 32 s). La moyenne horaire dépend de la phase des cycles, et
aucune entrée ne donne cette phase.

| Circuit, direction | RMSE livré | Bruit horaire (lag 1 / √2) | Lissage oracle 12 h | Niveau hebdomadaire oracle |
|---|---:|---:|---:|---:|
| 2551 directe | 0.262 | 0.231 | 0.276 | 0.230 |
| 2451 directe | 0.222 | 0.221 | 0.220 | 0.200 |
| 2551 inverse | 0.207 | 0.232 | 0.261 | 0.173 |
| 2451 inverse | 0.217 | 0.213 | 0.208 | 0.189 |

« Niveau hebdomadaire oracle » : la prédiction livrée dont on corrige la moyenne de chaque
semaine avec la vraie moyenne. Même avec ce niveau parfait, 2551 resterait à 0.23 en directe.

- Le régime long de 2551 (environ 0.04 hors saison, 0.5 en saison) suit les saisons
  d'éclipses : corrélation journalière de 0.74 avec `ltdata_eclipseduration_min`. 2451 monte
  en même temps. Le GRU suit déjà ces bascules semaine par semaine. Il garde une fuite de 0.1
  à 0.2 hors saison et quelques bascules en retard.
- Dans une orbite, le pic suit l'éclipse (corrélation 0.37 avec les minutes d'ombre de
  l'heure). Le résidu du modèle ne garde plus de corrélation avec l'ombre décalée de −3 à
  +7 h (|r| ≤ 0.15) : le GRU a déjà pris ce signal.
- Moyenner les graines 0–4 et 5–9 (10 GRU) ne change rien (2551 : 0.2610 contre
  0.2619 / 0.2605).

## Boosting spécialiste (`baselines/gbm_sacs.py`, `outils/specialistes.py`)

Nouvelles options : `circuits` (seules les colonnes visées), `lissage_cible` (moyenne centrée
de la cible à l'entraînement), ajout `orbite` (harmoniques 2–4 de la phase orbitale,
moyennes exponentielles 336 et 720 h du flux par face et de l'ombre). Base : B, V2, planning,
ema, comme le boosting livré (prédictions identiques sur les 2 colonnes).

```
python outils/specialistes.py entrainer --nom sp_base --options '{"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"]}'
```

| Variante (seule) | 2551 directe | 2451 directe | 2551 inverse | 2451 inverse |
|---|---:|---:|---:|---:|
| base (= boosting livré) | 0.3105 | 0.2390 | 0.2306 | 0.2367 |
| lissage 3 h | 0.3168 | 0.2446 | 0.2703 | 0.2366 |
| lissage 6 h | 0.3417 | 0.2555 | 0.2999 | 0.2455 |
| + orbite | 0.2990 | 0.2394 | 0.2320 | 0.2366 |
| réglage A | 0.2964 | 0.2424 | 0.2300 | 0.2409 |

Le lissage de la cible dégrade : la bande rapide retirée portait du signal réel (pic après
éclipse). Seul, le boosting reste loin du GRU sur ces circuits (0.30 contre 0.26 en directe).

Mélange par colonne (poids appris sur une direction, appliqués à l'autre ;
`outils/specialistes.py melanger`) : ajouter un seul boosting au modèle livré ne change rien
(poids optimal 0 dans les deux directions). Livré + GRU seul + orbite + A : critère rempli,
mais le gain est de 0.0015 au mieux (2551 directe 0.2619 → 0.2603) et presque nul ailleurs.

## GRU pondéré (`dio/monde.py`, T4 Kaggle)

Nouvelles options : `poids_circuits` (poids de colonnes dans la MSE, moyenne ramenée à 1) et
`lissage_cible` (moyenne centrée de certaines colonnes, entraînement seul). Configuration du
GRU livré par ailleurs, graines 0–4, 1030 s d'entraînement par run (2 directions en parallèle).
Prédictions refaites en local.

```
python outils/lancer_kaggle.py --fente 1 --sans-attendre -- outils/valider.py --nom sp_gru_p4 \
  --modele gru_mixte --options '{"poids_circuits": {"NPWD2551": 4, "NPWD2451": 4}}'
python outils/lancer_kaggle.py --fente 2 --sans-attendre -- outils/valider.py --nom sp_gru_p4l3 \
  --modele gru_mixte --options '{"poids_circuits": {"NPWD2551": 4, "NPWD2451": 4}, "lissage_cible": {"NPWD2551": 3, "NPWD2451": 3}}'
```

| GRU seul, graines 0–4 | 2551 directe | 2451 directe | 2551 inverse | 2451 inverse | Total directe / inverse |
|---|---:|---:|---:|---:|---:|
| GRU livré (`p1_gru_g04`) | 0.2644 | 0.2229 | 0.2086 | 0.2176 | 0.0791 / 0.0836 |
| poids ×4 | 0.2650 | 0.2244 | 0.2137 | 0.2213 | 0.0802 / 0.0854 |
| poids ×4 + lissage 3 h | 0.2902 | 0.2280 | 0.2532 | 0.2281 | 0.0837 / 0.0891 |

Le poids ×4 dégrade ses propres circuits. Le modèle les surapprend sur l'année d'entraînement,
comme le spécialiste de l'itération 4. Le lissage dégrade fortement, comme pour le boosting.

## Mélange par colonne et décision

| Composants ajoutés au modèle livré | Critère | Meilleur gain d'un circuit (poids transférés) |
|---|---|---|
| boosting de base, + orbite, réglage A, lissage 3 h (un à la fois) | non (poids optimal 0) | 0 |
| GRU pondéré ×4 | non (2551 inverse +0.0010) | 2551 directe −0.0008 |
| GRU seul + orbite + A | **oui** | 2551 directe −0.0016 |
| GRU seul + GRU ×4 + orbite + A | non | – |

Effet du seul mélange accepté sur le RMSE total : directe 0.07797 → 0.07780 et 0.07777 →
0.07759, inverse 0.08104 → 0.08098 et 0.08144 → 0.08143. Gain moyen : −0.00012 (−0.15 %).

**Décision : rien n'est installé, `poids/` est inchangé.** Le critère fixé avant la mesure
est rempli, mais le gain (−0.00012) est plus petit que l'optimisme de sélection d'une grille
de 286 poids par circuit choisis sur ces mêmes validations. Il ajouterait aussi deux boostings
et un aiguillage par colonne au modèle livré. Le plan ne fixait pas de gain minimal : ce seuil
est ajouté ici, après la mesure, et il est dit comme tel.

## Bilan

- Le RMSE ne peut pas être divisé par deux à partir de ces deux circuits : le plafond, avec
  2551 et 2451 parfaits, est 0.0560 (−30 %).
- Le modèle livré est déjà à leur plafond d'information. Tous les leviers sont restés à moins
  de 0.002 par circuit : variables d'orbite et d'inertie thermique, réglage, cible débruitée,
  poids dans la perte, 10 graines, mélange par colonne. Même chose qu'en hors bloc dans une
  même année (2551 : 0.272, section « Hybride » de `RESULTATS.md`). L'erreur restante est
  surtout la phase des cycles du thermostat, que rien dans les entrées ne donne.
- Marge restante ailleurs : NPWD2531 et 2532 en inverse (erreurs de régime lent).
- Coût : Kaggle 2 sessions T4 d'environ 30 min ; boostings locaux 6 × 70 s.
- Code gardé, coupé par défaut, pour reproduire ce journal : options `circuits`,
  `lissage_cible` et ajout `orbite` de `baselines/gbm_sacs.py` ; `poids_circuits` et
  `lissage_cible` de `dio/monde.py` ; `outils/plafond_circuits.py`, `outils/specialistes.py` ;
  test `test_lissage_cible_garde_les_trous_et_les_autres_colonnes`. Suite complète : 68 tests, OK.

# NPWD2531 et NPWD2532 (11 novembre 2026, suite)

Demande : traiter la plus grande marge restante, NPWD2531 et NPWD2532, mêmes règles.

## Plafond

| | Directe | Inverse |
|---|---:|---:|
| Modèle livré (graines 0–4) | 0.0780 | 0.0810 |
| 2531 et 2532 parfaits | 0.0724 | 0.0667 |

L'erreur de 2531 est surtout lente : 67 % (directe) et 84 % (inverse) de sa MSE tient dans les
moyennes hebdomadaires. 2531 a trois modes : éteint (0.001, 78 % des heures), un palier
d'impulsions de moyenne 0.1, et des campagnes à 0.6 (18 % d'annee1, 1 % d'annee2).
- Le palier 0.1 est marqué par les commandes ATTTF410/420 dans les deux années. C'est une suite
  d'impulsions : médiane 0.001, moyenne 0.09 à 0.11 sur les heures marquées.
- Les campagnes à 0.6 n'ont aucun marqueur commun aux deux années (ATMB045 n'existe qu'en annee1).
- En directe, le modèle appris sur annee1 prédit un faux palier de 0.1 partout (la moyenne des
  campagnes). En inverse, il fait décroître la campagne initiale (0.57) alors qu'elle tient
  20 semaines.

## Essais (`outils/specialistes.py`, prédictions de validation locales)

| Essai | 2531 directe | 2531 inverse | 2532 directe | 2532 inverse |
|---|---:|---:|---:|---:|
| Modèle livré | 0.1118 | 0.1850 | 0.1228 | 0.1888 |
| Boosting seul (livré) | 0.1067 | 0.1815 | 0.1263 | 0.1760 |
| Boosting + fenêtres du planning ±24/168 h (`regimes`) | 0.1066 | 0.1817 | 0.1263 | 0.1755 |
| Boosting à perte absolue (médiane) | 0.0881 | 0.1790 | 0.1279 | 0.1919 |
| Liste de marqueurs seule (`baselines/marqueurs.py`) | 0.0823 | 0.1822 | – | – |
| **Porte : ensemble près d'un marqueur (±168 h), médiane recalée ailleurs** | **0.0825** | **0.1740** | – | – |
| **0.6 boosting + 0.4 GRU physique (H1a)** | – | – | **0.1205** | **0.1714** |

- La perte absolue efface le palier 0.1 en inverse (médiane d'impulsions = 0) : la seconde
  moitié de l'inverse monte de +0.025. La liste seule fait de même dans le palier, car un niveau
  plat ne suit pas l'heure des impulsions.
- La porte garde l'ensemble là où un marqueur est proche, et met la médiane recalée ailleurs.
  Elle baisse les 4 cas et les 8 moitiés de période. Une porte à ±24 h laissait l'inverse
  monter de +0.0022 sur une moitié (graines 0–4) ; ±168 h, l'autre fenêtre déjà définie, est
  retenue. Ce choix est fait sur les validations.
- Marqueurs appris automatiquement (liste gloutonne, ≤ 3 familles, gain ≥ 1 %) : ATMB045 puis
  SCMN sur annee1, ATTTF420 puis ATMB076 sur annee2, et ATMB045, SCMN, ATTTF420 pour le modèle
  final. Une famille fréquente (SCMN, 93 % des heures d'annee1) peut entrer dans la liste en
  couvrant une partie des campagnes. Le test synthétique le montre aussi. Effet : la porte
  garde alors l'ensemble plus souvent, ce qui revient au modèle livré.
- 2532 : poids appris sur une direction et appliqués à l'autre (critère rempli, toutes les
  moitiés baissent), puis poids communs aux deux validations. Le GRU physique (H1a) a été
  réentraîné sur T4 Kaggle : il reproduit exactement les scores d'H1a (0.085326 / 0.087586).
  Ses poids de validation n'existent que pour les graines 0–4, utilisées pour les deux groupes.

```
python outils/lancer_kaggle.py --fente 1 --sans-attendre -- outils/valider.py --nom sp_phys --modele gru_mixte \
  --options '{"physique": true, "familles": false, "facteur_niveau_prediction": 1.0}'
python outils/lancer_kaggle.py --fente 2 --sans-attendre -- train.py --modele gru_mixte --out runs/sp_phys_final \
  --options '{"physique": true, "familles": false, "facteur_niveau_prediction": 1.0}'
```

## Validation de bout en bout (`predict.py` → `evaluate.py`, `runs/sp31_val_g04`, `runs/sp31_val_g59`)

| | Directe 0–4 | Directe 5–9 | Inverse 0–4 | Inverse 5–9 | Moyenne des 4 |
|---|---:|---:|---:|---:|---:|
| Modèle livré | 0.077975 | 0.077769 | 0.081042 | 0.081435 | 0.079555 |
| **+ spécialistes 2531 / 2532** | **0.076748** | **0.076533** | **0.079111** | **0.079497** | **0.077972** |

Les 8 moitiés de période baissent (directe 0.0692 → 0.0668 et 0.0859 → 0.0855 ; inverse 0.0850
→ 0.0824 et 0.0769 → 0.0757, graines 0–4). Seules les colonnes 2531 et 2532 changent : les 31
autres sont identiques au bit près (vérifié, et testé dans `tests/test_ensemble.py`).

## Installation

`poids/specialiste/` : `physique/` (GRU physique annee1 + annee2, T4 Kaggle, 7 Mo),
`marqueurs.json`, `specialistes.json`. Modèle livré dans `poids/` inchangé par ailleurs.
Prédiction : 53 s sur CPU local au lieu de 18 s (un ensemble de GRU de plus). Retour à
l'ensemble précédent : `--options '{"specialistes": false}'`. Reproduction :
`python train.py --options '{"gru_depuis": "poids", "physique_depuis": "runs/sp_phys_final"}'`
(boosting réentraîné, environ 6 min), ou `python train.py` complet sur GPU.
Tests : 70, OK (`tests/test_marqueurs.py`, test des colonnes inchangées dans `tests/test_ensemble.py`).

Limites :
- Le poids de 2532 (0.6 / 0.4) et la fenêtre de la porte (±168 h) sont choisis sur les deux
  validations : un optimisme de sélection reste.
- La porte suppose qu'une campagne de 2531 sans marqueur ne se voit que dans power_init. Une
  campagne qui démarre en cours de période sans marqueur connu serait prédite à 0.
