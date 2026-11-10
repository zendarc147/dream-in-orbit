# Journal — ensemble GRU + GBM (11–12 novembre 2026)

Programme : mesurer si un mélange GRU + boosting à sacs temporels bat le modèle installé
(`poids/`, GRU mixte avec familles de commandes et facteur de niveau 1.5), enrichir le
boosting par ablations séparées, et l'intégrer à `train.py` / `predict.py` seulement si les
deux directions et les deux groupes de graines confirment le gain. Branche locale
`ensemble-gbm`, rien n'est poussé. Heures en KST.

## Hypothèses de travail (prises sans consultation)

- **Calcul local.** Consigne de la session : travailler uniquement dans le dépôt et ne rien
  pousser vers un dépôt distant. `outils/lancer_kaggle.py` envoie le code et `data/` vers
  Kaggle : je ne l'utilise pas. Tout tourne sur le CPU local (16 cœurs, pas de GPU). Les
  boostings tiennent en quelques minutes ; un réentraînement de GRU (phase 4) coûterait des
  heures et sera dimensionné en conséquence.
- **Référence = prédictions locales.** Les prédictions GRU de validation figées
  (`runs/gen_h1b_k15_g04`, `runs/gen_h1b_k15_g59`) viennent de Kaggle, où `power.csv` est
  reconstruit à partir de `data/prepared/*/y.csv`. En local, `validation_inverse/` est
  construit à partir du `power.csv` minute par minute, comme la procédure officielle. Seule
  la dernière heure (partielle) de `power_init` diffère en inverse ; la directe est
  identique au bit près (écart max 2.4e-7). Je fige donc les re-prédictions locales des
  mêmes poids comme référence ; toutes les autres prédictions sont aussi calculées en local.
- **Sujet.** `docs/RESULTATS.md` rappelle que le sujet veut un réseau de neurones comme
  modèle final, les modèles classiques n'étant que des composants. La demande de cette
  session est d'intégrer le mélange : le GRU reste majoritaire (poids du boosting ≤ 0.5) et
  le retour au GRU seul reste possible par une option.
- **Règle de sélection des ablations (fixée avant mesure).** Une famille de variables ou un
  réglage est gardé si (a) le boosting seul baisse dans les deux directions sans dégrader le
  mélange (moyenne des 4 RMSE au meilleur poids, pas plus de +0.0001), ou (b) le mélange
  baisse d'au moins 0.0003 (moyenne des 4 RMSE au meilleur poids de la grille 0–0.5) et
  aucune direction du mélange (moyenne des deux groupes de graines) ne monte de plus de
  0.0002. Grille de poids : 0, 0.1, …, 0.5 (0.6 seulement si le minimum est au bord).

## Phase 1 — référence et gain immédiat (11 nov., 00 h 10 – 00 h 40)

GRU final actuel : poids de validation `gen_h1b_k15` (familles + facteur 1.5), copiés dans
`runs/p1_gru_g04` et `runs/p1_gru_g59` ; `pred_fige.csv` = prédiction Kaggle d'origine,
`pred.csv` = re-prédiction locale (21–26 s par direction).

```
python outils/valider.py --nom p1_gru_g04 --sans-entrainement   # idem p1_gru_g59
```

| GRU | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| graines 0–4, Kaggle (`pred_fige.csv`) | 0.079062 | 0.083603 | 0.081333 |
| graines 0–4, local (référence) | 0.079062 | 0.083579 | 0.081320 |
| graines 5–9, Kaggle | 0.078719 | 0.084251 | 0.081485 |
| graines 5–9, local (référence) | 0.078719 | 0.084229 | 0.081474 |

Empreintes sha256 des prédictions Kaggle figées : g04 directe `8ecc3891…`, inverse
`377c788e…` ; g59 directe `04c83615…`, inverse `313dfca1…`.

GBM existant (B_V2 de la session hybride) : modèles `runs/gbm_B_V2/*/gbm_sacs.pkl` copiés
dans `runs/p1_gbm_B_V2`, prédictions recalculées en local (13 s par direction) :
**0.087305 / 0.089753**, identiques aux scores enregistrés.

```
python outils/valider.py --nom p1_gbm_B_V2 --sans-entrainement
python outils/melanger.py --gru runs/p1_gru_g04 runs/p1_gru_g59 --autre runs/p1_gbm_B_V2 --nom p1_melange
```

| Poids du boosting | 0 | 0.1 | **0.2** | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Directe, graines 0–4 | 0.07906 | 0.07833 | **0.07795** | 0.07794 | 0.07828 | 0.07897 | 0.08001 |
| Directe, graines 5–9 | 0.07872 | 0.07805 | **0.07773** | 0.07776 | 0.07815 | 0.07888 | 0.07995 |
| Inverse, graines 0–4 | 0.08358 | 0.08314 | **0.08294** | 0.08298 | 0.08326 | 0.08378 | 0.08454 |
| Inverse, graines 5–9 | 0.08423 | 0.08370 | **0.08341** | 0.08337 | 0.08358 | 0.08403 | 0.08472 |
| Moyenne | 0.08140 | 0.08080 | **0.08051** | 0.08051 | 0.08082 | 0.08142 | 0.08230 |

`evaluate.py` au poids 0.2 : 0.077955 / 0.077729 (directe, 530 409 cellules sur 16 073 heures),
0.082940 / 0.083412 (inverse, 537 471 cellules sur 16 287 heures). Heures identiques à
`heures_a_predire.csv` (vérifié par `melanger.py`). Corrélation des erreurs GRU / boosting :
0.80 en directe, 0.87 en inverse.

Moitiés de période (GRU → mélange 0.2) :

| | 1re moitié | 2e moitié |
|---|---|---|
| Directe 0–4 | 0.07100 → 0.06937 | 0.08638 → 0.08568 |
| Directe 5–9 | 0.07043 → 0.06893 | 0.08621 → 0.08563 |
| Inverse 0–4 | 0.08834 → 0.08682 | 0.07853 → **0.07887** |
| Inverse 5–9 | 0.08949 → 0.08767 | 0.07862 → **0.07892** |

Circuits (RMSE GRU / boosting → mélange, graines 0–4 puis 5–9) :
- gains en directe : NPWD2531 0.1215 → 0.1127, 2551 0.2644 → 0.2600, 2532 0.1275 → 0.1245,
  2561 0.0931 → 0.0906, 2881 0.0971 → 0.0948 ;
- gains en inverse : 2532 0.2051 → 0.1978, 2531 0.1904 → 0.1880, 2851 0.1466 → 0.1444,
  2551 0.2086 → 0.2070 ;
- pertes : 2881 en inverse (0.1160 → 0.1217), 2561 en inverse (0.1040 → 0.1059), 2491, 2742,
  2791 et 2451 de quelques millièmes.

**Décision.** Le meilleur poids est 0.2 dans chaque direction (0.3 quasi égal) ; les quatre
RMSE baissent, mais le gain moyen n'est que de 0.00089 (contre 0.0026 avec l'ancien GRU) et
la seconde moitié de l'inverse se dégrade. Le GRU avec familles a absorbé une part de ce
qu'apportait le boosting (familles de commandes). Gain jugé insuffisant pour une intégration
anticipée : passage à la phase 2 pour rendre le boosting plus complémentaire.

## Phase 2 — boosting enrichi, ablations séparées (11 nov., 00 h 40 – 02 h 30)

Code : options `ajouts` (`planning`, `historique`, `ema`) et `sans_horloge` de
`baselines/gbm_sacs.py`, réglage `C` ; tests de fuite et de causalité dans
`tests/test_gbm_sacs.py`. Chaque variante part de B_V2 (150 itérations, 15 feuilles,
≥ 200 lignes par feuille, 5 sacs, embargo 168 h) et ne change qu'une chose.

- **planning** : familles de commandes avec lettres de variante fusionnées et préfixes
  numérotés regroupés (`MOCS.0000003738` → `MOCS`), vues ≥ 20 fois à l'entraînement (119
  sur annee1) ; par famille : heures depuis la dernière commande (plafond 168 h), comptes
  dans l'heure suivante et dans les six suivantes ; part active de chaque période début/fin
  appariée par numéro (paires apprises : MOCS/MOCE, PENS/PENE ; durée ≤ 7 jours).
- **historique** : moyennes passées 6/24/72/168 h du flux éclairé par face, de l'ombre, des
  minutes de chaque mode de pointage ftl et des communications (fenêtres déjà présentes
  dans V2 non dupliquées).
- **ema** : moyennes exponentielles causales (constantes 3/12/48/168 h) des mêmes séries.
- **sans_horloge** : retire `ltdata_sunmars_km`, `ltdata_earthmars_km`,
  `ltdata_sunmarsearthangle_deg` et `ltdata_solarconstantmars`, lents et quasi monotones sur
  une année (hypothèse : le flux par face, qui garde la constante solaire, suffit).
- **C** : 600 itérations, taux 0.02, 15 feuilles, ≥ 100 lignes par feuille, L2 = 3.
- **embargo** 0 h et 720 h au lieu de 168 h entre un sac et son bloc écarté.

```
bash outils/ablations_gbm.sh p2_base '{"config": "B", "jeu": "V2"}' \
  p2_planning '{"config": "B", "jeu": "V2", "ajouts": ["planning"]}' ...   # liste complète : section commandes
python outils/resume_ablations.py --reference p2_base p2_planning p2_historique p2_ema \
  p2_sans_horloge p2_reglage_C p2_embargo0 p2_embargo720
```

La référence réentraînée en local (`p2_base`) redonne exactement les prédictions de
`p1_gbm_B_V2` (fichiers identiques) : le boosting est déterministe d'une machine à l'autre.

| Variante | Seul directe | Seul inverse | w* | Mélange directe | Mélange inverse | Moyenne des 4 | Δ | Variables (dir./inv.) | Entraînement (2 dir.) | Poids/dir. | Décision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B_V2 (réf.) | 0.087305 | 0.089753 | 0.2 | 0.07784 | 0.08318 | 0.08051 | — | 250 / 268 | 240 s | 85 Mo | — |
| + planning | 0.086944 | **0.083135** | 0.4 | 0.07825 | **0.08128** | **0.07976** | −0.00075 | 601 / 698 | 610 s | 121 Mo | **garder** (a) |
| + historique | 0.094624 | 0.090042 | 0.1 | 0.07865 | 0.08349 | 0.08107 | +0.00056 | 310 / 329 | 265 s | 96 Mo | rejeter |
| + ema | 0.086913 | 0.087760 | 0.3 | 0.07775 | 0.08223 | 0.07999 | −0.00052 | 338 / 356 | 290 s | 110 Mo | **garder** (a, b) |
| sans horloge | 0.090679 | 0.088710 | 0.2 | 0.07888 | 0.08303 | 0.08095 | +0.00044 | 246 / 264 | 150 s | 83 Mo | rejeter |
| réglage C | 0.086255 | 0.089825 | 0.3 | 0.07768 | 0.08313 | 0.08040 | −0.00010 | 250 / 268 | 530 s | 254 Mo | rejeter |
| embargo 0 h | 0.089438 | 0.089692 | 0.2 | 0.07813 | 0.08318 | 0.08066 | +0.00015 | 250 / 268 | 145 s | 85 Mo | rejeter |
| embargo 720 h | 0.087748 | 0.089233 | 0.3 | 0.07788 | 0.08307 | 0.08047 | −0.00004 | 250 / 268 | 140 s | 85 Mo | rejeter |

« Mélange directe / inverse » : moyenne des deux groupes de graines du GRU, au w* qui
minimise la moyenne des 4 RMSE (grille 0–0.5). Durées mesurées avec d'autres calculs en
parallèle sur la machine : ordres de grandeur seulement.

Lecture :
- **planning** : gain du boosting seul surtout en inverse (−0.0066). Contributions à la MSE
  (×1e-4, B_V2 → planning) : NPWD2881 7.30 → 3.61, 2561 4.97 → 2.34, 2551 16.7 → 14.5,
  2451 17.5 → 16.8 en inverse ; en directe 2881 4.23 → 2.94, 2451 17.9 → 16.8, mais 2551
  29.1 → 31.3. Les circuits gagnants sont ceux que pilotent les commandes : signe d'une
  information réelle, pas d'une fuite (aucune consommation dans les variables, vocabulaire
  appris sur l'entraînement, test d'invariance par translation dans le temps). Au même poids,
  la directe du mélange est inchangée (0.0780 contre 0.0779 à w = 0.3) et l'inverse gagne
  0.0015 ; le poids optimal diffère par direction (0.2 en directe, 0.5 en inverse).
- **ema** : gain modeste mais dans les deux directions, seul et en mélange.
- **historique** : nette perte en directe. Les moyennes 168 h des modes de pointage et des
  communications servent probablement d'empreinte de la période d'entraînement. Les EMA des
  mêmes séries ne posent pas ce problème (pondération dominée par les heures récentes).
- **sans horloge** : la perte en directe (+0.0034) montre que ces variables lentes portent une
  information physique (saison, distance au Soleil) que le flux par face ne remplace pas.
- **C** : meilleur en directe seul, pire en inverse, gain de mélange négligeable pour un
  modèle trois fois plus lourd : rejet.
- **embargo** : 0 h dégrade (les sacs voient les heures voisines de leur bloc, le boosting
  apprend des structures très locales), 720 h ne change rien : 168 h conservé.

### Combinaison des familles retenues (11 nov., 02 h 30)

```
bash outils/ablations_gbm.sh p2_planning_ema '{"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"]}'
```

| Poids du boosting | 0 | 0.1 | 0.2 | **0.3** | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Directe, graines 0–4 | 0.07906 | 0.07839 | 0.07803 | **0.07797** | 0.07823 | 0.07878 |
| Directe, graines 5–9 | 0.07872 | 0.07810 | 0.07778 | **0.07777** | 0.07806 | 0.07866 |
| Inverse, graines 0–4 | 0.08358 | 0.08252 | 0.08168 | **0.08104** | 0.08063 | 0.08043 |
| Inverse, graines 5–9 | 0.08423 | 0.08308 | 0.08215 | **0.08144** | 0.08094 | 0.08068 |
| Moyenne | 0.08140 | 0.08053 | 0.07991 | **0.07956** | 0.07946 | 0.07964 |

Boosting seul : 0.085825 / 0.082774 (B_V2 : 0.087305 / 0.089753). Les deux ajouts se
cumulent : mélange −0.00104 contre B_V2, −0.0018 à −0.0019 contre le GRU seul.

## Phase 3 — intégration (11 nov., 02 h 30 – 03 h 30)

**Poids du mélange : 0.3, fixé avant l'entraînement final.** Le minimum de la moyenne des
4 RMSE est à 0.4 (0.07946 contre 0.07956 à 0.3), mais à 0.4 la seconde moitié de la directe
se dégrade (0.08638 → 0.08655 ; 0.08621 → 0.08649), alors qu'à 0.3 les 8 moitiés de période
gagnent. À 0.3 : blocs de 30 jours gagnants 18, 17 / 23 (directe), 18, 20 / 23 (inverse), contre
16, 15, 18, 19 à 0.4 ; pire régression d'un circuit +0.0018 (NPWD2561, 2851 en directe ;
2401, 2482 en inverse, circuits presque constants) contre +0.0031 à 0.4. 0.3 est aussi le
poids optimal de la directe seule et laisse 70 % au world model. Un seul poids pour les 33
circuits, aucun poids par circuit.

Composants : world model GRU mixte installé le 10 novembre (inchangé, membres identiques au bit
près) + boosting B_V2 + planning + ema entraîné sur annee1 + annee2 (5 sacs temporels sur la
période fusionnée). Modèle `gru_gbm` (`dio/ensemble.py`), devenu `modeles.FINAL`.

Validation bout en bout par `predict.py` → `evaluate.py` (`runs/p3_val_g04`, `runs/p3_val_g59` :
GRU de validation + boosting `p2_planning_ema`, poids 0.3) :

| | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| GRU seul, graines 0–4 | 0.079062 | 0.083579 | 0.081320 |
| **Ensemble, graines 0–4** | **0.077975** | **0.081042** | **0.079509** |
| GRU seul, graines 5–9 | 0.078719 | 0.084229 | 0.081474 |
| **Ensemble, graines 5–9** | **0.077769** | **0.081435** | **0.079602** |

Entraînement final et installation :

```
python train.py --modele gru_gbm --out runs/p3_final --options '{"gru_depuis": "poids"}'
```

- 5 min 38 s sur CPU local (16 cœurs) : boosting 2 années, puis compression lzma. Second
  entraînement identique (`runs/p3_final_bis`) : prédictions identiques au bit près sur les
  deux dossiers de validation.
- Poids : GRU 8.2 Mo + boosting 16.0 Mo (`gbm/gbm_sacs.pkl.xz`, 126 Mo sans compression).
- Prédiction : 18 s et 1.44 Go au pic (directe), 19 s et 1.43 Go (inverse), contre 11 s et
  0.89 Go pour le GRU seul (mesures `resource.getrusage` des processus fils).
- `tests/test_ensemble.py` sur les poids installés : 33 colonnes + `ut_ms`, heures égales à
  `heures_a_predire.csv`, valeurs finies et positives ; deux prédictions identiques ; ensemble =
  0.7 × GRU seul + 0.3 × boosting seul (options `poids_gbm` 0 et 1) ; un `power_init` qui
  empiète sur les heures à prédire est refusé (seule consommation lue : les 7 jours) ; un
  circuit entièrement absent et un jour manquant dans `power_init` donnent des prédictions
  finies. Suite complète : 67 tests, OK.
- Critère d'installation rempli (les deux directions et les deux groupes de graines baissent) :
  candidat copié dans `poids/`. Anciens poids : `runs/p3_poids_precedents/` (copie locale) et
  commit `8a4d761` (`git checkout 8a4d761 -- poids/`) ; retour au GRU seul sans changer de
  poids : `python predict.py ... --options '{"poids_gbm": 0}'`.
