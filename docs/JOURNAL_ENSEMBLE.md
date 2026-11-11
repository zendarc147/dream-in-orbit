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

### RMSE par circuit, GRU seul → ensemble (validation, `runs/p3_circuits.txt`)

| Circuit | Directe 0–4 | Directe 5–9 | Inverse 0–4 | Inverse 5–9 | Δ moyen |
|---|---|---|---|---|---:|
| NPWD2532 | 0.1275→0.1228 | 0.1274→0.1227 | 0.2051→0.1888 | 0.2029→0.1874 | -0.0103 |
| NPWD2531 | 0.1215→0.1118 | 0.1219→0.1120 | 0.1904→0.1850 | 0.1930→0.1866 | -0.0078 |
| NPWD2742 | 0.0522→0.0483 | 0.0516→0.0479 | 0.0474→0.0444 | 0.0477→0.0447 | -0.0034 |
| NPWD2881 | 0.0971→0.0955 | 0.0966→0.0951 | 0.1160→0.1129 | 0.1169→0.1134 | -0.0024 |
| NPWD2551 | 0.2644→0.2619 | 0.2619→0.2605 | 0.2086→0.2066 | 0.2111→0.2073 | -0.0024 |
| NPWD2851 | 0.1382→0.1398 | 0.1376→0.1394 | 0.1466→0.1411 | 0.1489→0.1427 | -0.0021 |
| NPWD2561 | 0.0931→0.0949 | 0.0929→0.0946 | 0.1040→0.0982 | 0.1039→0.0980 | -0.0020 |
| NPWD2791 | 0.0660→0.0649 | 0.0657→0.0646 | 0.0640→0.0617 | 0.0637→0.0615 | -0.0017 |
| NPWD2372 | 0.0309→0.0302 | 0.0313→0.0304 | 0.0309→0.0283 | 0.0306→0.0280 | -0.0017 |
| NPWD2451 | 0.2229→0.2217 | 0.2226→0.2216 | 0.2176→0.2174 | 0.2213→0.2202 | -0.0009 |
| NPWD2721 | 0.0656→0.0650 | 0.0654→0.0648 | 0.0618→0.0609 | 0.0617→0.0608 | -0.0007 |
| NPWD2562 | 0.0091→0.0086 | 0.0093→0.0087 | 0.0066→0.0058 | 0.0068→0.0060 | -0.0007 |
| NPWD2802 | 0.0604→0.0597 | 0.0605→0.0597 | 0.0580→0.0574 | 0.0578→0.0572 | -0.0007 |
| NPWD2501 | 0.0020→0.0014 | 0.0019→0.0013 | 0.0020→0.0014 | 0.0018→0.0013 | -0.0006 |
| NPWD2491 | 0.0413→0.0407 | 0.0414→0.0408 | 0.0280→0.0274 | 0.0284→0.0278 | -0.0006 |
| NPWD2821 | 0.0019→0.0013 | 0.0017→0.0012 | 0.0020→0.0014 | 0.0020→0.0014 | -0.0006 |
| NPWD2552 | 0.0017→0.0012 | 0.0017→0.0012 | 0.0019→0.0013 | 0.0018→0.0013 | -0.0005 |
| NPWD2882 | 0.0017→0.0012 | 0.0017→0.0012 | 0.0017→0.0012 | 0.0017→0.0012 | -0.0005 |
| NPWD2692 | 0.0015→0.0011 | 0.0017→0.0012 | 0.0019→0.0013 | 0.0016→0.0011 | -0.0005 |
| NPWD2801 | 0.0018→0.0013 | 0.0014→0.0010 | 0.0017→0.0012 | 0.0015→0.0010 | -0.0005 |
| NPWD2472 | 0.0016→0.0011 | 0.0012→0.0010 | 0.0017→0.0012 | 0.0024→0.0017 | -0.0005 |
| NPWD2691 | 0.0016→0.0011 | 0.0012→0.0009 | 0.0018→0.0012 | 0.0015→0.0011 | -0.0005 |
| NPWD2792 | 0.0012→0.0009 | 0.0013→0.0009 | 0.0021→0.0015 | 0.0021→0.0015 | -0.0005 |
| NPWD2402 | 0.0027→0.0023 | 0.0026→0.0023 | 0.0019→0.0014 | 0.0022→0.0016 | -0.0005 |
| NPWD2871 | 0.0012→0.0008 | 0.0019→0.0014 | 0.0014→0.0010 | 0.0014→0.0010 | -0.0004 |
| NPWD2852 | 0.0009→0.0007 | 0.0014→0.0010 | 0.0017→0.0012 | 0.0019→0.0013 | -0.0004 |
| NPWD2872 | 0.0016→0.0011 | 0.0012→0.0008 | 0.0016→0.0011 | 0.0014→0.0010 | -0.0004 |
| NPWD2471 | 0.0012→0.0009 | 0.0012→0.0009 | 0.0016→0.0011 | 0.0016→0.0011 | -0.0004 |
| NPWD2771 | 0.0139→0.0137 | 0.0138→0.0137 | 0.0075→0.0073 | 0.0076→0.0074 | -0.0002 |
| NPWD2481 | 0.0079→0.0078 | 0.0078→0.0077 | 0.0100→0.0098 | 0.0101→0.0099 | -0.0001 |
| NPWD2722 | 0.0072→0.0072 | 0.0072→0.0072 | 0.0025→0.0024 | 0.0027→0.0025 | -0.0001 |
| NPWD2482 | 0.0133→0.0133 | 0.0133→0.0133 | 0.0039→0.0052 | 0.0044→0.0056 | +0.0006 |
| NPWD2401 | 0.0142→0.0141 | 0.0141→0.0141 | 0.0042→0.0059 | 0.0052→0.0067 | +0.0008 |

Régressions : NPWD2851 (+0.0016 / +0.0018) et NPWD2561 (+0.0018 / +0.0017) en directe, alors
qu'ils gagnent en inverse ; NPWD2482 (+0.0013 / +0.0012) et NPWD2401 (+0.0017 / +0.0015) en
inverse, circuits presque constants dont le boosting rate le niveau d'annee1. Plus forts gains :
NPWD2532 (−0.010 en moyenne, −0.016 en inverse), 2531 (−0.008), 2742, 2881, 2551.

## Phase 4 — GRU autorégressif (11 nov., 01 h 30 puis 09 h 00)

Module `dio/autoregressif.py` (modèle `gru_ar`, non livré) : le simulateur reçoit les 33
courants de l'heure précédente, vérité ou propre prédiction ; scheduled sampling (probabilité
de la vérité 1 → 0 sur la première moitié des 600 pas) ; flux continus (32 flux qui avancent par
morceaux de 720 h, état et dernière sortie conservés, redémarrage aléatoire au bout de la
période) ; tête résiduelle (variation horaire) ; ni branche de commandes ni correction de
niveau. Prédiction : simulation libre complète depuis la dernière heure de `power_init`.
Témoin non autorégressif aux mêmes variables (`gru`, v2r + familles, sans commandes AOOO/ATTT,
sans bruit ni masquage, facteur de niveau 1) pour isoler l'effet du retour des sorties.

Incident : la session s'est arrêtée vers 01 h 35 pendant les deux premiers entraînements
locaux ; leurs processus sont morts au pas 0, sans résultat. À 09 h 00, consigne de
l'utilisateur : tous les entraînements sur Kaggle. Runs locaux arrêtés et supprimés ;
relance avec `outils/lancer_kaggle.py` :

```
python outils/lancer_kaggle.py --cpu --fente 1 --sans-attendre -- outils/valider.py --nom p4_ar --modele gru_ar
python outils/lancer_kaggle.py --fente 2 --sans-attendre -- outils/valider.py --nom p4_temoin --modele gru \
  --options '{"variables": "v2r", "bruit": 0.0, "commandes": null, "branche_commandes": null,
              "masque_saaf": 0.0, "abandon_etats": 0.0, "facteur_niveau_prediction": 1.0}'
```

Les prédictions des poids rapatriés seront refaites en local (moins de 2 min) pour comparer au
GRU de référence sur les mêmes dossiers de validation.

Relance des graines 5–9 sur GPU T4 (`dio/autoregressif.py` entraîne sur CUDA quand il est
disponible ; consigne de l'utilisateur à 09 h 55 : GPU T4) ; les graines 0–4, déjà lancées en
session CPU, sont laissées finir :

```
python outils/lancer_kaggle.py --fente 2 --sans-attendre -- outils/valider.py --nom p4_ar_g59 \
  --modele gru_ar --options '{"graines": [5, 6, 7, 8, 9]}'
```

Coût : témoin 7 min 40 (T4, 2 directions en parallèle) ; AR graines 5–9 47 min (T4, 2 graines à
la fois par direction, boucle heure par heure : le GPU accélère peu). Prédictions refaites en local.

| Modèle (simulation libre complète) | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| GRU mixte actuel, graines 5–9 | 0.078719 | 0.084229 | 0.081474 |
| Témoin non autorégressif (v2r + familles, 1 membre, graines 0–4) | 0.080236 | 0.084501 | 0.082369 |
| **AR, graines 5–9** | **0.084643** | **0.090977** | **0.087810** |

RMSE par horizon (jours 0–7, 7–30, 30–90, 90–180, 180–365, > 365) :

| | Directe | Inverse |
|---|---|---|
| GRU actuel 5–9 | 0.043 0.043 0.074 0.071 0.072 0.088 | 0.059 0.075 0.095 0.102 0.082 0.079 |
| Témoin | 0.043 0.046 0.073 0.072 0.074 0.090 | 0.062 0.082 0.098 0.102 0.081 0.079 |
| AR 5–9 | 0.043 0.044 0.076 0.082 0.078 0.093 | 0.123 0.131 0.119 0.114 0.080 0.078 |

- L'AR ne dérive pas sur le long terme : au-delà de 180 jours en inverse il égale ou bat
  légèrement le GRU (0.0798 contre 0.0816, 0.0783 contre 0.0788). Le scheduled sampling et les
  flux continus ont donc rendu la simulation libre stable sur 680 jours.
- Il perd au début de l'inverse (30 premiers jours 0.13 contre 0.06–0.07) : presque tout vient de
  NPWD2531 (contribution MSE 104e-4 contre 6e-4), dont le niveau initial élevé (0.57) est mal
  propagé par le retour des sorties. En directe, il perd surtout entre 90 et 365 jours.
- La RMSE d'entraînement lissée finit vers 0.028 en simulation libre dans les flux
  d'entraînement, contre 0.085–0.091 en validation : l'AR apprend très bien son année et
  généralise moins bien que le GRU sans retour.
- Ablation de fusion (`outils/melanger.py --gru runs/p3_val_g59 --autre runs/p4_ar_g59`) :
  ensemble livré + AR, poids 0.1 → 0.07960 → 0.07958 de moyenne (−0.00002). Pas d'apport.

**Décision : AR rejeté, non fusionné.** Aucun réglage supplémentaire tenté (règle d'arrêt :
pas de recherche de paramètres sur une variante qui perd de 0.006 en moyenne).

## Bilan (11 nov.)

### Avant / après (validation, `evaluate.py`, prédictions locales)

| Modèle | Directe 0–4 | Directe 5–9 | Inverse 0–4 | Inverse 5–9 | Moyenne des 4 |
|---|---:|---:|---:|---:|---:|
| Avant : GRU mixte installé le 10 nov. | 0.079062 | 0.078719 | 0.083579 | 0.084229 | 0.081397 |
| **Après : ensemble 0.7 GRU + 0.3 boosting enrichi (`gru_gbm`)** | **0.077975** | **0.077769** | **0.081042** | **0.081435** | **0.079555** |
| Écart | −0.001087 | −0.000950 | −0.002537 | −0.002794 | −0.001842 |

### Contribution de chaque changement (moyenne des 4 RMSE de mélange)

| Étape | Moyenne des 4 | Δ |
|---|---:|---:|
| GRU seul | 0.08140 | — |
| + ancien boosting B_V2, poids 0.2 | 0.08051 | −0.00089 |
| + commandes planifiées (planning), meilleur poids | 0.07976 | −0.00075 |
| + moyennes exponentielles (ema), meilleur poids 0.4 | 0.07946 | −0.00030 |
| poids fixé à 0.3 (robustesse, voir phase 3) | 0.07956 | +0.00010 |

Changements testés et écartés : fenêtres passées 6–168 h (+0.00056), retrait des variables
d'horloge indirecte (+0.00044), réglage lent C (−0.00010 pour un modèle 3 fois plus lourd),
embargo 0 h (+0.00015) et 720 h (−0.00004), GRU autorégressif (seul +0.006 ; en mélange −0.00002).

### Régressions par circuit

Voir le tableau de la phase 3. Pertes : NPWD2851 et 2561 en directe (+0.0016 à +0.0018,
gagnants en inverse), NPWD2482 et 2401 en inverse (+0.0012 à +0.0017, circuits presque
constants). Tous les autres circuits gagnent dans les deux directions et les deux groupes.

### Tests exécutés

- `python -m unittest discover -s tests` : 67 tests, OK (avec les poids installés).
- `tests/test_gbm_sacs.py` (8) : sacs et embargo, familles apprises sur l'entraînement, récence,
  comptes futurs, périodes appariées, famille absente du test, causalité de l'historique et des
  EMA, retrait de l'horloge, invariance par translation dans le temps, power_init hors variables.
- `tests/test_ensemble.py` (3) : format, heures, 33 colonnes, valeurs finies et positives,
  reproductibilité, mélange exact des deux composants, retour au GRU seul, power_init qui empiète
  refusé, circuit absent et jour manquant dans power_init.
- Entraînement final répété deux fois : prédictions identiques au bit près.

### Fichiers et commits locaux (branche `ensemble-gbm`, rien de poussé)

- `baselines/gbm_sacs.py` : ajouts planning / historique / ema, sans_horloge, réglage C,
  compression lzma.
- `dio/ensemble.py` (modèle `gru_gbm`, `modeles.FINAL`), `dio/modeles.py`,
  `dio/autoregressif.py` (étude, non livrée).
- `outils/ablations_gbm.sh`, `outils/resume_ablations.py` ; `outils/ablations.py` (options du
  world model seules).
- `tests/test_gbm_sacs.py`, `tests/test_ensemble.py`.
- `poids/modele.json`, `poids/ensemble.json`, `poids/gbm/gbm_sacs.pkl.xz` (membres GRU inchangés).
- `README.md`, `docs/RESULTATS.md`, ce journal.
- Commits : `220b186`, `f56b72a`, `9c495c5`, `41c84b9` (+ ce bilan).
- Prédictions de chaque étape (non versionnées, `runs/` est ignoré par git) : `runs/p1_*`,
  `runs/p2_*` et `runs/p2_*_mel`, `runs/p3_val_*`, `runs/p3_final*`, `runs/p4_*`.

### Coûts

- CPU local (16 cœurs), avant la consigne Kaggle : ~1 h 30 d'entraînements de boosting
  (9 variantes × 2 directions, 2 entraînements finaux), prédictions et mélanges quelques minutes.
- Kaggle : témoin 8 min (T4), AR graines 5–9 47 min (T4), AR graines 0–4 en session CPU.
- Modèle livré : poids 8.2 Mo (GRU) + 16.0 Mo (boosting) ; prédiction 18–19 s et 1.44 Go au pic
  (GRU seul : 11 s, 0.89 Go) ; réentraînement du boosting 5 min 40 sur 16 cœurs.

### Limites

- Le poids 0.3 et les familles retenues sont choisis sur les deux validations : un léger
  optimisme de sélection reste (4 décisions de familles, un scalaire), atténué par deux groupes
  de graines et le fait que les gains se reproduisent dans les deux directions.
- Le gain vient surtout de l'inverse (−0.0025 à −0.0028) ; la directe gagne −0.0010.
- Le vocabulaire des commandes planifiées compte toutes les familles vues ≥ 20 fois à
  l'entraînement : sur une année au planning très différent, le gain peut baisser (les familles
  inconnues sont ignorées, pas d'erreur).
- Les modèles de validation ont un an d'entraînement ; le modèle livré en a deux. Le poids 0.3 n'a
  pas été vérifié pour ce cas.
- Le sujet demande un réseau de neurones comme modèle final : le GRU garde 70 % du mélange et
  le retour au GRU seul est une option de `predict.py`.
- `python train.py` sans option réentraîne aussi les world models : il faut un GPU (sur CPU,
  plusieurs heures), et la reproduction au bit près des membres GRU n'a été vérifiée que sur T4.
- Écart Kaggle / local sur `validation_inverse` (dernière heure partielle de power_init) : toutes
  les comparaisons de ce journal utilisent des prédictions locales.

### Reproduire le modèle livré

```bash
# Boosting seul, world models installés réutilisés : reproduit exactement poids/ (~6 min, CPU)
python train.py --options '{"gru_depuis": "poids"}'
# Tout depuis zéro (world models + boosting), sur GPU T4 Kaggle
python outils/lancer_kaggle.py --recuperer poids -- train.py
# Prédire, avec ou sans le boosting
python predict.py --test_dir DOSSIER --out predictions.csv
python predict.py --test_dir DOSSIER --out predictions.csv --options '{"poids_gbm": 0}'
```

**Mise à jour 11 h 30.** AR graines 0–4 (session CPU Kaggle, 2 h 20 sans fin) abandonné à la
demande de l'utilisateur : rapatriement arrêté, sorties ignorées. Le rejet de l'AR repose donc sur
un seul groupe de graines (5–9), avec un écart de +0.006 et aucun apport en mélange.
