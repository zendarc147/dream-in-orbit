# Journal — structure sobre de l'ensemble (à partir du 15 novembre 2026, soir)

Mission fournie par l'utilisateur : rendre le modèle livré plus sobre et plus robuste à une année inconnue, par un
changement de structure de l'ensemble et de règle de décision (pas de recherche de variables ni d'architecture).
Branche locale `recette-complete`, rien n'est poussé. **Tous les entraînements tournent sur Kaggle `zendarc148`
(sessions T4)**, à la demande de l'utilisateur pour cette session ; en local : assemblages, prédictions avec des poids
existants, scores, tests.

## Reprendre ici

- Sessions Kaggle : `runs/st_S{1,2,3,4}.log` / `.pid` (attente par `kill -0 PID`), commandes `runs/st_kaggle.py`,
  lanceur `runs/st_lancer.sh SESSION FENTE`.
- Statut (16 novembre 2026) : **T3 (10 graines) installé à la demande de l'utilisateur** (section « Installation de
  T3 ») ; auparavant, C1 installé par la règle (premier admissible dans l'ordre
  de sobriété) ; livré précédent dans `poids_repli/` ; poids finaux de tous les candidats dans
  `poids_candidats/structure/`.

## Constat de départ (repris des journaux, non redécouvert)

- Livré : 0.076735 (moyenne des 4 RMSE directe/inverse × graines 0–4/5–9, avec spécialistes). Structure :
  0.5 × part GRU (0.7 GRU mixte deux couches + 0.3 GRU recette G2) + 0.5 × boosting R (recalé sur power_init), puis
  spécialistes NPWD2531 (porte des marqueurs), NPWD2532 (0.6 boosting + 0.4 world model physique) et boosting propre de
  NPWD2551 (D3L + orbite).
- Directe, inverse et chronologique ont servi à plusieurs centaines de décisions ; un écart de 0.001 est de l'ordre de
  l'effet d'environnement mesuré au programme 4 (`docs/JOURNAL_2551.md`).
- « La validation interne ne prédit pas le transfert entre années » pour NPWD2551 (programme 3, limite 3).
- 61 à 76 % de la MSE de NPWD2551 est le cycle du thermostat ; reste attaquable l'erreur de niveau prolongée
  (épisodes de plusieurs semaines à −0.3 / −0.4 aux bascules du réchauffeur, régime haut hors saison).
- GRU S borné (`variables="S"`, une couche de 128, `normalisation="minmax"`) : meilleur GRU seul en directe ; C1
  (0.5 mixte + 0.5 GRU S borné) gagne sur les 4 RMSE (0.076288) et n'a été écarté que par la porte chronologique à une
  seule coupure (`docs/JOURNAL_RECETTE_COMPLETE.md`).
- Boosting sobre (S ; recette B pour NPWD2551) : un peu moins bon en interne, niveau de NPWD2551 plus haut hors saison.

## Orientation imposée (décision de l'utilisateur, pas un résultat mesuré)

Ce qui suit est une **orientation imposée** par l'utilisateur. Elle n'est pas une conclusion des validations
internes et n'est pas rejustifiée ici par de nouvelles ablations.

- La sobriété prime sur le dernier millième interne. Une prochaine année apportera un régime de chauffage que les deux
  années connues ne contiennent pas. Dans ce cas, un composant à peu d'entrées, peu d'arbres et sans correction
  apprise se dégrade moins qu'un composant affûté sur deux années. L'utilisateur accepte de perdre jusqu'à 0.0010 de
  moyenne interne pour cela.
- Le risque à réduire est la sous-estimation prolongée des trois réchauffeurs NPWD2551, NPWD2451, NPWD2851 dans un
  régime haut hors saison d'éclipses. Un modèle qui descend moins bas et moins sûrement hors saison est préféré, même
  s'il paie un biais positif en interne.
- Le petit GRU physique devient le GRU principal. Le GRU mixte deux couches (268 entrées, épisodes de 365 jours, EMA de
  poids) passe au second plan ou sort.
- Le boosting sobre entre dans la part boosting, à égalité avec R ou à sa place, y compris pour NPWD2551 (recette B à
  côté ou à la place de D3L).
- Poids fixes et ronds (0.5 / 0.5, moyenne simple). Aucun poids par circuit, aucun poids optimisé sur une validation.

## Liste fermée des candidats (fixée avant toute mesure)

Mélange global inchangé : 0.5 × part GRU + 0.5 × part boosting. Spécialistes NPWD2531 et NPWD2532 conservés partout.

| Nom | Part GRU | Boosting général | Boosting NPWD2551 |
|---|---|---|---|
| L (livré) | 0.7 mixte + 0.3 G2 | R | D3L |
| C1 (déjà mesuré) | 0.5 mixte + 0.5 GRU S borné | R | D3L |
| T1 | GRU S borné seul | R | D3L |
| T2 | GRU S borné seul | 0.5 R + 0.5 S | 0.5 D3L + 0.5 B |
| T3 | GRU S borné seul | S seul | B seule |

- GRU S borné = `gru_recette` `{"variables": "S", "reinjection": false, "normalisation": "minmax"}`, recalage 1.0 à la
  prédiction (runs `rc3_gruA_norm`, `rc4_gruA_mm_g59` en directe/inverse).
- S = `gbm_sacs` `{"config": "D", "jeu": "S", "sacs": 5, "embargo_h": 0, "filtre": false}` (`rc1_gbm_S` ;
  `"compresser": true` pour les nouveaux entraînements, ce qui ne change que le format du fichier), recalé.
- B = recette B stricte (`baselines/recette_b.py`, `PARAMS_B`, masque collectif, sans recalage ; prédictions de
  référence `runs/b3k_rec_recette_coll`, environnement principal). Pour être utilisable par `predict.py`, le module
  reçoit `entrainer` / `predire` (paquet sauvegardé) ; les fonctions de construction et d'ajustement sont inchangées.
- Variantes autorisées, et seulement celles-ci : (i) GRU S à 10 graines (0–9 moyennées) au lieu de 5 ; (ii) T3 sans
  recalage power_init sur la part boosting.

## Règle de décision (écrite avant le calcul de tout score de T1–T3)

**Bruit de référence.** Sur chaque validation : |g04 − g59| du livré complet ; effet d'environnement du programme 4
(≈ 0.001 sur la RMSE de NPWD2551, soit ≈ 0.0001 sur la RMSE globale à 33 colonnes). Bruit = le plus grand des deux.
Tout écart plus petit est déclaré « égal » (information seulement : les garde-fous ci-dessous décident).

Directe / inverse (livré, avec spécialistes) : g04 0.075046 / 0.078474, g59 0.075078 / 0.078343 ; |g04 − g59| =
0.000032 (directe), 0.000131 (inverse) ; bruit retenu 0.0001 (directe), 0.00013 (inverse). Bruit des coupures : rempli
ci-dessous après la mesure du livré, avant celle des candidats.

Bruit des coupures (livré complet réentraîné dans chaque coupure, g04 / g59) :

| Coupure | g04 | g59 | \|g04 − g59\| | Bruit retenu |
|---|---:|---:|---:|---:|
| c50 | 0.064688 | 0.064723 | 0.000035 | 0.0001 |
| c62 | 0.065886 | 0.065814 | 0.000072 | 0.0001 |
| c75 | 0.088915 | 0.089058 | 0.000143 | 0.00014 |
| c87 | 0.068197 | 0.067799 | 0.000398 | 0.0004 |

Écart à l'ordre prévu : c87 (Kaggle S2, rentrée la première) a été prédite pour tous les candidats d'un seul passage,
avant que ce tableau ne soit écrit ; c50, c62 et c75 ont été notées pour le livré avant les candidats. La règle et
l'ordre de sobriété étaient commités (`985a9bb`) avant tout score de T1–T3 ; les seuils ne dépendent pas de ce bruit,
qui n'est qu'informatif.

**Validation à origine glissante** (remplace la porte chronologique à une coupure ; `outils/faire_chrono.py
--glissant`). annee1 et annee2 sont contiguës (16 488 h chacune). Coupure k (k = 0..3) au début d'annee2 + k × 4 122 h,
soit 50, 62,5, 75 et 87,5 % de la durée totale ; entraînement = annee1 + annee2 avant la coupure ; test = les 4 122 h
suivantes, dont 7 jours de power_init en tête (3 953 heures à prédire). Vocabulaires, schémas, normalisations et bornes
sont réappris dans chaque coupure par l'entraînement lui-même (le schéma de `prepare.py` reste appris sur annee1,
toujours incluse dans l'entraînement).

| Coupure | Entraînement | Test (ut_ms) | Remarque |
|---|---|---|---|
| c50 | annee1 | 1278720000000 → 1293559200000 | mêmes modèles que la directe |
| c62 | annee1 + 25 % d'annee2 | 1293559200000 → 1308398400000 | |
| c75 | annee1 + 50 % d'annee2 | 1308398400000 → 1323237600000 | entraînement = celui de la chrono, fichier à fichier |
| c87 | annee1 + 75 % d'annee2 | 1323237600000 → 1338076800000 | contient février–mars 2012 |

Graines : sur chaque validation, le livré et C1 sont notés avec leurs GRU des graines 0–4 (g04) et 5–9 (g59) ; le
score d'une coupure est la moyenne des deux groupes. Pour T1–T3 à 5 graines, GRU S 0–4 avec g04 et 5–9 avec g59. GRU S
à 10 graines : paquets 0–4 et 5–9 fusionnés (configuration, variables et statistiques vérifiées identiques), un seul
score par validation, comparé à chaque groupe du livré pour les 4 RMSE temporelles et à la moyenne des groupes du
livré sur les coupures.

**Garde-fous, pas concours.** Un candidat est admissible si, de bout en bout (`predict.py` → `evaluate.py`, avec
spécialistes) :
- moyenne des 4 RMSE temporelles ≤ livré + 0.0010 ;
- aucune des 4 RMSE ne monte de plus de 0.0020 ;
- moyenne des coupures glissantes ≤ livré + 0.0020, et aucune coupure ne monte de plus de 0.0040 ;
- aucune sortie non finie ; dérive rapportée (RMSE par quart d'horizon).

**Choix par ordre de sobriété, pas par score.** Ordre fixé avant mesure, variantes comprises : T3 sans recalage
(10 graines, puis 5), T3 (10, puis 5), T2 (10, puis 5), T1 (10, puis 5), C1. Le premier admissible est installé ; si
aucun ne l'est, le livré reste et le journal dit quel garde-fou a cédé et sur quel circuit. À structure égale, 10
graines passent avant 5 (même structure, moins de variance) ; « sans recalage » passe avant « avec » (aucune correction
apprise). Les scores ne servent pas à retoucher la liste, les poids ou les seuils.

**Variante sans recalage** : `predict.py --options '{"recalage": 0.0}'` ; seule la part boosting est touchée (S ; B
n'est jamais recalée) ; le GRU S garde son recalage 1.0, comme dans la définition de la mission.

**Indicateurs de rapport (jamais appris, jamais utilisés par la règle)** : « régime haut hors saison » d'un circuit =
heures d'un jour sans ombre de Mars dont la semaine a une moyenne vraie au-dessus de la médiane des moyennes
hebdomadaires du circuit sur annee1 + annee2.

## Étape 1 — Livré gelé (`runs/st0_gel/`)

`empreintes.json` : commit `06988f9`, sha256 des 13 fichiers de `poids/` et des 4 prédictions de référence
(`rc0_livre_g04|g59`). Inférence refaite sur des copies liées (`runs/st0_livre_g04|g59`, `valider.py
--sans-entrainement`) : voir ci-dessous.

| | Directe | Inverse |
|---|---:|---:|
| graines 0–4 | 0.075046 | 0.078474 |
| graines 5–9 | 0.075078 | 0.078343 |
| **Moyenne des 4** | | **0.076735** |

Les 4 fichiers de prédiction sont identiques au bit près à ceux de `rc0_livre_*`.

## Code ajouté (options désactivées par défaut ; livré inchangé)

- `outils/faire_chrono.py --glissant` (dossiers `validation_glissant/c50…c87`, ignorés par git ; aussi construits dans
  chaque session Kaggle par `outils/kaggle_distant.py`) ; `outils/valider.py` connaît `--direction c50 c62 c75 c87`.
  Contrôle : les 6 fichiers de `validation_glissant/c75/entrainement` sont identiques à ceux de
  `validation_chrono/entrainement`.
- `baselines/recette_b.py` : `entrainer` / `predire` (modèle `recette_b` de `dio/modeles.py`) ; l'ancienne fonction
  `predire(modeles, x)` devient `predire_sacs`.
- `dio/ensemble.py` : `part_recette = 1` ⇒ GRU mixte non chargé ; clé `gbm2` d'`ensemble.json` (second boosting) ;
  règle `recette_b` de NPWD2551 (dossier `specialiste/b_NPWD2551/`) ; option de prédiction `parts` (part GRU, part
  boosting, mélange avant les spécialistes 2531/2532, temps par composant, dans `<prédiction>.parts/`).
- `outils/candidat.py` : `--directions`, `--mixte`, `--gbm2`, `--b`, `--part-b` ; copies par liens durs (aucune
  écriture dans les fichiers de la base). `outils/fusionner_recette.py` : GRU S à 10 graines. `outils/structure.py` :
  assemblage, prédictions, bilan, décision.
- GRU S à 10 graines en directe et en inverse : `rc3_gruA_norm` + `rc4_gruA_mm_g59` fusionnés (`runs/st_gruS10`),
  configuration, variables, familles et statistiques identiques.

## Entraînements Kaggle (`zendarc148`, T4)

| Session | Contenu |
|---|---|
| S1 (fente 1) | B directe, B inverse ; c62 : B, GRU S g04 et g59, GBM S, GRU mixte g04, structure livrée complète (`train.py --modele gru_gbm`, `gru_depuis` = mixte g04 : R, G2, D3L, physique, marqueurs), GRU mixte g59, G2 g59 |
| S2 (fente 2) | c87 : idem |
| S3 | c75 : idem |
| S4 | finaux sur annee1 + annee2 : B, GRU S g04 et g59, GBM S |

Directe, inverse et c50 réutilisent les modèles existants (`rc0_livre_*`, `rc3_gruA_norm`, `rc4_gruA_mm_g59`,
`rc1_gbm_S`) ; seule B y est entraînée.

Durées d'entraînement sur T4 (secondes, `Entraînement terminé en …` des journaux Kaggle, `runs/_kaggle/`) :

| Composant | c62 (S1) | c75 (S3) | c87 (S2) | annee1 + annee2 (S4) |
|---|---:|---:|---:|---:|
| B | 82 | 93 | 109 | 133 |
| GRU S g04 / g59 | 390 / 389 | 482 / 475 | 538 / 534 | 619 / 612 |
| GBM S (compressé) | 970 | 1111 | 1492 | 2057 |
| GRU mixte g04 / g59 | 599 / 609 | 609 / 618 | 619 / 620 | — (livré) |
| structure livrée (R, G2 g04, D3L, physique, marqueurs) | 3094 | 3635 | 4129 | — (livré) |
| G2 g59 | 394 | 482 | 532 | — |

B directe et inverse (S1) : 71 s et 83 s. Contrôle : leurs prédictions sur `validation/test` et
`validation_inverse/test` égalent `runs/b3k_rec_recette_coll/*/pred.csv` (écart maximal 2.2e-16).

## Contrôles d'assemblage

- L réassemblé = `rc0_livre_*`, C1 = `rc6_val_*` : prédictions identiques au bit près (4 runs chacun).
- T2 et T3 contre leur version sans B (`st_T2ctl_directe_g04`, `st_T3ctl_directe_g04`) : seule la colonne NPWD2551
  diffère.
- `poids/` inchangé pendant toutes les mesures (13 empreintes du gel retrouvées).

## Résultats (de bout en bout, avec spécialistes)

Score d'une structure à 5 graines = moyenne des groupes g04 et g59 ; à 10 graines = un seul run. Bilan complet par run :
`runs/st_bilan/bilan.json` ; décision : `runs/st_bilan/decision.json` ; sobriété : `runs/st_bilan/sobriete.json`.

RMSE globale :

| Structure | Directe | Inverse | Moy. 4 | c50 | c62 | c75 | c87 | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 0.075062 | 0.078409 | 0.076735 | 0.064705 | 0.065850 | 0.088987 | 0.067998 | 0.071885 |
| C1 (5 g) | 0.074384 | 0.078192 | 0.076288 | 0.064088 | 0.064195 | 0.090523 | 0.071123 | 0.072482 |
| T1 (5 g) | 0.074371 | 0.079234 | 0.076802 | 0.063394 | 0.062569 | 0.092175 | 0.077072 | 0.073802 |
| T1 (10 g) | 0.074348 | 0.079197 | 0.076773 | 0.063379 | 0.062555 | 0.092161 | 0.076994 | 0.073772 |
| T2 (5 g) | 0.074310 | 0.079728 | 0.077019 | 0.063643 | 0.063110 | 0.092056 | 0.076645 | 0.073864 |
| T2 (10 g) | 0.074288 | 0.079691 | 0.076990 | 0.063628 | 0.063096 | 0.092042 | 0.076566 | 0.073833 |
| T3 (5 g) | 0.074626 | 0.080483 | 0.077554 | 0.064226 | 0.063862 | 0.092223 | 0.076442 | 0.074188 |
| T3 (10 g) | 0.074604 | 0.080447 | 0.077525 | 0.064210 | 0.063848 | 0.092209 | 0.076363 | 0.074158 |
| T3sr (5 g) | 0.074678 | 0.081122 | 0.077900 | 0.064420 | 0.063984 | 0.092614 | 0.076230 | 0.074312 |
| T3sr (10 g) | 0.074656 | 0.081086 | 0.077871 | 0.064405 | 0.063971 | 0.092600 | 0.076150 | 0.074282 |

Quatre objets, RMSE globale, moyenne des 6 validations (et c87) :
| Structure | Part GRU | Part boosting | Mélange | Complet | Corr. erreurs GRU/boosting | Part GRU c87 | Part boosting c87 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L | 0.076217 | 0.076887 | 0.074037 | 0.073502 | 0.868 | 0.074269 | 0.071038 |
| C1 (5 g) | 0.076411 | 0.076887 | 0.074080 | 0.073751 | 0.865 | 0.079433 | 0.071038 |
| T1 (5 g) | 0.080840 | 0.076887 | 0.075110 | 0.074802 | 0.811 | 0.094761 | 0.071038 |
| T1 (10 g) | 0.080730 | 0.076887 | 0.075077 | 0.074772 | 0.812 | 0.094507 | 0.071038 |
| T2 (5 g) | 0.080840 | 0.077219 | 0.075198 | 0.074915 | 0.808 | 0.094761 | 0.070772 |
| T2 (10 g) | 0.080730 | 0.077219 | 0.075166 | 0.074885 | 0.809 | 0.094507 | 0.070772 |
| T3 (5 g) | 0.080840 | 0.078620 | 0.075564 | 0.075310 | 0.793 | 0.094761 | 0.071504 |
| T3 (10 g) | 0.080730 | 0.078620 | 0.075532 | 0.075280 | 0.794 | 0.094507 | 0.071504 |
| T3sr (5 g) | 0.080840 | 0.080292 | 0.076172 | 0.075508 | 0.785 | 0.094761 | 0.071582 |
| T3sr (10 g) | 0.080730 | 0.080292 | 0.076140 | 0.075478 | 0.786 | 0.094507 | 0.071582 |

Circuits suivis, RMSE / biais, moyenne des 4 coupures :
| Structure | NPWD2551 | NPWD2451 | NPWD2851 | NPWD2532 | NPWD2531 |
|---|---:|---:|---:|---:|---:|
| L | 0.2444 / +0.0324 | 0.2154 / +0.0200 | 0.1206 / -0.0337 | 0.1123 / +0.0027 | 4.43e-02 / -1.3e-02 |
| C1 (5 g) | 0.2496 / +0.0449 | 0.2187 / +0.0263 | 0.1170 / -0.0259 | 0.1123 / +0.0027 | 4.37e-02 / -1.3e-02 |
| T1 (5 g) | 0.2579 / +0.0578 | 0.2237 / +0.0329 | 0.1140 / -0.0156 | 0.1123 / +0.0027 | 4.23e-02 / -1.3e-02 |
| T1 (10 g) | 0.2578 / +0.0577 | 0.2237 / +0.0329 | 0.1140 / -0.0156 | 0.1123 / +0.0027 | 4.23e-02 / -1.3e-02 |
| T2 (5 g) | 0.2590 / +0.0636 | 0.2234 / +0.0270 | 0.1138 / -0.0147 | 0.1122 / +0.0010 | 4.24e-02 / -1.3e-02 |
| T2 (10 g) | 0.2589 / +0.0636 | 0.2234 / +0.0270 | 0.1138 / -0.0147 | 0.1122 / +0.0010 | 4.24e-02 / -1.3e-02 |
| T3 (5 g) | 0.2612 / +0.0695 | 0.2237 / +0.0210 | 0.1137 / -0.0138 | 0.1129 / -0.0007 | 4.25e-02 / -1.3e-02 |
| T3 (10 g) | 0.2610 / +0.0694 | 0.2237 / +0.0210 | 0.1137 / -0.0138 | 0.1129 / -0.0007 | 4.25e-02 / -1.3e-02 |
| T3sr (5 g) | 0.2612 / +0.0695 | 0.2248 / +0.0229 | 0.1136 / -0.0176 | 0.1135 / -0.0015 | 4.29e-02 / -1.3e-02 |
| T3sr (10 g) | 0.2610 / +0.0694 | 0.2248 / +0.0229 | 0.1136 / -0.0176 | 0.1135 / -0.0015 | 4.29e-02 / -1.3e-02 |

Robustesse (coupures) :
| Structure | Sans biais hebdo (moy.) | Pire mois (max) | Pire 28 j (max) | Dérive q4 − q1 (moy.) | 2551 biais hors saison | 2551 biais haut hors saison | Épisodes 2551 (nombre / jours, somme) | Prédiction (s, c87) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 0.063954 | 0.1021 | 0.1111 | +0.0062 | +0.0128 | -0.1130 | 7 / 52 | 646 |
| C1 (5 g) | 0.063925 | 0.1061 | 0.1151 | +0.0063 | +0.0215 | -0.1093 | 8 / 51 | 616 |
| T1 (5 g) | 0.064208 | 0.1095 | 0.1178 | +0.0067 | +0.0275 | -0.1074 | 10 / 60 | 567 |
| T1 (10 g) | 0.064188 | 0.1093 | 0.1178 | +0.0067 | +0.0274 | -0.1074 | 11 / 62 | 722 |
| T2 (5 g) | 0.064257 | 0.1099 | 0.1185 | +0.0069 | +0.0368 | -0.0980 | 8 / 50 | 705 |
| T2 (10 g) | 0.064237 | 0.1098 | 0.1185 | +0.0069 | +0.0368 | -0.0981 | 7 / 47 | 755 |
| T3 (5 g) | 0.064473 | 0.1105 | 0.1194 | +0.0071 | +0.0462 | -0.0887 | 6 / 42 | 298 |
| T3 (10 g) | 0.064453 | 0.1104 | 0.1193 | +0.0071 | +0.0462 | -0.0888 | 6 / 43 | 353 |
| T3sr (5 g) | 0.064472 | 0.1110 | 0.1198 | +0.0069 | +0.0462 | -0.0887 | 6 / 42 | 369 |
| T3sr (10 g) | 0.064452 | 0.1108 | 0.1198 | +0.0069 | +0.0462 | -0.0888 | 6 / 43 | 255 |

Lecture (constat, pas décision) :
- **Le GRU S seul comme part GRU (T1–T3) gagne sur c50 et c62 mais perd beaucoup sur c75 et surtout c87** : sur c87,
  la part GRU passe de 0.0743 (livré) à 0.0945 (GRU S, 10 graines). Côté boosting, S fait presque jeu égal avec R
  (part boosting sur c87 : S 0.0715, R 0.0710, (R + S)/2 0.0708) ; B n'améliore pas NPWD2551 sur les coupures.
- Sur c87, la hausse d'erreur de T1–T3 vient à 76–81 % de NPWD2551 et à 20–24 % de NPWD2451 (part de la hausse de MSE,
  31 autres colonnes < 2 % chacune). Le GRU S **surestime** ces chauffages hors saison (biais NPWD2551 c87 : livré
  +0.065, C1 +0.097, T1 +0.137) : c'est l'inverse de la sous-estimation redoutée au départ.
- C1 (moitié GRU mixte, moitié GRU S dans la part GRU) garde l'essentiel du gain des 4 RMSE temporelles (−0.00045) mais
  perd sur les deux coupures les plus récentes (c75 +0.0015, c87 +0.0031 ; 78 % de la hausse de c87 sur NPWD2551, 26 %
  sur NPWD2451).
- T3 sans recalage est le plus mauvais partout sauf c87 ; le recalage de la part boosting aide sur les 4 RMSE.
- 10 graines contre 5 : −0.00003 sur les 4 RMSE, sous le bruit.

Sobriété des composants (directe, `outils/structure.py sobriete`) :

| Composant | Entrées / colonnes | Colonnes lues par les arbres | Arbres | Graines |
|---|---:|---:|---:|---:|
| GRU mixte (2 membres) | 172 et 236 | — | — | 2 × 5 |
| G2 | 236 | — | — | 5 |
| GRU S borné | 165 | — | — | 5 (ou 10) |
| R (33 circuits) | 801 | 689 | 37 569 | — |
| S (33 circuits) | 673 | 611 | 36 815 | — |
| D3L (NPWD2551) | 807 | 621 | 26 198 | — |
| B (NPWD2551) | 684 | 404 | 3 000 | — |

Structures : L = GRU mixte (2 × 5) + G2 (5) + R + D3L ; C1 = GRU mixte + GRU S + R + D3L (G2 remplacé par un GRU
plus petit) ; T1 = GRU S + R + D3L (sans GRU mixte) ; T2 = GRU S + (R + S)/2 + (D3L + B)/2 ; T3 = GRU S + S + B.
NPWD2532 (physique + boosting) et NPWD2531 (porte par marqueurs) sont les mêmes dans toutes. Les temps de prédiction
de `temps.json` (4 prédictions en parallèle sur la même machine) sont bruités ; T3 est le plus rapide (300 s environ
contre 650 s pour L sur c87), le boosting de 33 circuits dominant le temps.

## Décision mécanique (`outils/structure.py decider`)

Livré : moyenne des 4 RMSE temporelles 0.076735, moyenne des coupures 0.071885. Écarts au livré (positif = pire) :

| Ordre | Candidat | Δ moy. 4 | Δ max des 4 | Δ moy. coupures | Δ c50 | Δ c62 | Δ c75 | Δ c87 | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | T3sr_g10 | +0.001136 | +0.002742 | +0.002397 | -0.000300 | -0.001879 | +0.003613 | +0.008152 | refusé : moyenne_4 <= +0.0010 ; chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| 2 | T3sr_g5 | +0.001165 | +0.002774 | +0.002427 | -0.000285 | -0.001865 | +0.003627 | +0.008232 | refusé : moyenne_4 <= +0.0010 ; chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| 3 | T3_g10 | +0.000790 | +0.002103 | +0.002273 | -0.000495 | -0.002002 | +0.003222 | +0.008365 | refusé : chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| 4 | T3_g5 | +0.000819 | +0.002137 | +0.002303 | -0.000479 | -0.001988 | +0.003236 | +0.008444 | refusé : chacune des 4 <= +0.0020 ; moyenne coupures <= +0.0020 ; chaque coupure <= +0.0040 |
| 5 | T2_g10 | +0.000254 | +0.001348 | +0.001948 | -0.001077 | -0.002753 | +0.003056 | +0.008568 | refusé : chaque coupure <= +0.0040 |
| 6 | T2_g5 | +0.000284 | +0.001398 | +0.001979 | -0.001062 | -0.002740 | +0.003069 | +0.008647 | refusé : chaque coupure <= +0.0040 |
| 7 | T1_g10 | +0.000037 | +0.000854 | +0.001887 | -0.001326 | -0.003295 | +0.003174 | +0.008996 | refusé : chaque coupure <= +0.0040 |
| 8 | T1_g5 | +0.000067 | +0.000920 | +0.001918 | -0.001311 | -0.003281 | +0.003188 | +0.009074 | refusé : chaque coupure <= +0.0040 |
| 9 | C1_g5 | -0.000447 | -0.000155 | +0.000597 | -0.000618 | -0.001655 | +0.001536 | +0.003125 | **admissible, installé** |

**C1 (5 graines) est installé** : premier admissible dans l'ordre fixé. Tous les T cèdent sur « aucune coupure ne
monte de plus de 0.0040 », toujours sur c87 (+0.008 à +0.009), à cause de NPWD2551 et NPWD2451 (voir plus haut). T3
et T3sr cèdent aussi sur la moyenne des coupures et sur une des 4 RMSE (inverse) ; T3sr aussi sur la moyenne des 4.

## Installation

- `poids_repli/` : copie complète de l'ancien `poids/` (livré 0.076735) ; `poids_repli/EMPREINTES.json` (sha256),
  identiques aux empreintes du gel.
- `poids/` = `poids_candidats/structure/C1_g5/` : seul `recette/recette.pt` change (GRU S borné, graines 0–4,
  annee1 + annee2, Kaggle S4, `runs/st_final_gruS_g04`) et `ensemble.json` (`part_recette` 0.3 → 0.5) ; GRU mixte,
  R, D3L, physique et marqueurs sont ceux du livré. `poids/modele.json` : `options_recette` et note.
- `dio/ensemble.py` : `PART_RECETTE = 0.5`, `RECETTE = {"variables": "S", "reinjection": false, "normalisation":
  "minmax"}` (défauts changés à l'installation seulement ; `train.py --modele gru_gbm` reproduit la structure).
- Part du réseau : inchangée, 0.5 du mélange (part GRU = 0.5 GRU mixte + 0.5 GRU S) ; NPWD2532 : 0.4 réseau.
- `predict.py` avec le nouveau `poids/` sur `validation/test` : 107 s, format et valeurs vérifiés par la suite de
  tests (`uv run python -m unittest discover tests` : 112 tests OK, 2 sautés ; `tests.test_ensemble` relancé seul
  sur le `poids/` final : 5 OK). Score sur cette période sans valeur (annee2 est dans l'entraînement des poids finaux).
- Incident corrigé : `poids_candidats/structure/*` partageaient leurs fichiers avec `poids/` par liens durs ; la
  réécriture sur place de `poids/modele.json` a modifié leurs `modele.json`. Tous les fichiers de `poids/` sont
  désormais indépendants ; les `modele.json` des candidats ont été rétablis (celui du livré, sauf C1_g5 : celui de
  `poids/`). Les autres fichiers n'ont pas été écrits sur place ; `poids_repli/` = gel, vérifié après coup.

Structure choisie pour sa sobriété sur décision de l'utilisateur ; elle ne maximise pas la validation interne.

## Poids de tous les candidats (`poids_candidats/structure/`, non suivis par git sauf `INDEX.md`)

`python outils/structure.py finaux` : un dossier complet par candidat (L, C1_g5, T1/T2/T3/T3sr en 5 et 10 graines),
base = livré, composants nouveaux de S4. Tous passent `predict.py` (3953 × 33 valeurs finies ≥ 0 sur la coupure c87).
Détail et verdicts : `poids_candidats/structure/INDEX.md`.

## Reproduction

```bash
python outils/faire_chrono.py --glissant          # validation_glissant/c50…c87
python runs/st_kaggle.py                          # listes de commandes des sessions S1–S4
runs/st_lancer.sh S1 1                            # idem S2, S3, S4 (Kaggle T4, rapatriement des poids)
python outils/structure.py assembler --controles
python outils/structure.py predire
python outils/structure.py bilan
python outils/structure.py sobriete
python outils/structure.py decider
python outils/structure.py finaux
```

## Limites

- **Seuils jugés mal placés (signalés, non modifiés)** : la tolérance « chaque coupure ≤ +0.0040 » vaut 10 à 40 fois le
  bruit mesuré sur les coupures (0.0001 à 0.0004). Elle admet C1 alors que C1 est pire que le livré sur les deux
  coupures les plus récentes (c75, c87), c'est-à-dire celles qui ressemblent le plus à une année inconnue qui suit
  l'entraînement ; la moyenne des coupures de C1 est aussi au-dessus du livré (+0.0006). Avec un seuil proportionné au
  bruit, aucun candidat n'aurait passé et le livré serait resté.
- Le gain de C1 sur les 4 RMSE temporelles (−0.00045) n'est que quatre fois l'effet d'environnement.
- c50 réutilise les modèles de la directe : c50 et la directe ne sont pas indépendantes.
- Quatre coupures sur deux années seulement ; c87 contient le régime haut hors saison de février–mars 2012, une seule
  occurrence.
- Les poids finaux du GRU S n'ont pas de score propre : la règle a été appliquée sur les coupures, puis la structure a
  été réentraînée sur annee1 + annee2.

## Installation de T3 (décision de l'utilisateur, 16 novembre 2026)

L'utilisateur a demandé d'installer T3 comme modèle principal. Ce n'est pas un résultat de la règle : T3 y était
refusé (une des 4 RMSE, moyenne des coupures, c87). Variante prise : **T3 à 10 graines, avec recalage** de la part
boosting (`poids_candidats/structure/T3_g10/`), première variante de T3 dans l'ordre de sobriété fixé avant mesure
(10 graines avant 5 ; T3sr est une autre ligne de la liste).

- `poids/` = copie indépendante (sans liens durs) de `poids_candidats/structure/T3_g10/` : `recette/recette.pt`
  (GRU S borné, graines 0–9, `runs/st_final_gruS_*` fusionnés), `gbm/` (boosting S), `specialiste/b_NPWD2551/`
  (recette B), physique et marqueurs du livré ; `ensemble.json` : `part_recette` 1.0. Plus de GRU mixte
  (`membre0/1`) ni de `gbm_NPWD2551/`. `poids/modele.json` : note, `options_recette`, `options_gbm`.
- C1 reste dans `poids_candidats/structure/C1_g5/` (vérifié identique à l'ancien `poids/`, fichier par fichier), le
  livré L dans `poids_repli/`.
- Scores connus (aucune nouvelle mesure) : moyenne des 4 RMSE 0.077525 (directe 0.074604, inverse 0.080447) ;
  coupures c50–c87 0.064210 / 0.063848 / 0.092209 / 0.076363, moyenne 0.074158.
- `dio/ensemble.py` n'est pas modifié : `train.py --modele gru_gbm` reproduit toujours C1. Le GRU S à 10 graines est
  une fusion de deux entraînements (`outils/fusionner_recette.py`), que `train.py` ne fait pas seul.
