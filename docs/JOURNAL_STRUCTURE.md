# Journal — structure sobre de l'ensemble (à partir du 15 novembre 2026, soir)

Mission fournie par l'utilisateur : rendre le modèle livré plus sobre et plus robuste à une année inconnue, par un
changement de structure de l'ensemble et de règle de décision (pas de recherche de variables ni d'architecture).
Branche locale `recette-complete`, rien n'est poussé. **Tous les entraînements tournent sur Kaggle `zendarc148`
(sessions T4)**, à la demande de l'utilisateur pour cette session ; en local : assemblages, prédictions avec des poids
existants, scores, tests.

## Reprendre ici

- Sessions Kaggle : `runs/st_S{1,2,3,4}.log` / `.pid` (attente par `kill -0 PID`), commandes `runs/st_kaggle.py`,
  lanceur `runs/st_lancer.sh SESSION FENTE`.
- Statut : voir la dernière section remplie.

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
