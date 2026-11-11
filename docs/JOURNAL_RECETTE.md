# Journal — recette GBM puis GRU (à partir du 11 novembre 2026)

Programme en 9 points fourni par l'utilisateur : familles de commandes, retrait des encodages
par lettre de variante, événements de commande, soleil, géométrie orbitale, sélection des
entrées, réglage HGB, puis recette GRU. Chaque point est testé seul : hypothèse, modification,
comparaison, décision. Branche locale `recette` (depuis `ensemble-gbm`), un commit par décision,
rien n'est poussé. Les chiffres extérieurs donnés par l'utilisateur orientent les essais. Ils ne
sont pas reportés ici ; seules les mesures du dépôt le sont.

## Reprendre ici

- Branche : `recette`.
- État accepté du boosting (options de `gbm_sacs.entrainer`) :
  options de `runs/e22b_recette_sans_ema/directe/modele.json` (recette R, E22b gardé ; référence de comparaison : `e22b_recette_sans_ema`).
- Référence du boosting pour `outils/comparer_recette.py` : `ref_gbm` (voir phase 0).
- Prochaine étape : voir la dernière section.
- Runs Kaggle en cours : aucun (sinon `python outils/lancer_kaggle.py --fente N --rapatrier`).

## Protocole

- Validation directe (annee1 → annee2, `validation/`) et inverse (annee2 → annee1,
  `validation_inverse/`), notée par `evaluate.py`.
- Boosting entraîné sur Kaggle T4 :
  `python outils/lancer_kaggle.py --fente N -- outils/valider.py --nom NOM --modele gbm_sacs --options 'JSON'`.
  En local, seulement les calculs de moins de 2 min (re-prédictions, notes, mélanges, tests).
- Comparaison : `python outils/comparer_recette.py --reference REF --candidat NOM`. Mélange
  0.7 × GRU + 0.3 × boosting, poids fixe, avec les GRU de référence `runs/p1_gru_g04` et
  `runs/p1_gru_g59` (graines 0–4 et 5–9). Sans spécialistes.
- Règle de décision, fixée avant les mesures. On garde si :
  (a) la moyenne des 4 RMSE du mélange baisse d'au moins 0.0003, et aucune direction (moyenne des
  deux groupes de graines) ne monte de plus de 0.0002 ; ou
  (b) le boosting seul baisse dans les deux directions, et le mélange ne monte pas de plus de 0.0001.
  Sinon, rejet : l'option reste dans le code, désactivée par défaut.
- Toujours rapportés : boosting seul, mélange, contributions MSE de NPWD2551, NPWD2451 et
  NPWD2851 (×1e-4, mélange, moyenne des deux groupes de graines). Les 33 circuits sont dans
  `runs/NOM/comparaison.json`.

## Phase 0 — référence figée (11 nov.)

### Prédictions figées (sha256, `evaluate.py` refait : scores égaux aux `scores.json`)

| Run | Rôle | Directe | Inverse | sha256 directe / inverse |
|---|---|---:|---:|---|
| `p1_gru_g04` | GRU seul, graines 0–4 | 0.079062 | 0.083579 | `13bfd296…` / `c3039f7e…` |
| `p1_gru_g59` | GRU seul, graines 5–9 | 0.078719 | 0.084229 | `70d1615b…` / `60e9126d…` |
| `p2_planning_ema` | boosting seul (installé) | 0.085825 | 0.082774 | `97e1a519…` / `d14354b7…` |
| `p3_val_g04` | mélange 0.3, sans spécialistes | 0.077975 | 0.081042 | `345a256e…` / `b225e177…` |
| `p3_val_g59` | idem, graines 5–9 | 0.077769 | 0.081435 | `4c767223…` / `bfb22cc8…` |
| `sp31_val_g04` | avec spécialistes | 0.076748 | 0.079111 | `239d4680…` / `dd1b8a72…` |
| `sp31_val_g59` | idem, graines 5–9 | 0.076533 | 0.079497 | `c18838ae…` / `2a0f0bad…` |

Re-prédiction locale des mêmes poids (`runs/ref_gbm`, `runs/ref_mel_g04`, `valider.py
--sans-entrainement`) : mêmes RMSE à 1e-6. Les fichiers diffèrent au dernier bit (écart max
4.4e-16, sommes en parallèle) : reproduit, pas identique au bit près.

Mélange de référence (`comparer_recette.py`) : 0.077872 en directe, 0.081238 en inverse
(moyenne des groupes), **moyenne des 4 RMSE 0.079555**.

### Où est l'erreur (contributions MSE ×1e-4, graines 0–4)

| Circuit | Directe : GBM / GRU / mélange | Inverse : GBM / GRU / mélange |
|---|---|---|
| NPWD2551 | 29.22 / 21.18 / 20.78 | 16.11 / 13.19 / 12.94 |
| NPWD2451 | 17.31 / 15.05 / 14.89 | 16.98 / 14.35 / 14.32 |
| NPWD2851 | 6.87 / 5.79 / 5.92 | 5.45 / 6.51 / 6.04 |
| NPWD2532 | 4.83 / 4.92 / 4.57 | 9.39 / 12.74 / 10.80 |
| NPWD2531 | 3.45 / 4.47 / 3.79 | 9.98 / 10.99 / 10.38 |
| MSE totale | 73.66 / 62.51 / 60.80 | 68.52 / 69.85 / 65.68 |

Les trois circuits suivis portent 73 % (directe) et 56 % (inverse) de la MSE du boosting seul.
En directe, le boosting perd contre le GRU surtout sur NPWD2551 (+8.0e-4), 2451 (+2.3e-4) et
2851 (+1.1e-4). En inverse, il perd sur 2551 et 2451, mais gagne sur 2532, 2531 et 2851.

### Colonnes réellement consommées

- Boosting installé (directe, 689 colonnes après filtre) : 351 `plan_*`, 104 `dmopfam_*`,
  82 `ftl_*`, 54 `flux_*`, 49 `evtf_*`, 16 `saaf_*`, 8 `ombre*`, 6 `ltdata_*`, 6 `orbite_*`,
  3 `dmop_*` globaux, et **10 comptes exacts avec lettre de variante** (`dmop_AAAAF56A1_count`,
  `dmop_AAAAF20C1_count`, `dmop_AAAAF20E1_count`, `dmop_AAAAF23G1_count`, …). Pas d'état par
  lettre. Paires début/fin apprises : MOCS/MOCE et PENS/PENE.
- GRU de référence (membre v1, 243 variables) : 112 `dmopfam_*`, 66 `dmopcmd_*` (33 impulsions
  et 33 états par commande exacte AOOO/ATTT, dans la branche vers NPWD2881), les 10 mêmes comptes
  exacts, etc.

### Défaut de regroupement mesuré

`commandes.famille` (`^(A[A-Z]{3}F?\d+)`) coupe tout ce qui suit le numéro : AOOOF01D0 et
AOOOF01D1 tombent dans AOOOF01. Contre la règle corrigée (`famille_variante`), 8 anciennes
familles en regroupent plusieurs sur annee1 (AAAAF23, AHHHF17, AMMMF52, AOOOF02, AOOOF03,
AOOOF22, APSF06, APSF30), 6 sur annee2 (dont AOOOF01). Vocabulaire : 112 familles (≥ 20, ancienne
règle) contre 117 (≥ 50, nouvelle) sur annee1 ; 144 contre 142 sur annee2.

### Reproductibilité sur Kaggle

`r0_gbm` : boosting installé réentraîné sur Kaggle (1356 s pour les deux directions en parallèle).
- Prédictions faites sur Kaggle : directe 0.085825 (égale), inverse 0.082784 au lieu de 0.082774.
  Le `validation_inverse/` de Kaggle est reconstruit à partir de `y.csv` : la dernière heure,
  partielle, de power_init diffère, ce qui change le recalage.
- Mêmes modèles re-prédits en local : écart max 4.4e-16 avec `p2_planning_ema`, mêmes RMSE.
  **L'entraînement sur Kaggle reproduit le boosting installé.**
- Protocole retenu : entraînement sur Kaggle, téléchargement des modèles (`"compresser": true` pour
  limiter la taille), re-prédiction locale (`valider.py --sans-entrainement`, environ 40 s). Les
  prédictions Kaggle sont gardées dans `pred_kaggle.csv`. Référence du boosting : `p2_planning_ema`.
- Le téléchargement des pickles non compressés (150 Mo par direction) dépassait les 600 s de
  `lancer_kaggle.py` : délai porté à 3600 s pour `kernels output`.

## Expériences du boosting

Mode de travail : chaque variante d'une même vague part du même état accepté et ne change qu'une
chose. Les variantes gardées sont ensuite réunies, et la réunion est mesurée à son tour.

Code ajouté dans `baselines/gbm_sacs.py` (toutes les options sont désactivées par défaut, le boosting
installé ne change pas) :
- `familles_variante` (E1) : `commandes.famille_variante` dans `dmopfam_*` et `plan_*`. Vocabulaire :
  ≥ 50 occurrences dans une période d'entraînement, union des périodes. `MOCS.n` reste groupé en `MOCS`
  dans le planning ;
- `retirer` (E2, E17) : expressions régulières de colonnes écartées ; `LETTRES` = les 10 comptes exacts ;
- `plan` (E3–E6) : options de `apprendre_planning` : `changements`, `recence: "debut"`, `prefixes`,
  `paires: "liste"` ;
- `contexte` (E7–E9) : `soleil: "mesures" | "interpole"`, `ltdata: "lineaire"` (`dio/contexte.py`) ;
- ajouts `geometrie`, `altitude`, `ombres`, `contacts` (E10–E13), `decalages`, `histoires` (E14, E15) ;
- réglage `D` (E18) : celui du point 8, avec arrêt anticipé.

### Analyses préalables (sans entraînement)

- **Contacts stations (E13).** Les AOS/LOS portent un seuil d'élévation (`_00` à `_11`). Seul `_10`
  existe pour toutes les stations dans les deux années : c'est lui qui est apparié. L'union vaut 1 sur
  plus de 99.5 % des heures (au moins une station voit toujours la sonde) : le filtre des variables
  quasi constantes l'écarte. La somme par station (0 à 7) reste.
- **Ombres des lunes et occultations (E12).** Les intervalles PHO_/DEI_UMBRA, PENUMBRA et OCC_PHOBOS,
  OCC_DEIMOS sont nuls sur plus de 99.5 % des heures : le filtre les écarte tous. La fraction
  OCC_MARS est égale, au facteur 60 près, à `evtf_OCC_MARS_minutes` déjà présent (écart max 7e-15).
  Le HGB découpe les variables par quantiles : il est insensible à ce facteur. Avec le filtre, E12 ne
  change donc rien, et il n'est pas lancé seul. Il reste testé dans la recette complète (E22), avec le
  test sans filtre (E20). Pairage : un DEI_UMBRA_END rangé avant son START au même instant ouvrait un
  faux intervalle de plusieurs mois. Les paires de plus de 6 h (ombres) ou 24 h (contacts) sont écartées.
- **Flux par face (E16), corrigé après vérification colonne par colonne.** Formule :
  `(constante/600) × max(±cos, 0) × clip(1 − umbra − 0.5 pénombre, 0, 1)`, avec la moyenne horaire des
  cosinus des mesures. Écart maximal avec `flux_*_eclaire / 600` :
  - boosting installé (cosinus de l'angle moyen) : 0.1 à 0.4 selon la face et l'année. **Pas
    équivalent** : ma première conclusion (simple division par 600) était fausse ;
  - option `soleil: "mesures"` : 1e-16 sur annee1 ; écarts sur annee2, aux 259 heures sans mesure
    (cosinus de l'angle moyen rempli) ;
  - option `soleil: "interpole"` : 1e-16 sur les deux années ;
  - correction d'ombre : `1 − clip(u + 0.5p, 0, 1)` et `clip(1 − u − 0.5p, 0, 1)` sont égales
    (u, p ≥ 0) ; signes des faces identiques, sx− toujours nul ; **face sa− absente** du V2 (non nulle
    sur 0.01 % des heures, écartée par le filtre).
  E16 se ramène donc à E7/E8 (moyenne des cosinus), plus la face sa− (ajout `face_sa_moins`).

### Reproductibilité sur Kaggle

`r0_gbm` : boosting installé réentraîné sur Kaggle (1356 s pour les deux directions en parallèle).
- Prédictions faites sur Kaggle : directe 0.085825 (égale), inverse 0.082784 au lieu de 0.082774.
  Le `validation_inverse/` de Kaggle est reconstruit à partir de `y.csv` : la dernière heure,
  partielle, de power_init diffère, ce qui change le recalage.
- Mêmes modèles re-prédits en local : écart max 4.4e-16 avec `p2_planning_ema`, mêmes RMSE.
  **L'entraînement sur Kaggle reproduit le boosting installé.**
- Protocole retenu : entraînement sur Kaggle, téléchargement des modèles (`"compresser": true` pour
  limiter la taille), re-prédiction locale (`valider.py --sans-entrainement`, environ 40 s). Les
  prédictions Kaggle sont gardées dans `pred_kaggle.csv`. Référence du boosting : `p2_planning_ema`.
- Le téléchargement des pickles non compressés (150 Mo par direction) dépassait les 600 s de
  `lancer_kaggle.py` : délai porté à 3600 s pour `kernels output`.

## Expériences du boosting

Mode de travail : chaque variante d'une même vague part du même état accepté et ne change qu'une
chose. Les variantes gardées sont ensuite réunies, et la réunion est mesurée à son tour.

Code ajouté dans `baselines/gbm_sacs.py` (toutes les options sont désactivées par défaut, le boosting
installé ne change pas) :
- `familles_variante` (E1) : `commandes.famille_variante` dans `dmopfam_*` et `plan_*`. Vocabulaire :
  ≥ 50 occurrences dans une période d'entraînement, union des périodes. `MOCS.n` reste groupé en `MOCS`
  dans le planning ;
- `retirer` (E2, E17) : expressions régulières de colonnes écartées ; `LETTRES` = les 10 comptes exacts ;
- `plan` (E3–E6) : options de `apprendre_planning` : `changements`, `recence: "debut"`, `prefixes`,
  `paires: "liste"` ;
- `contexte` (E7–E9) : `soleil: "mesures" | "interpole"`, `ltdata: "lineaire"` (`dio/contexte.py`) ;
- ajouts `geometrie`, `altitude`, `ombres`, `contacts` (E10–E13), `decalages`, `histoires` (E14, E15) ;
- réglage `D` (E18) : celui du point 8, avec arrêt anticipé.

### Analyses préalables (sans entraînement)

- **Contacts stations (E13).** Les AOS/LOS portent un seuil d'élévation (`_00` à `_11`). Seul `_10`
  existe pour toutes les stations dans les deux années : c'est lui qui est apparié. L'union vaut 1 sur
  plus de 99.5 % des heures (au moins une station voit toujours la sonde) : le filtre des variables
  quasi constantes l'écarte. La somme par station (0 à 7) reste.
- **Ombres des lunes et occultations (E12).** Les intervalles PHO_/DEI_UMBRA, PENUMBRA et OCC_PHOBOS,
  OCC_DEIMOS sont nuls sur plus de 99.5 % des heures : le filtre les écarte tous. La fraction
  OCC_MARS est égale, au facteur 60 près, à `evtf_OCC_MARS_minutes` déjà présent (écart max 7e-15).
  Le HGB découpe les variables par quantiles : il est insensible à ce facteur. Avec le filtre, E12 ne
  change donc rien, et il n'est pas lancé seul. Il reste testé dans la recette complète (E22), avec le
  test sans filtre (E20). Pairage : un DEI_UMBRA_END rangé avant son START au même instant ouvrait un
  faux intervalle de plusieurs mois. Les paires de plus de 6 h (ombres) ou 24 h (contacts) sont écartées.
- **Flux par face (E16).** `(constante/600) × max(±cos, 0) × clip(1 − umbra − 0.5 pénombre, 0, 1)` est
  exactement `flux_*_eclaire` divisé par 600 : même découpage, mêmes arbres. Seule nouveauté : la face
  sa− (cos sa < 0), soit 0.2 % des mesures en annee1 et 0.7 % en annee2. Écartée elle aussi par le
  filtre sur annee1. Non lancé seul ; testé dans E22.

### E1 — familles corrigées (point 2)

Hypothèse : regrouper les commandes par `famille_variante` (la seule dernière lettre devient `_`, les
chiffres restent) au lieu de `commandes.famille` (qui coupe tout après le numéro) distingue des
familles fusionnées à tort (AOOOF01_0 / AOOOF01_1…) et aide le boosting.
Modification : `"familles_variante": true` (dmopfam_* et plan_* ; vocabulaire ≥ 50 par période).
Run `e1_familles`.

| Run | Seul dir. | Seul inv. | Mél. dir. | Mél. inv. | Moy. 4 | Δ | 2551 d/i | 2451 d/i | 2851 d/i |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| référence | 0.085825 | 0.082774 | 0.077872 | 0.081238 | 0.079555 | — | 20.67/12.98 | 14.89/14.51 | 5.91/6.10 |
| E1 | 0.085395 | 0.083265 | 0.077776 | 0.081419 | 0.079598 | +0.000043 | 20.64/12.98 | 14.92/14.51 | 5.91/6.10 |

GBM seul, plus forts changements de contribution (×1e-4) : directe 2561 −0.77, 2551 −0.41,
2451 +0.37 ; inverse 2561 +0.60, 2881 +0.20. Les trois circuits suivis bougent à peine.

**Décision : rejet.** Le gain en directe est compensé par une perte en inverse, et le mélange ne baisse
pas. Résultat reproduit (une seule graine du boosting, déterministe). La correction reste disponible
et sera retestée dans la recette complète (E22) et avec le retrait des comptes par lettre (E2b).

### Vague B (E2–E6), depuis l'état accepté

Lancée sur Kaggle. Incident : dans la première soumission, l'expression `retirer` d'E2a et E2b avait
été échappée deux fois dans le JSON (`\\\\w`) et ne retirait rien. Ces deux runs sont donc des doublons
de la référence et d'E1 : ils sont ignorés, et E2 est relancé avec `\\w` (vérifié : 10 colonnes retirées).
E3 à E6 ne sont pas touchés.

### E22 — recette complète, lancée sans attendre les essais isolés (demande de l'utilisateur)

Hypothèse : les ingrédients des points 2 à 8 peuvent ne servir qu'ensemble (E1 seul ne gagne pas).
Run `e22a_recette` : familles corrigées, sans comptes par lettre ni colonnes distance Terre et angle
Soleil–Mars–Terre ; planning avec changements de variante, récence au début de l'heure (720 h, log1p),
préfixes, paires de la liste ; soleil « interpole » (moyenne des cosinus, interpolation, écart-type de
sa), ltdata linéaire ; géométrie exacte, altitude, ombres, contacts ; décalages ±1–3 h, historiques
ciblés, face sa− ; EMA gardées ; réglage D (arrêt anticipé), embargo 0, sans filtre de fréquence.
`e22b_recette_sans_ema` : idem sans les EMA. Recalage sur power_init : mesuré avec et sans, à la
prédiction (local). Session Kaggle CPU (fente 3) : les deux fentes GPU étaient prises, et le boosting
n'utilise pas le GPU (accord de l'utilisateur).

Résultats E3–E6 (prédictions refaites en local à partir des modèles Kaggle) :

| Run | Seul dir. | Seul inv. | Mél. dir. | Mél. inv. | Moy. 4 | Δ | 2551 d/i | 2451 d/i | 2851 d/i | Décision |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| référence | 0.085825 | 0.082774 | 0.077872 | 0.081238 | 0.079555 | — | 20.67/12.98 | 14.89/14.51 | 5.91/6.10 | — |
| E3 changements de variante | 0.085599 | 0.082514 | 0.077794 | 0.081212 | 0.079503 | −0.000053 | 20.60/12.90 | 14.88/14.48 | 5.97/6.11 | **garder (b)** |
| E4 récence au début, 720 h, log1p | 0.086278 | 0.082646 | 0.077873 | 0.081193 | 0.079533 | −0.000022 | 21.13/12.90 | 14.81/14.53 | 5.88/6.07 | rejet |
| E5 préfixes (comptes, récence) | 0.085563 | 0.083017 | 0.077738 | 0.081216 | 0.079477 | −0.000078 | 20.60/13.05 | 14.86/14.52 | 5.91/6.13 | rejet (inverse seul +0.00024) |
| E6 paires de la liste | 0.085825 | 0.082774 | 0.077872 | 0.081238 | 0.079555 | 0 | idem | idem | idem | sans effet |

- E3 : le boosting seul baisse dans les deux directions (−0.00023, −0.00026), le mélange un peu
  (règle b). Gain faible, mais dans le sens attendu : la date d'un changement de variante sert sans
  l'identité de la lettre.
- E4 : la nouvelle convention dégrade la directe seule (+0.00045, 2551 surtout) ; le mélange bouge à peine.
- E5 : meilleur mélange de la vague (−0.00008), mais l'inverse seul monte : la règle (a) demande
  −0.0003 et la règle (b) deux baisses. À reconsidérer dans la recette complète.
- E6 : prédictions identiques au bit près. Les paires PPNS/PPNE, PDNS/PDNE, UPBS/UPBE et UDBS/UDBE
  (au plus 11 périodes par an) sont écartées par le filtre de fréquence ; MOCS/MOCE et PENS/PENE
  étaient déjà apprises.

**Nouvel état accepté** : `{"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"], "plan": {"changements": true}}`
(moyenne des 4 RMSE : 0.079503).

Incident d'attente : la boucle `while pgrep -f "lancer_kaggle.py --fente"` se trouvait elle-même
(le motif figure dans sa propre ligne de commande) et ne s'arrêtait jamais. Les runs étaient finis
depuis plusieurs minutes. Attente désormais sur le PID du lanceur.

### E22 — résultats (prédictions refaites en local)

Entraînement Kaggle CPU : 3779 s pour les deux directions en parallèle, par variante (réglage D).

| Run | Seul dir. | Seul inv. | Mél. dir. | Mél. inv. | Moy. 4 | Δ | 2551 d/i | 2451 d/i | 2851 d/i | Décision |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| référence | 0.085825 | 0.082774 | 0.077872 | 0.081238 | 0.079555 | — | 20.67/12.98 | 14.89/14.51 | 5.91/6.10 | — |
| E22a recette, EMA gardées | 0.082144 | 0.082203 | 0.077341 | 0.081267 | 0.079304 | −0.000251 | 20.58/12.75 | 14.76/14.61 | 5.93/6.10 | (b) |
| **E22b recette sans EMA** | **0.077775** | **0.080954** | **0.076677** | **0.081105** | **0.078891** | **−0.000664** | 20.39/12.59 | 14.68/14.44 | 5.81/6.03 | **garder (a)** |
| E22a sans recalage | 0.082944 | 0.090313 | 0.077488 | 0.083272 | 0.080380 | +0.000825 | | | | rejet |
| E22b sans recalage | 0.077891 | 0.089218 | 0.076715 | 0.083104 | 0.079909 | +0.000354 | | | | rejet |

- Les ingrédients ne servent qu'ensemble : seul, chacun gagne au plus 0.00008 (E3–E6), mais réunis
  ils font −0.00066 sur le mélange. Le GBM seul passe de 0.0858 à 0.0778 en directe (−0.0080) et de
  0.0828 à 0.0810 en inverse.
- Les EMA nuisent dans la recette (E22a contre E22b : +0.0044 en directe seule). Elles avaient aidé
  le boosting B_V2 : leur effet dépend du reste des variables.
- Le recalage sur power_init reste nécessaire : sans lui, l'inverse seul perd environ 0.008.
- GBM seul, contributions MSE (×1e-4) : directe 2551 −7.11, 2451 −1.98, 2561 −1.62, 2531 −0.81 ;
  inverse 2451 −0.99, 2551 −0.94, 2532 +0.70. Dans le mélange, la directe gagne sur 2561, 2531,
  2551 et 2451 ; l'inverse perd sur 2532 (+0.74), que le spécialiste 0.6 × GBM traite à part.
- Arrêt anticipé (réglage D) : médiane de 10 itérations par circuit (circuits presque constants),
  600 pour 2551, 2531 et 2532, 449 pour 2451, 566 pour 2851.

**Nouvel état accepté (R)** : options de `e22b_recette_sans_ema`, recalage conservé, poids 0.3.
Prochaine étape : ablations depuis R, un ingrédient retiré à la fois, pour mesurer la contribution de
chaque famille.

### Vague C — essais isolés depuis l'état E3 (avant E22)

Lancés avant le résultat d'E22. Chaque variante change une seule chose par rapport à l'état E3
(boosting B_V2 + planning + EMA + changements de variante). Ils mesurent l'effet de chaque ingrédient
seul. L'état accepté est désormais la recette R : ces décisions servent de diagnostic, sans
changer l'état. Prédictions refaites en local. Référence : `e3_changements` (0.079503).

| Run | Seul dir. | Seul inv. | Mél. dir. | Mél. inv. | Moy. 4 | Δ | 2551 d/i | 2451 d/i | 2851 d/i | Règle |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| E2a sans comptes par lettre | 0.085600 | 0.082518 | 0.077794 | 0.081212 | 0.079503 | +0.000001 | 20.60/12.90 | 14.88/14.48 | 5.97/6.11 | rejet |
| E2b E2a + familles corrigées | 0.085398 | 0.082851 | 0.077757 | 0.081324 | 0.079541 | +0.000038 | 20.57/12.87 | 14.95/14.50 | 6.02/6.12 | rejet |
| E7 soleil « mesures » | 0.085250 | 0.082211 | 0.077770 | 0.081192 | 0.079481 | −0.000022 | 20.59/12.80 | 14.83/14.49 | 5.93/6.10 | (b) |
| E8 soleil « interpole » | 0.085270 | 0.082465 | 0.077767 | 0.081290 | 0.079528 | +0.000026 | 20.59/12.83 | 14.83/14.51 | 5.93/6.12 | (b) |
| E9 ltdata linéaire | 0.085814 | 0.082251 | 0.077847 | 0.081150 | 0.079499 | −0.000004 | 20.64/12.84 | 14.85/14.48 | 6.01/6.09 | rejet |
| E10 péricentres exacts | 0.085179 | 0.082312 | 0.077753 | 0.081157 | 0.079455 | −0.000048 | 20.54/12.82 | 14.91/14.50 | 5.92/6.09 | (b) |
| E11 altitude | 0.085674 | 0.082682 | 0.077827 | 0.081243 | 0.079535 | +0.000033 | 20.58/12.90 | 14.89/14.52 | 6.00/6.11 | rejet |
| E13 contacts | 0.085566 | 0.082552 | 0.077784 | 0.081221 | 0.079502 | −0.000000 | 20.59/12.89 | 14.86/14.52 | 5.98/6.12 | rejet |
| E14 décalages ±1–3 h | 0.082823 | 0.082638 | 0.077371 | 0.081497 | 0.079434 | −0.000068 | 20.27/12.83 | 14.65/14.60 | 5.96/6.15 | rejet (inverse) |
| E15 historiques ciblés | 0.086221 | 0.082057 | 0.078106 | 0.081106 | 0.079606 | +0.000103 | 20.72/12.82 | 15.17/14.42 | 6.01/6.11 | rejet |
| E15b sans EMA | 0.088295 | 0.082700 | 0.078177 | 0.081558 | 0.079868 | +0.000365 | 20.80/12.49 | 14.85/14.53 | 6.30/6.09 | rejet |
| E17 sans distance Terre ni angle SMT | 0.084416 | 0.082429 | 0.077685 | 0.081126 | 0.079405 | −0.000097 | 20.59/12.88 | 14.87/14.52 | 5.87/6.10 | (b) |
| E18 réglage D (arrêt anticipé) | 0.087136 | 0.082173 | 0.078086 | 0.081080 | 0.079583 | +0.000080 | 20.70/12.90 | 14.83/14.48 | 5.94/6.09 | rejet |
| E19 embargo 0 | 0.088731 | 0.082495 | 0.078548 | 0.081138 | 0.079843 | +0.000341 | 20.53/12.81 | 14.91/14.55 | 6.25/6.10 | rejet |
| E20 sans filtre de fréquence | 0.085599 | 0.082514 | 0.077794 | 0.081212 | 0.079503 | +0.000000 | 20.60/12.90 | 14.88/14.48 | 5.97/6.11 | rejet |

Lecture :
- Les 10 comptes exacts avec lettre ne servent à rien au boosting (E2a, prédictions presque identiques).
- Isolés, le réglage D, l'embargo 0 et le retrait des EMA dégradent nettement. Pourtant, réunis aux
  autres ingrédients (E22b), ils font partie de la meilleure version : effets d'interaction, à
  démêler par les ablations depuis R.
- E14 (décalages) donne le plus gros gain isolé du GBM seul en directe (−0.0028), mais l'inverse
  perd. E17 est le seul retrait utile (−0.0012 en directe seule).
- E20 : prédictions identiques. Dans cet état, aucune colonne n'était écartée par le filtre.

## Recette GRU (point 9)

Module `dio/recette.py` (modèle `gru_recette`), entraîné sur Kaggle T4 : environ 10 min pour les deux
directions en parallèle, 5 graines, 80 époques. Entrées : 261 variables (v2r + 117 familles corrigées,
sans comptes par lettre). RMSE d'entraînement en fin de parcours : 0.054 à 0.059 en simulation libre.
Prédictions refaites en local.

| GRU (graines 0–4) | Directe | Inverse | 0–30 j / 30–180 j / > 180 j (inverse) | Mélange 0.3 avec R |
|---|---:|---:|---|---:|
| GRU actuel (`p1_gru_g04`) | 0.079062 | 0.083579 | 0.071 / 0.098 / 0.079 | 0.07886 |
| G1 recette, réinjection | 0.085289 | 0.106174 | 0.126 / 0.115 / 0.102 | 0.08714 |
| G1 + recalage power_init (prédiction) | 0.085442 | 0.099579 | | 0.08496 |
| G2 sans réinjection | 0.081458 | 0.092440 | 0.125 / 0.113 / 0.083 | 0.08203 |
| G2 sans réinjection + recalage | 0.081383 | 0.084537 | | 0.07957 |

- **Réinjection : nuisible** (G1 contre G2 : +0.004 en directe, +0.014 en inverse), dans le même
  sens que l'ablation donnée par l'utilisateur.
- L'erreur de l'inverse vient du début de période, et surtout de NPWD2531 (contribution 33.9e-4 contre 11.0e-4
  pour le GRU actuel). La recette n'a pas de correction de niveau tirée de power_init. L'option
  `recalage` de `predire` (même règle que le boosting : écart moyen entre power_init et la simulation
  libre sur ces heures, demi-vie 1440 h) réduit l'erreur, sans rattraper le GRU actuel.
- **Décision : recette GRU rejetée comme remplaçant du GRU actuel.** Confirmé sur les graines 5–9
  pour G1 : moyenne 0.097036 sans recalage, 0.093989 avec (graines 0–4 : 0.095731 et 0.092510).
- **Hypothèse non confirmée** : diversité. Avec 0.3 × G2 recalé dans la part GRU (mélange
  0.7 × (0.7 GRU actuel + 0.3 G2) + 0.3 R) : 0.07886 → 0.07827, les deux directions baissent
  (0.0768 → 0.0760, 0.0809 → 0.0805). Poids choisi sur la validation, un seul groupe de graines :
  G2 graines 5–9 lancé pour vérifier sur les 4 RMSE.

### Diversité GRU actuel + G2 (confirmée)

G2 graines 5–9 (sans réinjection, recalage à la prédiction) : 0.081255 / 0.084212. Mélange
`0.7 × ((1 − a) × GRU actuel + a × G2 recalé) + 0.3 × GBM`, chaque groupe de graines du GRU actuel
apparié au groupe de G2 de mêmes graines :

| a (part de G2 dans la part GRU) | 0 | 0.1 | 0.2 | **0.3** | 0.4 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| avec GBM installé, moy. 4 | 0.07956 | 0.07926 | 0.07903 | 0.07888 | 0.07881 | 0.07881 |
| avec GBM R, moy. 4 | 0.07889 | 0.07862 | 0.07842 | **0.07830** | 0.07825 | 0.07828 |
| GBM R, directe (graines 0–4 / 5–9) | 0.07680 / 0.07655 | | | 0.07602 / 0.07599 | | |
| GBM R, inverse (graines 0–4 / 5–9) | 0.08092 / 0.08129 | | | 0.08052 / 0.08065 | | |

a = 0.3 a été fixé sur les graines 0–4, avant le calcul des graines 5–9, qui confirment le gain :
les 4 RMSE baissent. **Gardé (règle a) : −0.00059 sur le mélange avec R.** Total depuis la
référence : 0.079555 → 0.07830 (−0.00126). Le poids 0.3 du GBM reste inchangé. a = 0.4 n'apporte
que −0.00005 de plus : non retenu (décision prise avant mesure, plateau).

Installation à faire : G2 entraîné sur annee1 + annee2 (lancé sur Kaggle T4, `runs/final_recette_g2`),
intégré à `dio/ensemble.py` comme troisième composant, puis revalidation des spécialistes.

### Validation de l'ensemble complet (prédiction seule, `predict.py` → `evaluate.py`)

Dossiers `runs/int_val_*` : GRU de validation (graines 0–4 ou 5–9) + GBM R + G2 de mêmes graines
(`ensemble.json` : `{"poids_gbm": 0.3, "part_recette": 0.3}`), avec ou sans les spécialistes de
`sp31_val_*` (inchangés).

| | Directe 0–4 | Inverse 0–4 | Directe 5–9 | Inverse 5–9 | Moy. 0–4 | Moy. 5–9 |
|---|---:|---:|---:|---:|---:|---:|
| Modèle livré, sans spécialistes (`p3_val`) | 0.077975 | 0.081042 | 0.077769 | 0.081435 | 0.079509 | 0.079602 |
| Nouveau, sans spécialistes | 0.076019 | 0.080522 | 0.075988 | 0.080653 | 0.078270 | 0.078321 |
| Modèle livré, avec spécialistes (`sp31_val`) | 0.076748 | 0.079111 | 0.076533 | 0.079497 | 0.077930 | 0.078015 |
| **Nouveau, avec spécialistes** | **0.075344** | **0.078703** | **0.075226** | **0.078900** | **0.077024** | **0.077063** |

Les quatre RMSE baissent, avec ou sans spécialistes. Les spécialistes gardent leur apport (−0.0012
à −0.0013) avec le nouveau GBM, alors que NPWD2532 utilise 0.6 × GBM.

### Ablations depuis R (un ingrédient retiré à la fois) — session 5

Référence : `e22b_recette_sans_ema` (0.078891). « Δ » : effet du **retrait** (positif = l'ingrédient aide).

| Ablation | Seul dir. | Seul inv. | Moy. 4 | Δ | Lecture |
|---|---:|---:|---:|---:|---|
| avec filtre de fréquence | 0.077775 | 0.080954 | 0.078891 | 0 | prédictions identiques : les colonnes écartées ne servaient à aucun arbre |
| cosinus de l'angle moyen (sans `soleil`) | 0.078619 | 0.080826 | 0.078904 | +0.000013 | moyenne des cosinus utile en directe seule (−0.0008) |
| avec comptes par lettre | 0.077800 | 0.080969 | 0.078894 | +0.000003 | sans effet |
| sans altitude | 0.077760 | 0.080934 | 0.078887 | −0.000004 | retrait permis par la règle (b), gain négligeable |
| récence en fin d'heure, 168 h | 0.077604 | 0.081270 | 0.078932 | +0.000042 | la convention de début d'heure aide (inverse) |
