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
  `{"config": "B", "jeu": "V2", "ajouts": ["planning", "ema"]}` (boosting installé).
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

En cours : `r0_gbm`, boosting installé réentraîné sur Kaggle (fente 1).

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
