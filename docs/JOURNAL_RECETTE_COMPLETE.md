# Journal — recette complète « S » (à partir du 12 novembre 2026, soir)

Programme en 10 points fourni par l'utilisateur : référence figée, recette de features exacte et distincte de la
préparation actuelle (`dio/caracteristiques.py`), GBM et GRU à entrées séparées, recette complète puis ablations,
complémentarité, choix. Tout est calculé **sur la machine locale** (demande de l'utilisateur pour cette session :
pas de Kaggle). Branche locale `recette-complete` (depuis `recette`), rien n'est poussé.

## Reprendre ici

- Branche : `recette-complete`.
- Jobs : voir `runs/rc_lot*.sh` / `.log` / `.pid` (attente par `kill -0 PID`).
- Statut : voir « État » en bas.

## Matériel et temps

AMD Ryzen AI 7 350 (8 cœurs, 16 threads), 30 Go, pas de GPU utilisable. Python 3.13, torch 2.15.0 (CPU),
scikit-learn 1.9.1, pandas 2.3.3, numpy 2.3.5 (liste complète : `runs/rc0_reference/manifeste.json`).
Les temps de cette session sont tous mesurés sur cette machine ; ceux des journaux précédents viennent de Kaggle.

## Critères de décision (fixés avant les mesures)

- Ablations et composants, sans spécialistes : règle du dépôt (`docs/JOURNAL_RECETTE.md`), référence = recette complète S.
  (a) moyenne des RMSE du mélange en baisse ≥ 0.0003 et aucune direction en hausse > 0.0002 ;
  (b) composant changé seul en baisse dans les deux directions et mélange ≤ +0.0001.
  Mélange des ablations : 0.5 × GBM + 0.5 × GRU, poids fixe. Outil : `outils/comparer_complete.py`.
- Installation (modèle complet avec spécialistes, `predict.py` → `evaluate.py`, 4 RMSE directe/inverse × graines
  0–4/5–9) contre le livré 0.076735 : règle (a) ou les 4 RMSE en baisse ; et validation chronologique pas pire que la
  référence de plus de 0.0002. Sinon, le modèle livré reste.
- Poids et parts choisis sur les graines 0–4, confirmés sur 5–9.

## Point 1 — référence figée

- `runs/rc0_reference/manifeste.json` : commit `070ae7f`, sha256 des 60 sources suivies, des 13 fichiers de `poids/`,
  de 18 prédictions de référence, bibliothèques (`uv pip freeze`), machine, configurations (ensemble, modèle,
  spécialistes).
- **Inférence du livré reproduite en local** (`runs/rc0_livre_g04|g59`, copies de `s51_val_*`,
  `valider.py --sans-entrainement`) :

| | Directe | Inverse | Prédiction (s, local) |
|---|---:|---:|---|
| graines 0–4 | 0.075046 | 0.078474 | 46 / 51 |
| graines 5–9 | 0.075078 | 0.078343 | 44 / 54 |
| **Moyenne des 4** | | **0.076735** | identique au journal |

- Référence sans spécialistes, mélange 0.5 × GBM R + 0.5 × GRU mixte sans lettres (sans GRU recette) : 0.077971 ;
  installé sans spécialistes (avec GRU recette 0.3) : 0.07768 (`docs/JOURNAL_RECETTE.md`).

## Écarts formule par formule entre la spec et le boosting R (avant ce programme)

| Spec | R (`gbm_sacs` + `contexte`) |
|---|---|
| SAAF : moyennes arithmétiques, std pandas ddof=1, NaN→0 si heure observée | moyennes pondérées par la durée (maintien 60 s), écart-type de population pondéré |
| LTDATA au début de l'heure | milieu de l'heure |
| récences : occurrences quantifiées à l'heure | instants exacts (fractions d'heure) |
| changements : ≥ 2 variantes à ≥ 50 par période, union ; quantifiés ; 0 dans l'heure | seuil 20 sur l'ensemble ; instants exacts ; pas de 0 |
| paires : union des intervalles | somme plafonnée à 1 |
| altitude : moyenne des 4 log | moyenne des 4 altitudes |
| historiques : umbra seule, flux_incident = solar × shade | umbra + 0.5 pénombre, somme des flux des faces |
| décalages : 10 séries (dont log-altitude) | 9 séries |
| vocabulaires de chaque validation appris sur son entraînement | schéma `prepare.py` (types FTL, 10 commandes, médianes) appris sur annee1, aussi en inverse |
| pas de batteries glissantes ni EMA | fenêtres v2r 3–24 h passées/futures, moyennes `physiques` 6/24/72 h |

Conformes et réutilisés : `commandes.famille_variante`, réglage HGB « D » (= spec), 5 sacs contigus sans embargo ni
filtre, géométrie aux péricentres exacts (`contexte.geometrie`), garde-fous d'appariement (`contexte.intervalles`,
contacts au seuil `_10` seulement), architecture et boucle d'entraînement de `dio/recette.py`.

## Point 2 à 6 — `dio/caracteristiques.py`

Conventions : docstring du module (grille, SAAF, LTDATA, familles, récences, changements, paires, événements, FTL,
géométrie, altitude, thermique, décalages). Tests : `tests/test_caracteristiques.py` (11 tests, exemples de la spec).
Vocabulaire appris sur annee1 + annee2 (pour information) : 149 familles, 23 préfixes, 29 familles à changements,
16 types FTL. Construction d'une année : environ 2 s.

## Points 7 et 8 — recette complète S (mesures locales)

Commandes :
`python outils/valider.py --nom rc1_gbm_S --modele gbm_sacs --options '{"config":"D","jeu":"S","sacs":5,"embargo_h":0,"filtre":false}'`,
`--nom rc1_gru_A --modele gru_recette --options '{"variables":"S","reinjection":false}'`,
`--nom rc1_gru_B --modele gru_recette --options '{"variables":"S","reinjection":true,"premier_segment_verite":true}'`,
recalage : `--sans-entrainement --options-prediction '{"recalage": 1.0}'`.

| Composant (graines 0–4) | Colonnes dir / inv | Directe | Inverse | Entraînement (2 directions en parallèle, seul sur la machine) |
|---|---|---:|---:|---|
| GBM R (référence, Kaggle) | 689 | 0.077775 | 0.080954 | 3779 s sur Kaggle CPU |
| **GBM S** (recalage, défaut) | 673 / 794 | 0.079094 | 0.082357 | 988 s |
| GRU mixte sans lettres (livré, graines 0–4) | 268 | 0.079030 | 0.083565 | (Kaggle T4) |
| GRU recette G2, anciennes entrées, recalé | 261 | 0.081383 | 0.084537 | (Kaggle T4) |
| **GRU S A** sans réinjection | 160 / 192 | 0.078472 | 0.105959 | 137 s |
| GRU S A, recalé | | 0.078469 | 0.098340 | |
| **GRU S B** avec réinjection | | 0.079671 | 0.095853 | 147 s |
| GRU S B, recalé | | 0.079711 | 0.087938 | |

Mélanges 0.5 × GBM + 0.5 × GRU (graines 0–4 ; `runs/rc1_*/bilan.json`) :

| Mélange | Directe | Inverse | Moyenne |
|---|---:|---:|---:|
| GBM R + GRU mixte (référence de comparaison) | 0.076090 | 0.079919 | 0.078004 |
| GBM S + GRU S A recalé | 0.074947 | 0.085789 | 0.080368 |
| GBM S + GRU S A sans recalage | 0.074932 | 0.088896 | 0.081914 |
| GBM S + GRU S B recalé | 0.075137 | 0.082038 | 0.078587 |
| GBM R + GRU S A recalé | 0.074774 | 0.084537 | 0.079656 |
| GBM S + GRU mixte | 0.076219 | 0.080482 | 0.078351 |

- GBM S seul est moins bon que R dans les deux directions (+0.0013 / +0.0014) ; pertes sur NPWD2551 en directe
  (+1.5e-4 de contribution MSE), NPWD2451 et NPWD2532 en inverse (+1.1e-4, +1.05e-4). Prédiction 11 s contre ~40 s.
  Itérations effectives (`iterations.json`) : 600 (plafond) pour NPWD2372, 2531, 2532, 2551 dans tous les sacs ;
  10 à 600 ailleurs.
- GRU S A : meilleur GRU seul en directe (0.0785 contre 0.0790), mais inverse très mauvais, même recalé : l'erreur
  monte dans la seconde moitié de l'année (quarts : 0.091, 0.088, 0.107, 0.105) sur NPWD2551 et 2451.
- Réinjection (B) : moins bonne en directe (+0.0012), meilleure en inverse ; ne corrige pas la dérive.
- Recalage power_init : aide le GRU en inverse (−0.0076 pour A, −0.0079 pour B), neutre en directe ; gardé.

### Diagnostic de la dérive inverse du GRU S

Décalage de distribution des entrées GRU entre annee2 (entraînement inverse) et la seconde moitié d'annee1 :
`geo_periode_h` (période orbitale entre péricentres) est quasi constante en annee2 et se trouve en moyenne à
**−43 écarts-types** en annee1 (68 % des heures hors de l'intervalle d'entraînement) ; après standardisation, elle
sature à −10. Toutes les autres entrées restent sous 1.3 écart-type en moyenne.
Ablations GRU de diagnostic (A, graines 0–4, recalé ; référence GRU S A : 0.078469 / 0.098340) :

| Retrait | GRU directe | GRU inverse | Mélange moy. | Δ mélange |
|---|---:|---:|---:|---:|
| contacts | 0.078649 | 0.100858 | 0.080870 | +0.00050 |
| ftl_debuts | 0.078415 | 0.094667 | 0.079679 | −0.00069 |
| géométrie (péricentres, période, phase) | 0.084627 | 0.085419 | 0.079080 | −0.00129 |
| ombres des lunes et occultations Phobos/Deimos | 0.078534 | 0.097840 | 0.080391 | +0.00002 |
| **période orbitale seule** | **0.086294** | **0.085554** | 0.079421 | −0.00095 |

La période orbitale porte à la fois le gain en directe (−0.0078) et la dérive en inverse (+0.0128) : c'est un
indicateur de régime d'orbite qui sort du domaine appris. Les retraits mineurs varient de ±0.003 en inverse : le bruit
d'entraînement du GRU en inverse est grand (une seule série de graines) ; toute adoption GRU est confirmée sur 5–9.

## Point 9 — ablations depuis la recette complète (un bloc à la fois)

Référence : GBM S + GRU S A recalé, mélange 0.5 (0.080368). Δ = mélange de l'essai − référence. Décision par la règle.

| Bloc changé | Composant | Seul directe | Seul inverse | Mélange moy. | Δ | Décision |
|---|---|---:|---:|---:|---:|---|
| SAAF pondéré par la durée | GBM | 0.078878 | 0.082600 | 0.080342 | −0.000026 | rejet |
| SAAF pondéré par la durée | GRU | 0.078080 | 0.101302 | 0.080861 | +0.000493 | rejet |
| LTDATA au milieu de l'heure | GBM | 0.079027 | 0.082549 | 0.080390 | +0.000022 | rejet |
| LTDATA au milieu de l'heure | GRU | 0.078354 | 0.097985 | 0.080270 | −0.000098 | (b), bruit possible |
| historiques formule R (umbra + 0.5 p, somme des faces) | GBM | 0.079137 | 0.081665 | 0.080272 | −0.000096 | rejet |
| sans altitude ni ses décalages | GBM | 0.078913 | 0.082450 | 0.080318 | −0.000050 | rejet |
| sans altitude | GRU | 0.078233 | 0.096756 | 0.079999 | −0.000369 | (a), bruit possible |
| récence en instants exacts | GBM | 0.078557 | 0.082177 | 0.080098 | −0.000270 | **garder (b)** |
| **normalisation min/max d'entraînement + clip ±5** | GRU | **0.078319** | **0.085929** | **0.078202** | **−0.002166** | **garder (a)(b)** |
| réinjection (B contre A), standard | GRU | 0.079711 | 0.087938 | 0.078587 | −0.001781 | voir ci-dessous |

- **Normalisation** : borner chaque entrée aux min/max de l'entraînement supprime l'extrapolation de la période
  orbitale ; l'inverse passe de 0.0983 à 0.0859 sans perte en directe. Confirmé graines 5–9 :
  0.078848 / 0.085595 (`rc4_gruA_mm_g59`).
- **Réinjection avec bornage** (`rc4_gruB_mm`) : 0.079586 / 0.091314, moins bon que A borné dans les deux
  directions. **A (sans réinjection) retenu.** La réinjection n'apporte rien une fois l'extrapolation corrigée.
- A borné sans `ftl_debuts` : 0.078345 / 0.086035 (sans effet ; retrait non retenu).
- Les gains GRU « LTDATA milieu » et « sans altitude » viennent de l'inverse non borné, très bruité ; leur réunion
  avec le bornage est mesurée à part (`rc4_gruA_mm_reunion`).

## Point 10 — croisement et choix (graines 0–4 pour choisir, 5–9 pour confirmer)

`python outils/croiser.py --gbm e22b_recette_sans_ema rc1_gbm_S --nom rc5_croisement` (sans spécialistes). Mélange
w × GBM + (1 − w) × [(1 − a − b) GRU mixte + a G2 + b GRU S A borné], recalage partout.

| GBM | w | a (G2) | b (GRU S) | g04 dir | g04 inv | g59 dir | g59 inv | moy. 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 0.5 | 0.3 | 0 | 0.075570 | 0.079802 | 0.075640 | 0.079723 | 0.077683 (livré) |
| **R** | **0.5** | **0** | **0.5** | **0.074790** | **0.079497** | **0.074741** | **0.079530** | **0.077139** |
| R | 0.5 | 0 | 0.7 | 0.074630 | 0.079744 | 0.074615 | 0.079781 | 0.077193 |
| R | 0.4 | 0 | 0.5 | 0.074711 | 0.079674 | 0.074657 | 0.079718 | 0.077190 |
| S | 0.4 | 0 | 0.5 | 0.074756 | 0.080256 | 0.074692 | 0.080267 | 0.077493 |

- Le meilleur GBM reste R : le GBM S est moins bon seul et en mélange. Le meilleur GRU complémentaire est le GRU S A
  borné, qui remplace le GRU recette G2 dans la part GRU (part 0.5). Poids GBM 0.5 inchangé.
- Choix fait sur les graines 0–4 ; graines 5–9 : les 2 RMSE baissent aussi (0.07474 / 0.07953 contre 0.07564 / 0.07972).

### Bout en bout avec spécialistes (`predict.py` → `evaluate.py`)

`python outils/candidat.py --nom rc6_val_g04 --base s51_val_g04 --recette rc3_gruA_norm --part 0.5` (idem g59 avec
`rc4_gruA_mm_g59`), puis `valider.py --sans-entrainement`. Spécialistes du livré inchangés.

| | Directe 0–4 | Inverse 0–4 | Directe 5–9 | Inverse 5–9 | Moyenne |
|---|---:|---:|---:|---:|---:|
| Livré (`rc0_livre_*`) | 0.075046 | 0.078474 | 0.075078 | 0.078343 | 0.076735 |
| **Candidat C1** (`rc6_val_*`) | **0.074401** | **0.078196** | **0.074367** | **0.078189** | **0.076288** |

Δ = −0.000447, les 4 RMSE baissent : critère d'installation rempli sur les deux validations temporelles.
Porte restante : validation chronologique (`rc7_chrono_*`).
