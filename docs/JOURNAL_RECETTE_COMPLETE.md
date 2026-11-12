# Journal — recette complète « S » (à partir du 12 novembre 2026, soir)

Programme en 10 points fourni par l'utilisateur : référence figée, recette de features exacte et distincte de la
préparation actuelle (`dio/caracteristiques.py`), GBM et GRU à entrées séparées, recette complète puis ablations,
complémentarité, choix. Tout est calculé **sur la machine locale** (demande de l'utilisateur pour cette session :
pas de Kaggle). Branche locale `recette-complete` (depuis `recette`), rien n'est poussé.

## Reprendre ici

- Branche : `recette-complete`.
- Jobs : voir `runs/rc_lot*.sh` / `.log` / `.pid` (attente par `kill -0 PID`).
- Statut : programme terminé, modèle livré conservé (voir « Décision finale »). Aucun job en cours.

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

### Validation chronologique (porte d'installation)

`python outils/faire_chrono.py` puis `valider.py --direction chrono` : entraînement annee1 + première moitié d'annee2
(8244 h), test sur la seconde moitié (7 jours de power_init, 8075 h notées). Composants réentraînés en local.
Règle précisée avant le calcul des mélanges chrono : parmi les candidats qui remplissent le critère temporel,
seuls ceux dont le mélange chrono n'est pas pire que celui de la structure livrée de plus de 0.0002 sont
éligibles ; le choix entre eux se fait sur les validations temporelles (graines 0–4), jamais sur la chrono.

| GRU seul (chrono, recalé) | RMSE | NPWD2551 / 2451 (×1e-4) | biais 2451 |
|---|---:|---|---:|
| G2 (anciennes entrées) | 0.082744 | 23.56 / 16.11 | +0.014 |
| GRU S A borné | 0.088092 | 29.01 / 21.11 | +0.111 |

Le GRU S perd surtout sur NPWD2451 au deuxième quart du test (biais +0.11), sans entrée hors domaine (toutes à
moins de 1.1 écart-type de l'entraînement). Hypothèse : la période orbitale, très informative en directe, sert
d'identifiant d'époque (une valeur ≈ une phase de mission) ; vérification par `rc7_chrono_gruA_mm_sans_periode`.

### Ablations et réintroductions GBM, rapportées à la référence GBM S + GRU S A borné (0.078202)

| Essai GBM (depuis S) | Seul directe | Seul inverse | Mélange moy. | Δ | Décision |
|---|---:|---:|---:|---:|---|
| SAAF pondéré par la durée | 0.078878 | 0.082600 | 0.078197 | −0.000005 | rejet |
| LTDATA au milieu de l'heure | 0.079027 | 0.082549 | 0.078223 | +0.000021 | rejet |
| historiques formule R | 0.079137 | 0.081665 | 0.078116 | −0.000087 | rejet |
| sans altitude ni ses décalages | 0.078913 | 0.082450 | 0.078181 | −0.000021 | rejet |
| récence en instants exacts | 0.078557 | 0.082177 | 0.077940 | −0.000262 | **garder (b)** |
| sélection : + distance Terre et angle SMT | 0.083103 | 0.082397 | 0.078190 | −0.000012 | rejet (directe seule +0.004) |
| réintro fenêtres v2r | 0.078893 | 0.082531 | 0.078281 | +0.000079 | rejet |
| réintro moyennes passées `physiques` | 0.078914 | 0.081295 | 0.078006 | −0.000196 | **garder (b)** |
| réintro harmoniques d'orbite + EMA lentes | 0.079214 | 0.082359 | 0.078008 | −0.000194 | rejet (directe seule monte) |
| réintro colonnes brutes de prepare.py | 0.083594 | 0.082605 | 0.078313 | +0.000110 | rejet |

Les conventions de la spec (SAAF arithmétique, LTDATA début d'heure, historiques umbra seule) ne changent presque
rien au GBM (|Δ| ≤ 0.0001) ; la récence quantifiée est moins bonne que la récence exacte. Réunion des deux adoptions :
`rc2_gbm_reunion`. GRU, réintroductions : `rc9_gruA_mm_*`.

### GRU S borné : réintroductions (point 9, blocs du modèle actuel, un par un)

| GRU S A borné + | Directe | Inverse | Mélange moy. (avec GBM S) | Δ | Chrono seul | Décision |
|---|---:|---:|---:|---:|---:|---|
| (rien, référence) | 0.078319 | 0.085929 | 0.078202 | — | 0.088092 | |
| fenêtres v2r | 0.080333 | 0.083129 | 0.077995 | −0.000208 | 0.090023 | rejet |
| moyennes passées `physiques` | 0.078884 | 0.081944 | 0.077328 | −0.000874 | 0.089402 | (a) sur les validations temporelles ; échoue en chrono |
| colonnes brutes de prepare.py | 0.078613 | 0.082419 | 0.077601 | −0.000601 | | rejet (directe seule monte) |

GRU S borné + `physiques` est le meilleur GRU seul sur les deux validations temporelles (mixte : 0.0790 / 0.0836).

### Réunion GBM

`rc2_gbm_reunion` (S + récence exacte + `physiques`) : 0.078645 / 0.080970, mélange 0.077780 (Δ −0.000422 contre S,
garder). Toujours derrière R seul en directe (0.077775). Dans la structure livrée (poids 0.5, G2 0.3) : 0.077651 contre
0.077683 pour R (−0.00003, directe 0–4 en hausse) : ne remplace pas R. Chrono du GBM S : 0.081078 contre 0.079899 (R).

### Chrono : mélanges (GBM R, poids 0.5, recalage partout ; `runs/rc7_chrono_melanges.json`)

| Part GRU | RMSE chrono | Δ contre le livré |
|---|---:|---:|
| **livré : 0.7 mixte + 0.3 G2** | **0.077893** | — |
| C1 : 0.5 mixte + 0.5 GRU S borné | 0.079085 | +0.001192 |
| 0.7 mixte + 0.3 GRU S borné | 0.078808 | +0.000915 |
| C2 : 0.5 mixte + 0.5 GRU S borné + physiques | 0.079429 | +0.001536 |
| 0.7 mixte + 0.3 GRU S borné + physiques | 0.079004 | +0.001111 |
| 0.5 mixte + 0.5 GRU S borné + v2r | 0.079127 | +0.001234 |

GRU seuls en chrono : mixte 0.082871, G2 0.082744, GRU S borné 0.088092, sans période 0.100439, + physiques 0.089402,
+ v2r 0.090023. **Tous les candidats avec un GRU S échouent à la porte chronologique** (+0.0009 à +0.0015 > 0.0002).
Retirer la période n'aide pas en chrono (0.100) : la faiblesse vient de l'ensemble des entrées S, pas de la seule période.
Lecture : les entrées S (contexte instantané, sans fenêtres passées/futures ni les variables brutes de prepare.py)
séparent bien annee1 et annee2 mais généralisent moins bien à l'intérieur d'une même année, où les régimes de
NPWD2451/2551 évoluent sans signal net dans le contexte.

Observation non exploitée : sur la chrono, 0.5 mixte + 0.5 G2 fait 0.077627 (−0.00027). Cette part n'était pas dans la
grille fixée d'avance et la chrono ne sert pas à choisir : piste à valider sur les validations temporelles seules.

### Spécialistes (testés sur le modèle livré, qui reste)

| Retrait | Directe 0–4 | Inverse 0–4 | Directe 5–9 | Inverse 5–9 | Moyenne | Δ |
|---|---:|---:|---:|---:|---:|---:|
| aucun (livré) | 0.075046 | 0.078474 | 0.075078 | 0.078343 | 0.076735 | — |
| sans NPWD2551 (boosting propre) | 0.075138 | 0.078541 | 0.075170 | 0.078408 | 0.076814 | +0.000079 |
| sans NPWD2532 (0.6 GBM + 0.4 physique) | 0.075124 | 0.079193 | 0.075146 | 0.079020 | 0.077121 | +0.000386 |
| sans NPWD2531 (porte des marqueurs) | 0.075399 | 0.079023 | 0.075482 | 0.078988 | 0.077223 | +0.000488 |

Les trois spécialistes restent utiles (chaque retrait dégrade les 4 RMSE).

## Décision finale

**Le modèle livré est conservé** (`poids/` inchangé, validation 0.076735 avec spécialistes). Le candidat C1 (part GRU
0.5 mixte + 0.5 GRU S A borné) remplissait le critère temporel (0.076288, les 4 RMSE en baisse) mais dégrade la
validation chronologique de 0.0012 ; la règle fixée avant les mesures l'exclut. Aucune autre combinaison ne passe.

Ce qui est démontré :
- code prêt : `dio/caracteristiques.py` (recette S exacte, testée), GBM `jeu="S"`, GRU `variables="S"`, options
  `normalisation`, `premier_segment_verite`, `retirer`, validation chronologique, outils de bilan ;
- inférence du livré reproduite en local (0.076735) ; entraînement local de R reproduit (`rc0_gbm_R_local`, ci-dessous) ;
- gains démontrés sur les validations temporelles mais **non confirmés en chrono** : GRU S borné (directe 0.0783 contre
  0.0790 pour le mixte), GRU S borné + physiques ;
- résultat robuste et réutilisable : borner les entrées du GRU aux min/max d'entraînement supprime l'extrapolation
  (inverse 0.098 → 0.086) ; la réinjection de la consommation n'aide pas ; la récence quantifiée de la spec est moins
  bonne que la récence exacte ; les conventions SAAF/LTDATA/historiques de la spec sont neutres pour le GBM.

Limites : une seule série de graines pour la plupart des ablations GRU (bruit en inverse de ±0.003 avant bornage) ;
validation chronologique sur une seule coupure ; GRU mixte non réentraîné sur les validations temporelles avec les
nouvelles entrées (seulement sur la chrono, 3748 s sur CPU) ; poids et parts choisis sur les validations elles-mêmes.

## Reproduction

```bash
uv run python -m unittest tests.test_caracteristiques
uv run python outils/faire_chrono.py
uv run python outils/valider.py --nom rc1_gbm_S --modele gbm_sacs --options '{"config":"D","jeu":"S","sacs":5,"embargo_h":0,"filtre":false}'
uv run python outils/valider.py --nom rc3_gruA_norm --modele gru_recette --options '{"variables":"S","reinjection":false,"normalisation":"minmax"}' --options-prediction '{"recalage": 1.0}'
uv run python outils/croiser.py --gbm e22b_recette_sans_ema rc1_gbm_S --nom rc5_croisement
uv run python outils/candidat.py --nom rc6_val_g04 --base s51_val_g04 --recette rc3_gruA_norm --part 0.5
uv run python outils/valider.py --nom rc6_val_g04 --sans-entrainement
uv run python outils/valider.py --nom rc7_chrono_gruA_mm --direction chrono --modele gru_recette --options '{"variables":"S","reinjection":false,"normalisation":"minmax"}' --options-prediction '{"recalage": 1.0}'
```
Tous les runs de cette session : `runs/rc*` (scores, `bilan.json`, `colonnes.json`, `iterations.json`, `train.log`).

## Derniers contrôles (13 nov., 02:50)

- **Entraînement de R reproduit en local** (`rc0_gbm_R_local`, seul sur la machine) : 0.077775 / 0.080954, identiques
  au run Kaggle. Temps, même matériel : R 1284 s, GBM S 988 s (deux directions en parallèle) ; prédiction 14–20 s
  contre 11–12 s. GRU S A : 137 s seul sur la machine.
- GRU S borné + physiques, graines 5–9 (`rc9_gruA_mm_physiques_g59`) : 0.078559 / 0.082284, gain sur les
  validations temporelles confirmé ; ne change pas la décision (porte chronologique).
- `python -m unittest tests.test_caracteristiques tests.test_gbm_sacs tests.test_recette tests.test_ensemble` :
  33 tests OK (dont l'ensemble livré sur `poids/`).
