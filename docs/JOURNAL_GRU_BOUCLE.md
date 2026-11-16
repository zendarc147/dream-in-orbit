# Journal — second GRU « à boucle » dans la part réseau de T3 (à partir du 16 novembre 2026, soir)

Mission fournie par l'utilisateur : ajouter dans la part réseau de la structure T3 un second GRU, entraîné autrement
(il relit sa propre prédiction), et mesurer honnêtement ce que ça change. Pas de nouvelles variables, pas de nouvelle
architecture. Branche locale `recette-complete`, rien n'est poussé. **Tous les entraînements tournent sur Kaggle
`zendarc147` (sessions T4)**, à la demande de l'utilisateur ; en local : assemblages, prédictions avec des poids
existants, scores, tests. L'utilisateur a demandé qu'aucune question ne lui soit posée pendant l'exécution : chaque
choix laissé ouvert est pris ici et noté « choix d'exécution ».

## Reprendre ici

- Outil : `outils/gru_boucle.py` (commandes Kaggle, fusion, assemblage, prédictions, bilan, décision, poids finaux).
- Sessions Kaggle : `runs/gb_K{0,1,2,3}.log` (attente par `kill -0 PID`).
- Statut (16 novembre 2026, soir) : **S2 installé dans `poids/`** par la règle (aucun échec technique) ; T3 dans
  `poids_candidats/structure/T3_g10/` ; S1, S2 à 0.6 et GB-mm final dans `poids_candidats/gru_boucle/`. Aucun job en cours.

## Point de départ (vérifié, pas redécouvert)

- `poids/` = structure T3 à 10 graines, installée à la demande de l'utilisateur alors que la règle de
  `docs/JOURNAL_STRUCTURE.md` l'avait refusée. Part réseau : GRU S borné (G0 : `variables="S"`, `reinjection=false`,
  `normalisation="minmax"`, recalage 1.0 à la prédiction, graines 0–9 fusionnées, `recette/`). Part boosting :
  boosting S (`gbm/`, recalé) ; NPWD2551 : recette B seule (`specialiste/b_NPWD2551/`). Mélange 0.5 / 0.5, puis
  spécialistes NPWD2532 et NPWD2531 du livré. Repli : `poids_repli/` (livré L), `poids_candidats/structure/`.
- Gel (commit `6a0030b`, l'installation de T3 n'était pas encore commitée) : sha256 des 12 fichiers de `poids/`.
  Les 11 fichiers de calcul sont identiques à `poids_candidats/structure/T3_g10/` ; seul `modele.json` diffère
  (métadonnées : la note d'installation de T3 dans `poids/`, celle du livré dans `T3_g10/`, rétablie après l'incident
  des liens durs). Tests : `uv run python -m unittest discover tests` : 112 tests OK, 4 sautés (639 s).

| Fichier de `poids/` | sha256 (16 premiers caractères) |
|---|---|
| ensemble.json | 25fed607868d8dca |
| gbm/gbm_sacs.pkl.xz | b9da1c6795c09bd5 |
| modele.json | 2dc6321958fa5b49 |
| recette/recette.pt | 5c032b865bc75f79 |
| specialiste/b_NPWD2551/info.json | 4aa9297738405888 |
| specialiste/b_NPWD2551/modele_b.pkl.xz | 3664b3e4d9798d7c |
| specialiste/b_NPWD2551/modele.json | 0d4fe111efc13824 |
| specialiste/marqueurs.json | 6a39bc7923d7d1fe |
| specialiste/physique/membre0/monde.pt | 1bd57817cd86eeec |
| specialiste/physique/membre1/monde.pt | dde94b17ed71762f |
| specialiste/physique/membres.json | 6e082f6e039835b9 |
| specialiste/specialistes.json | 1cef688010b8a97d |

- Établi par les journaux : la part boosting de T3 vaut à peu près la part boosting R (c87 : S 0.0715, R 0.0710).
  Le point faible de T3 est sa part réseau : sur c87, 0.0743 (livré) → 0.0945 (GRU S à 10 graines), hausse à 76–81 %
  sur NPWD2551 et 20–24 % sur NPWD2451.
- Le GRU S « B » (avec réinjection) n'a été mesuré qu'une fois en normalisation standard (0.0797 / 0.0879 recalé,
  contre 0.0785 / 0.0983 pour A non borné) et une fois borné (`rc4_gruB_mm` : 0.0796 / 0.0913) ; le jugement « la
  réinjection n'apporte rien une fois l'extrapolation corrigée » n'a jamais été vérifié dans un mélange ni sur les
  coupures glissantes.

## Orientation imposée (décision de l'utilisateur, pas un résultat mesuré)

Ce qui suit est une **orientation imposée** par l'utilisateur. Ce n'est pas une conclusion des validations internes ;
elle n'est ni présentée comme un résultat mesuré ni rejustifiée par de nouvelles ablations.

- La part réseau doit contenir deux GRU de natures différentes : le GRU S actuel, sans boucle, et un GRU qui relit sa
  propre prédiction (réinjection, scheduled sampling jusqu'à p = 0). Un world model qui ne relit jamais son état n'est
  pas ce que l'utilisateur veut livrer ; un ensemble de deux dynamiques différentes est plus robuste à une année
  inconnue qu'un seul GRU affûté.
- Le GRU à boucle est entraîné comme un world model classique :
  - premier segment de chaque flux avec la vérité réinjectée ;
  - normalisation standard (moyenne / écart-type d'entraînement, écart-type < 1e-3 remplacé par 1, clip à ±10), sans
    bornage min/max ;
  - aucun recalage sur power_init à la prédiction : les 7 jours réels servent seulement à réchauffer l'état.
- Structure installée : S2, quel que soit son classement interne, sauf échec technique (règle ci-dessous).
- Le poids global reste 0.5 réseau / 0.5 boosting. Le règlement (`docs/sujet.md` : « Modèle final : un réseau de
  neurones. Les modèles classiques servent de baseline ou de composant uniquement ») interdit de donner plus de la
  moitié au boosting sans l'accord explicite de l'utilisateur.

## Liste fermée des candidats (fixée avant toute mesure)

GRU, 10 graines chacun (0–9 ; deux entraînements de 5 graines fusionnés par `outils/fusionner_recette.py`, comme G0) :

| Nom | reinjection | premier_segment_verite | normalisation | Recalage à la prédiction |
|---|---|---|---|---|
| G0 (installé) | false | false | minmax | 1.0 |
| GB | true | true | standard | 0.0 |
| GB-r (diagnostic, prédiction seule) | true | true | standard | 1.0 |
| GB-mm (diagnostic, un entraînement) | true | true | minmax | 0.0 |

Toutes les autres options de GB sont celles de G0 (défauts de `dio/recette.py`, vérifiés dans la configuration de
`poids/recette/recette.pt`) : `variables="S"`, mêmes `retirer`, taille 128, dropout 0.3, dropout des variables 0.2,
80 époques, 16 flux, segments de 168 h, 15 époques de vérité, bruit de retour 0.05, lr 1e-3, weight decay 1e-3,
montée 0.15. Options d'entraînement de GB : `{"variables": "S", "reinjection": true, "premier_segment_verite": true,
"normalisation": "standard"}` ; GB-mm : idem avec `"minmax"`. À la prédiction, `dio/recette.predire` réinjecte la
vérité des 7 jours de power_init pour réchauffer l'état, puis simule librement (inchangé).

Structures : 0.5 × part réseau + 0.5 × part boosting de T3 ; spécialistes 2531 et 2532 inchangés.

| Nom | Part réseau |
|---|---|
| S0 | G0 seul (= T3 installé) |
| S1 | GB seul |
| S2 | 0.5 × G0 + 0.5 × GB |

Variante de rapport, jamais installée sans accord de l'utilisateur : S2 avec un poids boosting de 0.6 (réseau 0.4),
mesurée sur toutes les validations (`predict.py --options '{"poids_gbm": 0.6}'`).

Aucun autre candidat : pas de nouvelle variable, pas de nouveau spécialiste, pas de poids par circuit, pas d'autre
modèle d'arbres, pas d'adaptation par gradient sur les 7 jours réels.

Code ajouté (défauts inchangés) : `dio/ensemble.py` lit dans `ensemble.json` la clé `recette2` (poids du second GRU
dans la part réseau ; absente = 0, rien ne change) et `recalage_recette2` (absente = 0.0), dossier `recette2/` ;
`recette2 = 1` : `recette/` n'est pas lu. Avec l'option `parts`, les deux GRU sont aussi écrits séparément
(`gru_recette.csv`, `gru_recette2.csv`). `outils/candidat.py` : `--recette2`, `--part2`, `--recalage2`.

## Règle (écrite avant tout score de GB)

**Échec technique**, seul cas où S2 n'est pas installé :
- sortie non finie ou négative après bornage ;
- moyenne des RMSE temporelles de S2 > S0 + 0.0030 ;
- une coupure glissante > S0 + 0.0060 ;
- dérive manifeste : RMSE du dernier quart d'horizon > 1.5 × celle du premier quart, sur une validation où S0 ne le
  fait pas.

Dans ce cas, le livré reste S0 et ce journal dit quelle limite a cédé et sur quel circuit. **Sinon, S2 est installé**,
même s'il est moins bon que S0 en interne ; l'écart réel est écrit sans arrondi. Les scores ne servent pas à modifier
la liste, les options de GB, les poids ou les seuils ; un seuil jugé mal placé est signalé dans « Limites » sans être
changé.

Précision (choix d'exécution, écrit avant mesure) : S0 et S2 sont à 10 graines, donc un seul run par direction ; la
« moyenne des 4 RMSE temporelles » est ici la moyenne directe / inverse, comme pour T3_g10 (0.077525) dans
`docs/JOURNAL_STRUCTURE.md`. Les quarts d'horizon sont ceux de `outils/structure.py` (4 tranches égales des heures
prédites, RMSE sur les 33 colonnes), sur le modèle complet, pour les 6 validations.

**Bruit de référence** (repris de `docs/JOURNAL_STRUCTURE.md`, |g04 − g59| du livré complet, au moins l'effet
d'environnement du programme 4, ≈ 0.0001 sur la RMSE globale) : directe 0.0001, inverse 0.00013, c50 0.0001, c62
0.0001, c75 0.00014, c87 0.0004. Tout écart plus petit est « égal ».

Validations : directe, inverse, coupures glissantes c50, c62, c75, c87 (`validation_glissant/`, définition de
`docs/JOURNAL_STRUCTURE.md`) ; c50 réutilise les modèles de la directe. Vocabulaires et statistiques de normalisation
réappris dans chaque entraînement. Seuls les 7 jours réels initiaux de chaque test sont lus (réchauffement de GB,
recalage de G0 et du boosting comme dans T3).

## Ordre d'exécution

1. Contrôle de GB sur la coupure courte c62 (session K0) : sorties finies, pas de dérive, temps. Arrêt si GB diverge.
2. Directe et inverse (K1), c75 et c87 (K2), puis annee1 + annee2 (K3) ; GB et GB-mm, graines 0–4 et 5–9.
3. Assemblage S1, S2, S2 à 0.6 ; contrôles au bit près (part boosting, NPWD2532, spécialistes) contre T3.
4. Bilan, décision mécanique, installation, documents.

## Étape 1 — S0 reproduit (`outils/gru_boucle.py repro`)

`predict.py` → `evaluate.py` sur les six dossiers `runs/st_T3_<v>_g10`, avec le `dio/ensemble.py` modifié (clé
`recette2` absente) : directe 0.074604, inverse 0.080447, c50 0.064210, c62 0.063848, c75 0.092209, c87 0.076363,
soit exactement les scores de T3 de `docs/JOURNAL_STRUCTURE.md`. Les six `pred_repro.csv` et leurs parts (réseau,
boosting) sont identiques au bit près aux `pred.csv` d'origine : le code ajouté ne change rien sans `recette2`.

## Étape 2 — contrôle de GB sur c62 (session K0, Kaggle `zendarc147-1`, T4)

Quatre entraînements de 5 graines (GB g04 / g59, GB-mm g04 / g59) : 419, 417, 422 et 423 s. RMSE d'entraînement de la
dernière époque (p = 0, sans vérité réinjectée) : 0.056 à 0.058. GRU seuls sur c62 (10 graines, prédiction locale
8–9 s) :

| GRU | RMSE | Sorties finies | Quarts q1 / q2 / q3 / q4 | q4 / q1 |
|---|---:|---|---|---:|
| G0 (recalé) | 0.062202 | oui | 0.0608 / 0.0838 / 0.0464 / 0.0509 | 0.84 |
| GB | 0.061160 | oui | 0.0582 / 0.0806 / 0.0485 / 0.0522 | 0.90 |
| GB-mm | 0.061246 | oui | 0.0588 / 0.0807 / 0.0483 / 0.0519 | 0.88 |

GB ne diverge pas : les autres sessions sont lancées (K1 directe + inverse, fente 1 ; K2 c75 + c87, fente 2).

## Étape 3 — entraînements Kaggle (`zendarc147`, T4)

| Session | Fente | Contenu |
|---|---|---|
| K0 | 1 | c62 : GB g04 / g59, GB-mm g04 / g59 (contrôle) |
| K1 | 1 | directe (annee1) et inverse (annee2) : idem |
| K2 | 2 | c75 et c87 : idem |
| K3 | 1 | annee1 + annee2 : idem (poids finaux) |

Commandes : `uv run python outils/gru_boucle.py kaggle K0` (imprime les `train.py --modele gru_recette …`),
lancées par `runs/gb_lancer.sh K0 1 [--donnees]` (`outils/lancer_kaggle.py --fente …`). Le dataset des données a été
renvoyé une fois avant K0 (dernière version du 4 novembre, `data/` réorganisé le 7). Le paquet de code embarquait
`poids_repli/` (61 Mo, refus Kaggle assuré) : `poids_repli/` est désormais exclu du paquet (323 ko). Vocabulaires et
statistiques de normalisation sont réappris par chaque `train.py`. Paquets à 10 graines : `outils/gru_boucle.py
fusionner` (configuration, variables, familles, statistiques vérifiées identiques entre g04 et g59).

## Étape 4 — assemblage et contrôles

Base de chaque validation : le dossier T3 à 10 graines (`runs/st_T3_<v>_g10`). `outils/candidat.py --recette2
gb_<v>_GB_g10 --part2 1` (S1) ou `--part2 0.5` (S2) ; S2 à 0.6 = dossier de S2 prédit avec `{"poids_gbm": 0.6}`.
Contrôles (`outils/gru_boucle.py controles`), sur les 6 validations, pour S1 et S2 : part boosting (`gbm.csv` des
parts) identique au bit près à celle de T3 ; colonne NPWD2532 du modèle complet identique au bit près ; fichiers de
`specialiste/` et de `gbm/` identiques (sha256) ; `recette/` de S2 = G0 de T3. **Tous OK.**

## Résultats (`runs/gb_bilan/bilan.json`, `runs/gb_bilan/tableaux.md`)

Tous les scores sont de bout en bout (`predict.py` → `evaluate.py`) pour les structures ; GRU seuls = sortie de
`dio/recette.predire` (10 graines) notée sur les mêmes heures. G0sr = G0 sans recalage (lecture du diagnostic de
réinjection). Les parts des structures : part réseau (avant mélange), part boosting (après remplacement de NPWD2551 par
B), mélange avant les spécialistes 2531 / 2532, modèle complet.

#### RMSE globale, GRU seuls (part réseau seule)

| | directe | inverse | c50 | c62 | c75 | c87 | Moy. dir/inv | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.078496 | 0.085615 | 0.065348 | 0.062202 | 0.098213 | 0.094507 | 0.082056 | 0.080067 |
| G0sr | 0.078505 | 0.093401 | 0.065404 | 0.062837 | 0.098895 | 0.093758 | 0.085953 | 0.080224 |
| GB | 0.079570 | 0.093860 | 0.067564 | 0.061160 | 0.101570 | 0.089988 | 0.086715 | 0.080070 |
| GBr | 0.079612 | 0.085944 | 0.067738 | 0.060873 | 0.100312 | 0.100687 | 0.082778 | 0.082403 |
| GBmm | 0.079493 | 0.098685 | 0.067704 | 0.061246 | 0.101285 | 0.089346 | 0.089089 | 0.079895 |

#### RMSE globale, structures (modèle complet)

| | directe | inverse | c50 | c62 | c75 | c87 | Moy. dir/inv | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.074604 | 0.080447 | 0.064210 | 0.063848 | 0.092209 | 0.076363 | 0.077525 | 0.074158 |
| S1 | 0.074752 | 0.080938 | 0.065801 | 0.062705 | 0.093110 | 0.073420 | 0.077845 | 0.073759 |
| S2 | 0.074572 | 0.080497 | 0.064867 | 0.063204 | 0.092547 | 0.074218 | 0.077534 | 0.073709 |
| S2b06 | 0.074741 | 0.080391 | 0.065561 | 0.065096 | 0.091784 | 0.072417 | 0.077566 | 0.073714 |

#### Structures : part réseau

| | directe | inverse | c50 | c62 | c75 | c87 | Moy. dir/inv | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.078496 | 0.085615 | 0.065348 | 0.062202 | 0.098213 | 0.094507 | 0.082056 | 0.080067 |
| S1 | 0.079570 | 0.093860 | 0.067564 | 0.061160 | 0.101570 | 0.089988 | 0.086715 | 0.080070 |
| S2 | 0.078621 | 0.088144 | 0.065879 | 0.061352 | 0.099458 | 0.089982 | 0.083382 | 0.079168 |
| S2b06 | 0.078621 | 0.088144 | 0.065879 | 0.061352 | 0.099458 | 0.089982 | 0.083382 | 0.079168 |

#### Structures : part boosting

| | directe | inverse | c50 | c62 | c75 | c87 | Moy. dir/inv | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.078902 | 0.082585 | 0.072776 | 0.076242 | 0.089708 | 0.071504 | 0.080743 | 0.077558 |
| S1 | 0.078902 | 0.082585 | 0.072776 | 0.076242 | 0.089708 | 0.071504 | 0.080743 | 0.077558 |
| S2 | 0.078902 | 0.082585 | 0.072776 | 0.076242 | 0.089708 | 0.071504 | 0.080743 | 0.077558 |
| S2b06 | 0.078902 | 0.082585 | 0.072776 | 0.076242 | 0.089708 | 0.071504 | 0.080743 | 0.077558 |

#### Structures : mélange avant spécialistes

| | directe | inverse | c50 | c62 | c75 | c87 | Moy. dir/inv | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0.074902 | 0.081492 | 0.065031 | 0.064292 | 0.091137 | 0.076336 | 0.078197 | 0.074199 |
| S1 | 0.075079 | 0.084517 | 0.066680 | 0.063241 | 0.091916 | 0.073350 | 0.079798 | 0.073797 |
| S2 | 0.074882 | 0.082564 | 0.065713 | 0.063689 | 0.091405 | 0.074157 | 0.078723 | 0.073741 |
| S2b06 | 0.075070 | 0.082104 | 0.066587 | 0.065502 | 0.090526 | 0.072423 | 0.078587 | 0.073760 |

#### Corrélation des erreurs G0 / GB (GRU seuls)

| Validation | globale | NPWD2551 | NPWD2451 | NPWD2851 |
|---|---:|---:|---:|---:|
| directe | 0.979 | 0.976 | 0.980 | 0.989 |
| inverse | 0.930 | 0.944 | 0.971 | 0.952 |
| c50 | 0.966 | 0.941 | 0.984 | 0.987 |
| c62 | 0.979 | 0.991 | 0.982 | 0.992 |
| c75 | 0.983 | 0.986 | 0.987 | 0.993 |
| c87 | 0.902 | 0.872 | 0.962 | 0.954 |

#### Circuits suivis : RMSE / MAE / biais, moyenne des 6 validations (et c87)

| Objet | NPWD2551 | NPWD2451 | NPWD2851 | NPWD2532 | NPWD2531 |
|---|---|---|---|---|---|
| G0 (moy. 6) | 0.2766 / 0.1764 / +0.0447 | 0.2359 / 0.1860 / +0.0308 | 0.1204 / 0.0934 / +0.0049 | 0.1376 / 0.0906 / +0.0201 | 0.0727 / 0.0299 / -0.0137 |
| G0 (c87) | 0.3812 / 0.2982 / +0.2075 | 0.2833 / 0.2374 / +0.1655 | 0.1217 / 0.1015 / +0.0562 | 0.0689 / 0.0497 / -0.0026 | 0.0125 / 0.0051 / +0.0043 |
| G0sr (moy. 6) | 0.2768 / 0.1750 / +0.0449 | 0.2370 / 0.1865 / +0.0301 | 0.1244 / 0.0953 / +0.0092 | 0.1378 / 0.0908 / +0.0200 | 0.0869 / 0.0348 / -0.0246 |
| G0sr (c87) | 0.3805 / 0.2973 / +0.2062 | 0.2788 / 0.2332 / +0.1583 | 0.1192 / 0.0978 / +0.0454 | 0.0693 / 0.0497 / +0.0069 | 0.0147 / 0.0069 / +0.0063 |
| GB (moy. 6) | 0.2779 / 0.1717 / +0.0226 | 0.2345 / 0.1843 / +0.0320 | 0.1258 / 0.0968 / +0.0095 | 0.1403 / 0.0911 / +0.0268 | 0.0855 / 0.0338 / -0.0251 |
| GB (c87) | 0.3498 / 0.2647 / +0.0990 | 0.2527 / 0.2090 / +0.1149 | 0.1338 / 0.1072 / +0.0426 | 0.0746 / 0.0537 / +0.0190 | 0.0047 / 0.0025 / +0.0019 |
| GBr (moy. 6) | 0.2838 / 0.1805 / +0.0337 | 0.2399 / 0.1900 / +0.0401 | 0.1278 / 0.1005 / +0.0185 | 0.1397 / 0.0909 / +0.0236 | 0.0707 / 0.0282 / -0.0152 |
| GBr (c87) | 0.3888 / 0.3067 / +0.1920 | 0.2977 / 0.2518 / +0.1895 | 0.1677 / 0.1427 / +0.1229 | 0.0725 / 0.0524 / +0.0005 | 0.0012 / 0.0005 / -0.0005 |
| GBmm (moy. 6) | 0.2759 / 0.1723 / +0.0319 | 0.2335 / 0.1841 / +0.0391 | 0.1307 / 0.1009 / +0.0074 | 0.1388 / 0.0908 / +0.0250 | 0.0925 / 0.0455 / -0.0108 |
| GBmm (c87) | 0.3431 / 0.2601 / +0.0900 | 0.2526 / 0.2088 / +0.1170 | 0.1349 / 0.1080 / +0.0430 | 0.0751 / 0.0543 / +0.0209 | 0.0052 / 0.0030 / +0.0025 |
| S0 (moy. 6) | 0.2527 / 0.1792 / +0.0462 | 0.2242 / 0.1769 / +0.0086 | 0.1189 / 0.0910 / -0.0118 | 0.1260 / 0.0864 / +0.0043 | 0.0713 / 0.0281 / -0.0164 |
| S0 (c87) | 0.2851 / 0.2341 / +0.1439 | 0.2293 / 0.1904 / +0.0664 | 0.1090 / 0.0875 / +0.0346 | 0.0741 / 0.0589 / -0.0285 | 0.0000 / 0.0000 / -0.0000 |
| S1 (moy. 6) | 0.2502 / 0.1746 / +0.0352 | 0.2241 / 0.1768 / +0.0091 | 0.1207 / 0.0921 / -0.0095 | 0.1260 / 0.0864 / +0.0043 | 0.0712 / 0.0274 / -0.0172 |
| S1 (c87) | 0.2555 / 0.2066 / +0.0897 | 0.2224 / 0.1829 / +0.0411 | 0.1152 / 0.0909 / +0.0278 | 0.0741 / 0.0589 / -0.0285 | 0.0000 / 0.0000 / -0.0000 |
| S2 (moy. 6) | 0.2503 / 0.1762 / +0.0407 | 0.2237 / 0.1766 / +0.0089 | 0.1196 / 0.0914 / -0.0106 | 0.1260 / 0.0864 / +0.0043 | 0.0712 / 0.0278 / -0.0168 |
| S2 (c87) | 0.2661 / 0.2171 / +0.1168 | 0.2250 / 0.1861 / +0.0538 | 0.1116 / 0.0889 / +0.0312 | 0.0741 / 0.0589 / -0.0285 | 0.0000 / 0.0000 / -0.0000 |
| S2b06 (moy. 6) | 0.2511 / 0.1793 / +0.0421 | 0.2238 / 0.1765 / +0.0043 | 0.1200 / 0.0917 / -0.0142 | 0.1260 / 0.0864 / +0.0043 | 0.0713 / 0.0279 / -0.0166 |
| S2b06 (c87) | 0.2551 / 0.2089 / +0.1095 | 0.2221 / 0.1823 / +0.0365 | 0.1093 / 0.0864 / +0.0275 | 0.0741 / 0.0589 / -0.0285 | 0.0000 / 0.0000 / -0.0000 |

#### RMSE par quart d'horizon (q1 / q2 / q3 / q4)

| Objet | directe | inverse | c50 | c62 | c75 | c87 |
|---|---|---|---|---|---|---|
| G0 | 0.0654 / 0.0646 / 0.1015 / 0.0767 | 0.0897 / 0.0756 / 0.0933 / 0.0827 | 0.0433 / 0.0752 / 0.0635 / 0.0743 | 0.0608 / 0.0838 / 0.0464 / 0.0509 | 0.0762 / 0.0913 / 0.1217 / 0.0981 | 0.0887 / 0.0866 / 0.0953 / 0.1062 |
| G0sr | 0.0655 / 0.0646 / 0.1015 / 0.0767 | 0.1170 / 0.0752 / 0.0934 / 0.0827 | 0.0433 / 0.0749 / 0.0638 / 0.0746 | 0.0620 / 0.0851 / 0.0461 / 0.0508 | 0.0754 / 0.0924 / 0.1231 / 0.0987 | 0.0885 / 0.0854 / 0.0943 / 0.1055 |
| GB | 0.0676 / 0.0644 / 0.1034 / 0.0769 | 0.1179 / 0.0765 / 0.0926 / 0.0832 | 0.0428 / 0.0839 / 0.0630 / 0.0737 | 0.0582 / 0.0806 / 0.0485 / 0.0522 | 0.0754 / 0.0925 / 0.1285 / 0.1025 | 0.0749 / 0.0760 / 0.0898 / 0.1138 |
| GBr | 0.0678 / 0.0644 / 0.1034 / 0.0769 | 0.0906 / 0.0766 / 0.0926 / 0.0832 | 0.0430 / 0.0850 / 0.0625 / 0.0733 | 0.0577 / 0.0797 / 0.0490 / 0.0523 | 0.0784 / 0.0911 / 0.1250 / 0.1008 | 0.1031 / 0.0779 / 0.0959 / 0.1210 |
| GBmm | 0.0678 / 0.0641 / 0.1035 / 0.0765 | 0.1174 / 0.0759 / 0.1024 / 0.0945 | 0.0427 / 0.0834 / 0.0633 / 0.0745 | 0.0588 / 0.0807 / 0.0483 / 0.0519 | 0.0755 / 0.0924 / 0.1282 / 0.1019 | 0.0751 / 0.0766 / 0.0896 / 0.1113 |
| S0 | 0.0642 / 0.0628 / 0.0950 / 0.0720 | 0.0868 / 0.0743 / 0.0863 / 0.0734 | 0.0417 / 0.0785 / 0.0622 / 0.0687 | 0.0706 / 0.0783 / 0.0504 / 0.0516 | 0.0755 / 0.0904 / 0.1126 / 0.0864 | 0.0757 / 0.0738 / 0.0698 / 0.0853 |
| S1 | 0.0658 / 0.0619 / 0.0955 / 0.0712 | 0.0893 / 0.0739 / 0.0857 / 0.0736 | 0.0417 / 0.0840 / 0.0620 / 0.0684 | 0.0682 / 0.0761 / 0.0510 / 0.0518 | 0.0754 / 0.0910 / 0.1150 / 0.0866 | 0.0691 / 0.0667 / 0.0675 / 0.0882 |
| S2 | 0.0649 / 0.0623 / 0.0952 / 0.0715 | 0.0877 / 0.0740 / 0.0859 / 0.0733 | 0.0416 / 0.0810 / 0.0621 / 0.0685 | 0.0694 / 0.0771 / 0.0506 / 0.0516 | 0.0754 / 0.0906 / 0.1137 / 0.0863 | 0.0721 / 0.0688 / 0.0679 / 0.0865 |
| S2b06 | 0.0655 / 0.0628 / 0.0945 / 0.0720 | 0.0873 / 0.0751 / 0.0857 / 0.0724 | 0.0417 / 0.0820 / 0.0632 / 0.0688 | 0.0728 / 0.0773 / 0.0528 / 0.0536 | 0.0753 / 0.0907 / 0.1121 / 0.0850 | 0.0710 / 0.0686 / 0.0651 / 0.0837 |

#### Biais hors saison d'éclipses / régime haut hors saison (moyenne des 4 coupures ; directe ; inverse)

| Objet | 2551 hors saison | 2551 haut hors saison | 2451 hors saison | 2451 haut hors saison |
|---|---|---|---|---|
| G0 | +0.0162 ; -0.0461 ; -0.0105 | -0.1300 ; -0.1831 ; -0.3685 | +0.0279 ; +0.0489 ; -0.0415 | -0.0434 ; +0.0681 ; -0.0809 |
| G0sr | +0.0211 ; -0.0439 ; -0.0355 | -0.1257 ; -0.1821 ; -0.3691 | +0.0369 ; +0.0508 ; -0.0567 | -0.0556 ; +0.0682 ; -0.1077 |
| GB | -0.0125 ; -0.0619 ; -0.0296 | -0.1711 ; -0.2296 ; -0.2835 | +0.0370 ; +0.0559 ; -0.0763 | -0.0412 ; +0.0380 ; -0.1580 |
| GBr | +0.0041 ; -0.0655 ; -0.0021 | -0.1575 ; -0.2310 ; -0.2823 | +0.0336 ; +0.0524 ; -0.0535 | +0.0016 ; +0.0377 ; -0.1177 |
| GBmm | -0.0153 ; -0.0563 ; -0.0110 | -0.1743 ; -0.2201 ; -0.2810 | +0.0382 ; +0.0596 ; -0.0355 | -0.0389 ; +0.0454 ; -0.0875 |
| S0 | +0.0462 ; +0.0064 ; -0.0064 | -0.0888 ; -0.0887 ; -0.3081 | +0.0085 ; +0.0267 ; -0.0167 | -0.1060 ; +0.0081 ; -0.0852 |
| S1 | +0.0318 ; -0.0015 ; -0.0160 | -0.1093 ; -0.1120 ; -0.2656 | +0.0131 ; +0.0302 ; -0.0341 | -0.1049 ; -0.0070 ; -0.1237 |
| S2 | +0.0390 ; +0.0024 ; -0.0112 | -0.0990 ; -0.1003 ; -0.2869 | +0.0108 ; +0.0284 ; -0.0254 | -0.1055 ; +0.0006 ; -0.1045 |
| S2b06 | +0.0464 ; +0.0137 ; -0.0094 | -0.0887 ; -0.0791 ; -0.2790 | +0.0064 ; +0.0237 ; -0.0187 | -0.1181 ; -0.0099 ; -0.1015 |

#### Diagnostics de réglage (RMSE globale ; Δ négatif = le réglage aide)

| Validation | réinjection : GBmm − G0sr | normalisation : GB − GBmm | recalage : GB − GBr |
|---|---:|---:|---:|
| directe | +0.000988 | +0.000076 | -0.000042 |
| inverse | +0.005284 | -0.004825 | +0.007916 |
| c50 | +0.002300 | -0.000140 | -0.000174 |
| c62 | -0.001591 | -0.000086 | +0.000287 |
| c75 | +0.002390 | +0.000285 | +0.001258 |
| c87 | -0.004412 | +0.000641 | -0.010700 |

Notes de lecture des tableaux : « moyenne des 4 coupures » du régime haut hors saison de NPWD2451 = moyenne des seules
coupures qui en contiennent (c62 : 336 h, c87 : 2 146 h ; aucune heure sur c50 et c75). Pour NPWD2551 : c50 168 h,
c62 336 h, c75 82 h, c87 1 310 h. Corrélation des erreurs GB / GB-mm (même boucle, normalisation différente) : 0.998
à 0.999 partout sauf l'inverse (0.901). Corrélation des erreurs part réseau / part boosting : S0 0.69 à 0.88, S2 0.69
à 0.87 (presque inchangée).

#### Temps et entrées

- Entraînement sur T4 (5 graines par entraînement, deux par GRU à 10 graines) : GB 368–377 s (directe, inverse),
  417–419 s (c62), 523–531 s (c75), 573–579 s (c87) ; GB-mm 370–582 s, mêmes ordres. Pour mémoire, G0 (GRU S borné) :
  389–538 s sur les coupures (`docs/JOURNAL_STRUCTURE.md`). La boucle ne coûte presque rien à l'entraînement.
- Prédiction d'un GRU à 10 graines seul (local, 4 en parallèle) : 7 à 13 s sur une coupure, 39 à 50 s sur une année ;
  GB et G0 au même niveau. Prédiction de bout en bout (local, 3 en parallèle, temps très bruités par la concurrence) :
  S2 96 à 625 s, S0 32 à 740 s.
- Entrées réellement lues : G0 165 variables ; GB et GB-mm 165 variables + 33 consommations réinjectées (198).
  Seuls les 7 jours de power_init sont lus côté cibles (réchauffement de GB, recalage de G0 et du boosting).

## Décision mécanique (`outils/gru_boucle.py decider`, `runs/gb_bilan/decision.json`)

| | Directe | Inverse | Moy. temporelle | c50 | c62 | c75 | c87 | Moy. coupures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 (T3) | 0.074604 | 0.080447 | 0.077525 | 0.064210 | 0.063848 | 0.092209 | 0.076363 | 0.074158 |
| S2 | 0.074572 | 0.080497 | 0.077534 | 0.064867 | 0.063204 | 0.092547 | 0.074218 | 0.073709 |
| Écart S2 − S0 | −0.000031 | +0.000050 | +0.000009 | +0.000656 | −0.000644 | +0.000339 | −0.002145 | −0.000449 |
| Bruit | 0.0001 | 0.00013 | | 0.0001 | 0.0001 | 0.00014 | 0.0004 | |
| Lecture | égal | égal | égal | pire | meilleur | pire | meilleur | |

Limites de la règle : sorties non finies ou négatives : non ; moyenne temporelle S2 > S0 + 0.0030 : non (+0.000009) ;
une coupure > S0 + 0.0060 : non (pire coupure c50, +0.000656) ; dérive manifeste : non (q4 / q1 de S2 : directe 1.10,
inverse 0.84, c50 1.65 mais S0 1.65 aussi, c62 0.74, c75 1.14, c87 1.20 ; S0 1.13 sur c87). **Aucune limite ne cède :
S2 est installé.**

Écart réel : S2 est **égal** à S0 sur les deux RMSE temporelles (+0.000009 en moyenne, sous le bruit), **meilleur** en
moyenne des coupures (−0.000449), gagne nettement sur c87 (−0.002145) et c62 (−0.000644), perd sur c50 (+0.000656)
et c75 (+0.000339). Variante de rapport S2 à 0.6 (réseau 0.4, non installée) : moyenne temporelle 0.077566 (+0.000041),
moyenne des coupures 0.073714 (−0.000444) ; meilleure que S2 sur c75 et c87, pire sur c50 et c62. S1 (GB seul) :
moyenne temporelle 0.077845 (+0.000320), coupures 0.073759.

### Lecture (constats, pas décisions)

- **Diversité G0 / GB** : corrélation des erreurs 0.902 à 0.983 selon la validation (NPWD2551 : 0.872 à 0.991 ;
  NPWD2451 : 0.962 à 0.987 ; NPWD2851 : 0.952 à 0.993). C'est moins que deux GRU de même recette (0.96–0.99) surtout
  sur c87 (0.902 ; 0.872 pour NPWD2551) et en inverse (0.930), mais à peine moins ailleurs (0.97–0.98). La moyenne des
  deux GRU fait mieux que G0 seul sur les coupures (part réseau 0.079168 contre 0.080067) et moins bien sur directe /
  inverse (0.083382 contre 0.082056, surtout l'inverse : 0.088144 contre 0.085615).
- **c87, la coupure qui motivait la mission** : part réseau 0.094507 (G0) → 0.089982 (S2) ; NPWD2551 du modèle
  complet 0.2851 → 0.2661, biais +0.1439 → +0.1168 ; NPWD2451 0.2293 → 0.2250. GB seul surestime moins ces chauffages
  hors saison (biais NPWD2551 c87 +0.099 contre +0.208 pour G0).
- **Revers du même effet** : GB descend plus bas que G0 en régime haut hors saison. Biais de NPWD2551 en régime haut
  hors saison, moyenne des coupures : S0 −0.0888, S2 −0.0990, S1 −0.1093 (directe : −0.0887, −0.1003, −0.1120). La
  sous-estimation redoutée par `docs/JOURNAL_STRUCTURE.md` augmente donc d'environ 0.01 avec S2. En inverse, S2 la
  réduit (−0.3081 → −0.2869).
- **Dérive** : GB seul dépasse 1.5 × q1 au dernier quart de c87 (0.1138 / 0.0749 = 1.52) alors que G0 reste à 1.20 ;
  dans le modèle complet, S2 reste à 1.20 (S0 1.13). La règle porte sur le modèle complet ; la boucle seule dérive
  légèrement en fin de c87, elle est amortie par le mélange.

### Diagnostics de réglage (lecture seule, ne décident rien)

- **Réinjection** (GB-mm contre G0 sans recalage, même normalisation minmax, seul change la boucle) : pire sur directe
  (+0.0010), inverse (+0.0053), c50 (+0.0023), c75 (+0.0024) ; meilleure sur c62 (−0.0016) et c87 (−0.0044).
  Conforme au jugement de `docs/JOURNAL_RECETTE_COMPLETE.md` sur directe / inverse ; faux sur les deux coupures où le
  régime change (c62, c87).
- **Normalisation** (GB standard contre GB-mm) : égale sur directe, c50, c62 (|Δ| ≤ 0.00014), meilleure en inverse
  (−0.0048), pire sur c75 (+0.0003) et c87 (+0.0006). Contrairement à G0, le bornage min/max n'aide pas la boucle en
  inverse : la consommation réinjectée suffit à ramener l'état quand la période orbitale sort du domaine appris.
- **Recalage** (GB contre GB-r) : égal en directe et c50, sans recalage pire en inverse (+0.0079), c75 (+0.0013) et
  c62 (+0.0003), nettement meilleur sur c87 (−0.0107). Le recalage sur power_init aide quand le niveau des 7 premiers
  jours prolonge l'année ; il nuit quand le régime bascule juste après (c87).

## Installation (S2)

- K3 (Kaggle `zendarc147`, T4) : GB final 680 / 676 s, GB-mm final 689 / 683 s (annee1 + annee2, 5 graines chacun),
  fusionnés en 10 graines.
- `outils/gru_boucle.py finaux` : `poids_candidats/gru_boucle/{S1, S2, S2_b06, GBmm_final}` (copies indépendantes,
  aucun fichier à liens multiples ; `INDEX.md`). `poids_candidats/structure/T3_g10/` vérifié inchangé (sha256 des 11
  fichiers de calcul identiques au gel) avant l'installation.
- `poids/` = copie indépendante de `poids_candidats/gru_boucle/S2/` (13 fichiers, tous à un seul lien ; seule
  `modele.json` diffère, réécrite après `unlink` avec la note d'installation). Nouveaux fichiers : `recette2/recette.pt`
  (sha256 `c379acfaacdf5340…`), `ensemble.json` (`{"poids_gbm": 0.5, "part_recette": 1.0, "recette2": 0.5,
  "recalage_recette2": 0.0}`). `modele.json` : note, `options_recette` (G0), `options_recette2` (GB, graines 0–9),
  `recalage_recette2` 0.0. Les autres fichiers sont ceux de T3.
- `dio/ensemble.py` : aucun défaut modifié. Comme pour T3, `train.py --modele gru_gbm` ne reproduit pas la structure
  installée (il reproduit C1) : les GRU à 10 graines sont des fusions de deux entraînements, et S2 s'assemble avec
  `outils/candidat.py` / `outils/gru_boucle.py finaux` à partir des commandes de reproduction ci-dessous.
- `predict.py` avec le nouveau `poids/` sur `validation_glissant/c87/test` : 3953 × 33 valeurs finies et ≥ 0, heures
  conformes, 64 s. Score sans valeur sur cette période (annee2 est dans l'entraînement des poids finaux).
- Test ajouté (`tests/test_ensemble.py`, `test_part_gru_deux_gru_recette`) : part réseau = 0.5 × G0 + 0.5 × GB.
- Suite complète sur le `poids/` installé : `uv run python -m unittest discover tests` : 113 tests OK, 4 sautés
  (les mêmes qu'avant ; le nouveau test passe), 3 969 s (en parallèle avec la preuve de reproduction).

Part du réseau : 0.5 du mélange, partagée par deux GRU (NPWD2532 : 0.4). **Second GRU à boucle ajouté sur décision de
l'utilisateur ; ce choix ne maximise pas la validation interne** : S2 est égal à T3 sur directe / inverse (+0.000009)
et reste derrière le livré L (0.076735) et C1 (0.076288).

## Reproduction

```bash
python outils/faire_chrono.py --glissant
runs/gb_lancer.sh K0 1 --donnees      # puis K1 1, K2 2, K3 1 (uv run python outils/gru_boucle.py kaggle K<n> : commandes)
uv run python outils/gru_boucle.py fusionner
uv run python outils/gru_boucle.py repro
uv run python outils/gru_boucle.py assembler
uv run python outils/gru_boucle.py gru
uv run python outils/gru_boucle.py predire
uv run python outils/gru_boucle.py controles
uv run python outils/gru_boucle.py bilan
uv run python outils/gru_boucle.py decider
uv run python outils/gru_boucle.py finaux
```

`runs/gb_lancer.sh` (non suivi, comme `runs/`) : `mapfile -d '' ARGS < <(uv run python outils/gru_boucle.py kaggle
"$1"); exec uv run python outils/lancer_kaggle.py --fente "$2" "${@:3}" -- "${ARGS[@]}"`. Les dossiers T3 à 10 graines
de chaque validation (`runs/st_T3_<v>_g10`) viennent de `outils/structure.py assembler` (`docs/JOURNAL_STRUCTURE.md`).

## Limites

- **Seuils jugés mal placés (signalés, non modifiés)** : « moyenne temporelle ≤ S0 + 0.0030 » et « chaque coupure ≤
  S0 + 0.0060 » valent 20 à 60 fois le bruit mesuré (0.0001 à 0.0004). Ils ne pouvaient refuser qu'un GB cassé, ce qui
  était leur rôle (échec technique), mais ils n'auraient pas vu une perte réelle de quelques millièmes. La perte de S2
  sur c50 (+0.00066) et c75 (+0.00034) dépasse le bruit.
- Le critère de dérive porte sur le modèle complet : GB seul dépasse 1.5 × q1 au dernier quart de c87 (1.52, G0 1.20).
- La diversité apportée est modeste : erreurs G0 / GB corrélées à 0.90–0.98 ; seules c87 et l'inverse descendent
  sous 0.95.
- S2 augmente un peu la sous-estimation de NPWD2551 en régime haut hors saison sur les coupures (−0.089 → −0.099), le
  risque que `docs/JOURNAL_STRUCTURE.md` voulait réduire ; il la réduit en inverse.
- c50 réutilise les modèles de la directe (non indépendantes) ; quatre coupures sur deux années ; c87 contient le seul
  régime haut hors saison de février–mars 2012. Les poids finaux n'ont pas de score propre.
- Une seule série d'entraînements par GRU (10 graines, deux groupes fusionnés) : le bruit entre deux séries de GB n'est
  pas mesuré ; celui de G0 l'est (|g04 − g59|).
- Temps de prédiction locaux mesurés avec plusieurs prédictions en parallèle : bruités.
- Aucun score n'est promis sur une année future.

## Preuve de reproduction

`predict.py` → `evaluate.py` relancés sur les six dossiers S2 (`runs/gb_S2_<v>/<v>/pred_verif.csv`) : directe
0.074572, inverse 0.080497, c50 0.064867, c62 0.063204, c75 0.092547, c87 0.074218, identiques aux scores annoncés ;
les six fichiers de prédiction sont identiques au bit près aux `pred.csv` de la mesure.
