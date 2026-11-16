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
