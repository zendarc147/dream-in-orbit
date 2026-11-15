# Recette B : boosting horaire indépendant pour NPWD2551

## Objectif et statut

Implémenter une recette figée, puis mesurer son transfert sur les validations internes. Utiliser les données années 1 et 2 disponibles dans le projet. Cette recette décrit un HistGradientBoostingRegressor, pas XGBoost. Elle ne reproduit pas un GRU ni un ensemble GRU/boosting complet. Aucun gain n'est garanti.

Première étape : reproduire exactement les entrées et l'entraînement ci-dessous. Les changements jugés souhaitables viennent ensuite, comme ablations distinctes. Ne pas réutiliser une variable existante sous prétexte qu'elle porte un nom similaire : comparer sa formule et ses valeurs.

Périmètre : un spécialiste pour NPWD2551. Le modèle livré et les 32 autres sorties restent inchangés pendant l'expérience. Lire les 33 cibles d'entraînement pour construire le masque collectif, mais n'ajuster que NPWD2551.

## 1. Grille, fichiers et précision

- H = 3 600 000 millisecondes. Index `ut_ms`, UTC, début de l'heure.
- Grille complète : `arange(floor(start/H)*H, floor(end/H)*H + H, H)`, int64.
- En entraînement, start/end = première/dernière heure de la moyenne horaire de `power.csv`.
- Cibles : arrondir chaque timestamp vers le bas à l'heure, moyenne arithmétique par heure et circuit, réindexer sur la grille complète, convertir en float32. Une cible absente reste NaN.
- Au test : grille de la première heure de `power_init` jusqu'à la dernière heure demandée, incluses. Calculer les entrées sur toute cette grille ; sélectionner les heures demandées seulement à la sortie.
- Contexte brut : `saaf.csv`, `ltdata.csv`, `evtf.csv`, `ftl.csv`, `dmop.csv`.
- Ne pas inclure `ut_ms`, les cibles ou les prédictions du livré comme variables.
- Construire d'abord chaque période séparément, puis concaténer les périodes d'entraînement dans l'ordre chronologique. Les historiques ne traversent pas la frontière des dossiers.
- Pas de normalisation des entrées, pas de standardisation de la cible, pas de bornage des entrées aux extrema d'entraînement.
- À la fin de l'assemblage ET des variables dérivées : remplacer ±inf par NaN, forward-fill, backward-fill, puis remplir les NaN restants à zéro, colonne par colonne ; convertir X en float32. Ce remplissage concerne uniquement X.

### Interpolation commune

Pour SAAF et LTDATA : union triée des timestamps de la table et de la grille ; interpolation linéaire selon la valeur numérique de l'index (`method='index', limit_direction='both'`), puis sélection de la grille et ffill/bfill. Les interpolations du contexte futur sont autorisées puisque ce contexte est fourni au test. Elles ne concernent jamais les puissances futures.

## 2. SAAF : 9 colonnes

Pour chaque mesure brute, calculer `cos(deg2rad(angle))` pour sa, sx, sy, sz AVANT l'agrégation. Grouper les mesures par heure :

- `saaf_sa`, `saaf_sx`, `saaf_sy`, `saaf_sz` : moyenne arithmétique des mesures de l'heure.
- `saaf_cos_sa`, `saaf_cos_sx`, `saaf_cos_sy`, `saaf_cos_sz` : moyenne arithmétique des cosinus des mesures.
- `saaf_sa_std` : écart-type pandas de sa, ddof=1 ; NaN de cet écart-type remplacé par 0 avant interpolation.

Interpoler ensuite selon la convention commune. Aucune pondération par durée, aucun maintien de 60 secondes, aucune couverture supplémentaire.

## 3. LTDATA

Lire les colonnes dans leur ordre CSV, indexer par ut_ms trié et préfixer tous les noms par `lt_`. Interpoler aux débuts d'heure, sans décalage de 30 minutes. Deux colonnes seront exclues à la sélection finale : celles dont le nom contient `lt_earthmars` ou `lt_sunmarsearth`.

## 4. Intervalles : convention exacte

Une fraction horaire = durée de l'union des intervalles [s,e) intersectant [h,h+H), divisée par H. Fusionner les intervalles chevauchants ou contigus avant le calcul. Une fraction par série est comprise entre 0 et 1.

Appariement START/END : prendre uniquement les événements correspondants, les trier par ut_ms ; un START ouvre ou remplace l'ouverture courante ; le END suivant ferme cette ouverture ; ignorer END sans START et START sans END. Ne pas ajouter de durée maximale ni corriger silencieusement cette convention. Certains appariements sont volontairement communs à plusieurs libellés : une correction physique constitue une ablation distincte.

## 5. EVTF : 15 colonnes, dans cet ordre

Utiliser `description.astype(str)` et les sélections suivantes :

1. `ev_umbra` : MAR_UMBRA_START / MAR_UMBRA_END, égalité exacte.
2. `ev_penumbra` : MAR_PENUMBRA_START / MAR_PENUMBRA_END.
3. `ev_moon_shadow` : START regex `^(PHO|DEI)_(UMBRA|PENUMBRA)_START$`, END équivalent ; appariement commun à tous ces événements.
4. `ev_occ_mars` : regex début `^OCC_MARS_START` / `^OCC_MARS_END`.
5. `ev_occ_mars_km` : `^OCC_MARS_\d+KM_START` / END, toutes altitudes réunies.
6. `ev_mrb_range` : `^MRB_/_RANGE_\d+KM_START` / END, toutes distances réunies.
7. `ev_gs_any`, puis 8. `ev_gs_sum` : détecter stations par `^([A-Z0-9]{3})_(AOS|LOS)_`. Pour chaque station triée, apparier tous ses AOS/LOS, sans distinguer les suffixes. `any` = union des intervalles de toutes les stations ; `sum` = somme des fractions par station, pouvant dépasser 1.
9–13. `ev_since_peri_h`, `ev_to_peri_h`, `ev_orbit_phase_cos`, `ev_orbit_phase_sin`, `ev_orbit_period_h` : voir ci-dessous.
14. `ev_log_alt` : voir ci-dessous.
15. `ev_n_events` : nombre de toutes les lignes EVTF dont le timestamp tombe dans l'heure, 0 sinon.

### Phase orbitale et bords

P = timestamps triés des descriptions commençant par PERICENTRE_PASSAGE ; A = APOCENTRE_PASSAGE. Au milieu d'heure m=h+H/2, si au moins deux péricentres :

```
i = searchsorted(P, m, side='right') - 1
prev = P[clip(i, 0, len(P)-1)]
next = P[clip(i+1, 0, len(P)-1)]
period = next-prev si next>prev, sinon median(diff(P))
since = m-prev si i>=0, sinon 0
phase = clip(since/period, 0, 1)
```

Les cinq sorties sont since/H, max(next-m,0)/H, cos(2πphase), sin(2πphase), period/H. Avec moins de deux péricentres, les cinq sorties valent 0. Après le dernier péricentre, conserver exactement ces formules : ne pas prolonger cycliquement la phase.

### Altitude logarithmique

Points : événements regex `^(\d+)_KM_(ASCEND|DESCEND)$` avec altitude numérique en km ; ajouter péricentres à 330 km et apocentres à 10 100 km. Concaténation dans cet ordre : points KM dans l'ordre CSV, péricentres triés, apocentres triés. Trier par timestamp avec `np.argsort` par défaut ; ne pas dédupliquer. Prendre le logarithme naturel des altitudes. Interpoler avec `np.interp` (maintien des valeurs aux bords), aux quatre instants h+H/8, h+3H/8, h+5H/8, h+7H/8. Moyenne des quatre LOGARITHMES. Avec moins de deux points, log(10100). Ce n'est ni la moyenne des altitudes, ni le logarithme de cette moyenne.

## 6. Vocabulaire : entraînement seulement

DMOP : `cmd = subsystem.astype(str).str.strip('"')`, préfixe = quatre premiers caractères ; h = floor(ut_ms/H)*H.

Transformation `split_stem(cmd)` : chercher la dernière lettre majuscule A–Z dans `cmd[4:]`, remplacer cette lettre par `_`, garder tout le reste (notamment les chiffres après cette lettre), et retourner aussi cette lettre comme variante. Si aucune lettre n'existe après le quatrième caractère, stem=commande entière, variante vide.

Exemples : AAAAF20C1 → (AAAAF20_1,C), AOOOF01D0 → (AOOOF01_0,D), AOOOF100C → (AOOOF100_,C), ATTT305C → (ATTT305_,C).

Apprendre les vocabulaires séparément dans CHAQUE période d'entraînement, puis faire leur union triée :

- Préfixes : tous les préfixes rencontrés, y compris commandes contenant un point.
- Pour stems et variantes, exclure les commandes contenant un point.
- Stems : fréquence >=50 dans au moins une période, toutes variantes confondues.
- Switches : dans une période, garder couples (stem,variante) de fréquence >=50 ; un stem est switch si au moins deux variantes passent ce seuil dans cette période. Union des couples entre périodes, puis tri lexicographique. Ne pas recalculer ces seuils sur la concaténation.
- FTL : union triée des types (`astype(str).str.strip('"')`) présents dans l'entraînement.
- En validation chronologique, limiter l'apprentissage du vocabulaire aux commandes dont h est strictement inférieur à la coupure, et aux FTL dont utb_ms est strictement inférieur à cette coupure.
- Les vocabulaires sont appris une fois sur l'entraînement externe, avant les cinq sacs. Ils ne sont pas réappris dans chaque sac.
- Les commandes exactes fréquentes serviraient à des colonnes supprimées dans cette recette ; il est inutile de les construire. Ne pas supprimer pour autant les préfixes/stems/changements qui sont conservés.

## 7. FTL

Dans l'ordre des types triés du vocabulaire : `ftl_TYPE` = fraction de l'union des intervalles utb_ms/ute_ms de ce type. Ajouter ensuite :

- `ftl_other = clip(1 - somme_des_fractions_des_types_connus, 0, 1)`.
- `ftl_comms` : fraction de l'union des intervalles dont `flagcomms.astype(str).str.upper()` contient TRUE.
- `ftl_n_switch` : nombre de débuts utb_ms dans chaque heure (pas un compte des seuls changements de type).

Ne pas exprimer ces fractions en minutes.

## 8. DMOP conservé : ordre et calcul

1. `dm_p_PREFIX` pour préfixes triés : nombre d'événements dans chaque heure ; puis `dm_total` : nombre de toutes les commandes de l'heure. Zéro quand aucune.
2. `dm_act_PREFIX_START` dans l'ordre : PENS/PENE, MOCS/MOCE, PPNS/PPNE, PDNS/PDNE, UPBS/UPBE, UDBS/UDBE. Extraire l'identifiant après le dernier point ; jointure interne sur cet identifiant entre START et END (tous les couples si doublons), ne garder que e>s ; fraction de leur union. Zéro si un côté manque.
3. `dm_pr_PREFIX` pour préfixes triés : log1p des heures depuis la dernière occurrence, plafond 720 h.
4. `dm_st_STEM` pour stems triés : même récence, toutes variantes réunies.
5. `dm_sc_STEM` : comptes par stem dans l'heure courante.
6. Tous les `dm_sn1_STEM`, puis tous les `dm_sn6_STEM` : compte à t+1 ; somme des comptes à t+1..t+6. Zéro hors grille. Les cumuls de cette étape sont en float32.
7. Pour chaque stem de switch trié : `dm_sw_STEM_chg`, puis `dm_sw_STEM_next`, log1p des heures depuis/jusqu'au changement de variante, plafond 720.

### Récence : lire AVANT de mettre à jour

Compter d'abord les événements sur la grille horaire. À l'heure d'indice i, récence=min(i-last,720), avec last=-inf au départ ; ENSUITE last=i si au moins une occurrence dans cette heure. Ainsi une commande à 12:20 n'a pas une récence nulle à 12:00 ; à 13:00 la récence vaut 1. Utiliser float32 pour ces matrices de récence et log1p.

### Changements : horaire et lettre inconnue

Pour chaque stem de switch appris, récupérer TOUTES ses variantes au test, y compris nouvelles. Trier les commandes par timestamp exact ; un changement existe quand la lettre diffère de celle de la ligne précédente. La première émission n'est pas un changement. Quantifier chaque changement à son heure, dédupliquer ces heures. Depuis et jusqu'au changement valent 0 pendant son heure, puis distances entières en heures, plafond 720. Sans changement connu, 720. Ne pas exposer l'identité de la variante au boosting.

Les comptes et récences de stems utilisent toutes les commandes brutes correspondant au stem appris. Le filtrage des commandes à point concerne l'apprentissage du vocabulaire, pas une suppression globale des événements à la prédiction.

## 9. Variables dérivées conservées

Construire les dérivées AVANT le remplissage global des entrées manquantes.

```
sol = lt_solarconstantmars / 600
shade = clip(1 - ev_umbra - 0.5*ev_penumbra, 0, 1)
```

Dans l'ordre sa,sx,sy,sz, ajouter `fx_AX_plus = sol*max(cos_AX,0)*shade`, puis `fx_AX_minus = sol*max(-cos_AX,0)*shade`. Ajouter `fx_total=sol*shade`. Attention : fx_total n'est PAS la somme des flux des faces.

Puis, dans cet ordre :

- `hist_h_since_umbra` : dernier indice i avec ev_umbra[i]>0, mise à jour AVANT calcul ; initialisation last=-720 ; log1p(min(i-last,720)). La pénombre seule ne remet pas ce compteur à zéro.
- `hist_umbra_7h`, `hist_umbra_24h` : sommes glissantes passées d'umbra sur 7 et 24 lignes.
- `hist_umbra_7d` : moyenne glissante passée sur 168 lignes.
- `hist_fx_6h`, `hist_fx_24h` : moyennes glissantes de fx_total sur 6 et 24 lignes.
- `hist_comms_24h` : moyenne glissante passée de ftl_comms sur 24 lignes.

Toutes les fenêtres incluent l'heure courante, `min_periods=1`. Ne pas inventer une variable « durée de dernière ombre » : elle n'est pas dans cette liste.

Enfin, décalages pour ces 10 séries, dans cet ordre : ev_umbra, ev_penumbra, saaf_cos_sa, saaf_cos_sx, saaf_cos_sy, saaf_cos_sz, ftl_comms, ftl_EARTH, ftl_SLEW, ev_log_alt. Pour chaque série présente, ajouter k=-3,-2,-1,+1,+2,+3, colonne `ll_SERIE_±k`, calcul pandas `serie.shift(-k)` : k positif = futur. Exemple : `ll_ev_umbra_+1`. Si une série n'existe pas (type FTL absent du vocabulaire), ne pas créer ses décalages. Les NaN des bords sont traités seulement par le remplissage global.

## 10. Sélection finale

Concaténation : SAAF, LTDATA, EVTF, FTL, DMOP dans les ordres ci-dessus, puis flux, historiques, décalages. Conserver l'ordre d'apparition et exclure par `re.search` :

```
lt_earthmars|lt_sunmarsearth|^roll_|^ewm_|^dm_s_|^dm_c_|^dm_r_
```

Ne pas exclure `dm_sc_`, `dm_st_`, `dm_sw_`, `hist_`, `ll_` ou `fx_`. Pas de filtre supplémentaire des colonnes constantes. Les colonnes roll/ewm et commandes exactes exclues peuvent être omises dès la construction puisqu'elles n'alimentent aucune variable conservée.

Les listes varient avec les données d'entraînement. 846 colonnes est un contrôle pour le vocabulaire années 1+2 joint, pas un nombre à imposer aux validations.

## 11. Entraînement exact

Environnement numérique de référence : scikit-learn 1.6.1, NumPy 1.26.4, pandas 2.3.3. Ne pas modifier l'environnement existant : créer un environnement séparé si une reproduction numérique exige ces versions (choisir un Python compatible, par exemple 3.11). Une autre version doit être consignée ; aucune identité des poids n'est alors garantie.

Cinq HistGradientBoostingRegressor indépendants pour NPWD2551 :

```
loss='squared_error'
learning_rate=0.02
max_iter=600
max_leaf_nodes=15
max_depth=None
min_samples_leaf=100
l2_regularization=3.0
max_bins=255
categorical_features='from_dtype'
max_features=1.0
quantile=None
verbose=0
monotonic_cst=None
interaction_cst=None
warm_start=False
early_stopping='auto'
scoring='loss'
validation_fraction=0.1
n_iter_no_change=10
tol=1e-7
random_state=0
```

Les matrices sont des tableaux NumPy float32 : `from_dtype` n'identifie donc aucune variable catégorielle. `parametres_reference.json` précise aussi les paramètres de la configuration de référence. La limite de 600 ne signifie pas 600 arbres effectivement appris. L'arrêt anticipé utilise la validation interne sklearn ; ne pas le remplacer silencieusement par un arrêt sur la validation externe.

Pour une matrice X de n lignes horaires AVANT suppression des cibles manquantes : bornes=`np.linspace(0,n,6).astype(int)`. Le sac i apprend sur toutes les lignes hors [borne_i,borne_i+1), sans embargo. Ensuite seulement, exclure du sac chaque ligne où AU MOINS UNE des 33 cibles est NaN. Ajuster la cible brute NPWD2551 sur les lignes restantes. Ce masque collectif est indispensable à la reproduction, même si une seule sortie est entraînée. Les entrées sont float32. Toutes les graines des cinq HGB valent 0.

Prédiction finale = `clip(mean(predictions_des_5_sacs),0,+inf)`. Faire la moyenne avant le bornage. Aucun recalage additif, multiplicatif, correction de biais, demi-vie ou adaptation via power_init. Ce dernier fixe seulement le début de grille pour les entrées.

Prédictions hors bloc : pour les lignes du bloc i, utiliser uniquement le modèle i, puis clip à zéro. Ne pas noter la moyenne des cinq modèles comme si elle était hors entraînement.

## 12. Protocole et comparaison contrôlée

- Directe : apprendre vocabulaire et poids sur année 1, prédire année 2.
- Inverse : apprendre sur année 2, prédire année 1.
- Chronologique : année 1 + première moitié d'année 2 pour l'entraînement ; seconde moitié comme validation. Fixer la coupure et les sept jours initiaux avant mesure, vérifier les heures exactes du protocole local.
- Pour la construction des entrées chronologiques, les fenêtres peuvent lire le contexte futur connu, jamais les cibles futures. La recette de référence calcule les entrées de l'année 2 complète puis tranche ; si le projet coupe les fichiers contexte en dossiers, documenter que les valeurs de bord ne sont plus identiques. Ne pas confondre ce changement avec un effet du modèle.
- Directe et chrono partagent une période de test ; les répétitions de graines ne sont pas de nouvelles périodes indépendantes.
- Limiter les expériences et leur sélection aux validations définies sur les années 1 et 2.

Comparer, pour chaque protocole, la matrice d'entrées actuelle et la recette B avec les mêmes hyperparamètres HGB et le même masque de lignes ; puis, à entrées fixes, comparer les réglages d'entraînement. Ainsi isoler entrées / réglages / masque. Garder séparément la reproduction stricte (masque collectif) et l'ablation masque propre à NPWD2551.

Rapporter : spécialiste livré seul, nouveau spécialiste seul, modèle livré complet, remplacement de sa seule part boosting pour NPWD2551 (en conservant sa part GRU), et éventuellement mélange dont le coefficient est choisi hors période notée. Ne pas confondre le score du boosting seul avec celui de l'ensemble complet.

Évaluer sur les heures attendues, uniquement les vérités finies. Vérifier unicité, ordre, couverture et prédictions finies. RMSE, MAE, biais, résultats mensuels et par régime. Ne pas installer automatiquement. Fixer avant mesure les critères de gain agrégé et de régression tolérée ; autoriser une égalité sur une période où une correction ne s'active pas.

## 13. Contrôles fournis et ordre de travail

Fichiers du dossier :

- `colonnes_reference.json` : noms ET ordre exacts des 846 entrées pour années 1+2.
- `vocabulaire_reference.json` : vocabulaire années 1+2, uniquement pour contrôler la préparation finale. INTERDIT de l'utiliser pour valider un modèle appris sur une seule année.
- `controle_entrees_annee1.csv` : lignes échantillons, toutes les entrées conservées ; pas de cibles.
- `controle_reference.json` : taille, bornes de grille, empreintes et tolérances.
- `parametres_reference.json` : paramètres d'un HGB observé, à confronter à la version sklearn.

1. Reproduire X année 1 avec le vocabulaire de contrôle années 1+2, sans entraînement ni score. Vérifier noms, ordre, float32 et valeurs aux timestamps fournis (atol=rtol=1e-6). L'empreinte binaire est un contrôle plus strict, dépendant de l'environnement ; en cas de divergence, produire les premières colonnes/timestamps divergents et écarts max, pas seulement « hash différent ».
2. Réapprendre les vocabulaires propres à chaque validation ; comparer les matrices avant les entraînements coûteux.
3. Exécuter les entraînements et comparaisons contrôlées.
4. Journaliser hypothèses, environnements, sources d'entraînement, hashes des entrées et décisions.
5. Ne toucher aux poids livrés qu'après résultat validé, puis prouver que les 32 autres sorties sont inchangées.

Les contrôles de préparation ne prouvent pas une amélioration prédictive. Ce dossier ne contient aucun poids et ne promet aucun score. Le résultat recherché est d'abord une reproduction fidèle permettant d'isoler l'origine d'un écart.
