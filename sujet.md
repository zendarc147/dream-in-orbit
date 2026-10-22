Un satellite doit se chauffer pour que ses équipements ne gèlent pas, et ce chauffage consomme une grande partie de son électricité. Pour planifier ses opérations (pointer un instrument, communiquer avec la Terre, traverser une éclipse), les opérateurs doivent savoir à l'avance combien il va consommer. Les modèles physiques écrits à la main ne sont pas assez précis : c'est là que l'IA entre en jeu.
Les données de cette piscine sont de vraies données de télémétrie d'un satellite en opération, rendues publiques par une agence spatiale.
La solution s'appelle un world model : un simulateur appris à partir de l'historique, au lieu d'être écrit à la main avec des équations. C'est l'un des sujets les plus actifs de l'IA aujourd'hui (Genie chez Google DeepMind, AMI Labs de Yann LeCun, robotique, conduite autonome).
Comment marche un world model ? Il répond toujours à la même question :
état actuel + actions + contexte → état suivant
Pour simuler une longue période, il prédit l'heure suivante, puis réutilise sa propre prédiction pour prédire l'heure d'après, et ainsi de suite. C'est le « rêve ». Le problème : chaque petite erreur s'ajoute à la suivante et le rêve dérive. Tout l'enjeu est de construire un modèle qui reste fidèle à la réalité sur le long terme.
Construire une IA qui, à partir du seul planning du satellite (ce qu'il fait, où est le Soleil, quand arrivent les éclipses) et de ses 7 premiers jours réels, simule heure par heure la consommation de ses 33 circuits électriques sur une année entière.
Chaque dossier contient six fichiers. Les temps sont en millisecondes UTC (colonne ut_ms).
- Semaine 1 : comprendre les données. Explorer les fichiers, comprendre le lien entre éclipses, Soleil et chauffage. Tout aligner sur une grille horaire. Entraînement sur annee1, validation sur annee2 (jamais de découpage aléatoire).
- Semaine 2 : les baselines. Une prédiction constante, puis un modèle classique (régression ou gradient boosting) heure par heure.
- Semaine 3 : le world model. Un réseau de neurones séquentiel (GRU, LSTM ou transformer) qui garde la mémoire du passé. Simulation sur toute la période de validation à partir de 7 jours réels. Mesure de la dérive. Estimation de l'incertitude (plusieurs modèles qui votent).
- Semaine 4 : prouver et présenter. Ablations (qu'apportent les éclipses, le Soleil, les commandes, la mémoire ?), analyse des erreurs, démo, rapport.
À la fin de la piscine, l'orga lance ton modèle sur un jeu de test que tu n'as pas, au même format que le kit et identique pour tous. Ton modèle reçoit le planning et le contexte de la période de test, ses 7 premiers jours réels de consommation, et doit simuler tout le reste.
Ton dépôt doit contenir cette commande :
python predict.py --test_dir DOSSIER --out predictions.csv
- Entrée : DOSSIER contient dmop.csv, evtf.csv, ftl.csv, saaf.csv, ltdata.csv, power_init.csv (7 premiers jours) et heures_a_predire.csv.
- Sortie : un CSV avec la colonne ut_ms et les 33 colonnes NPWDxxxx, une ligne par heure, aucune valeur manquante.
Pour tester exactement dans les mêmes conditions :
1. python make_validation.py --annee annee2 --out validation
2. python predict.py --test_dir validation/test --out pred.csv
3. python evaluate.py --pred pred.csv --verite validation/verite_horaire.csv --heures validation/test/heures_a_predire.csv
C'est l'écart moyen entre la prédiction et la réalité, sur toutes les heures et tous les circuits. Plus c'est bas, mieux c'est. Repère : une prédiction constante donne environ 0,11 à 0,13. L'objectif est de faire nettement
Utilise ton world model pour proposer un planning qui consomme moins, en gardant les mêmes communications avec la Terre. Non noté automatiquement : tu défends tes choix à la soutenance.
- Code sur le repo GitHub (README d'installation)
- train.py : entraîne tout à partir du kit, en une commande
- predict.py : respecte l'interface ci-dessus
- Les poids du modèle entraîné
- Un rapport : fiches des données, ablations (2 pages max), courbe de dérive, analyse des erreurs
- Une démo : courbe « rêve contre réalité » sur la période de validation, avec l'incertitude
- Une soutenance : tes choix, tes échecs et ton code
World models
- [Ha et Schmidhuber 2018, World Models (article interactif)](https://worldmodels.github.io/)
- [Hafner et al. 2023, DreamerV3](https://arxiv.org/abs/2301.04104)
Séries temporelles et modèles séquentiels
- [pandas : séries temporelles, resample et merge_asof](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [PyTorch, Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [Lim et al. 2021, Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
Dérive et incertitude
- [Bengio et al. 2015, Scheduled Sampling](https://arxiv.org/abs/1506.03099)
- [Lakshminarayanan et al. 2017, Deep Ensembles](https://arxiv.org/abs/1612.01474)
Calcul gratuit
- Google Colab (GPU T4 gratuit, sauvegarde tes checkpoints sur Drive) ou Kaggle Notebooks (quota hebdomadaire). Aucun GPU personnel n'est nécessaire.