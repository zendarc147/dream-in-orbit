# Ancien `poids/` (structure S2 assemblée), remplacé le 17 novembre 2026

Copie du `poids/` livré jusqu'au commit `af5432a`. Ce dossier était lui-même une copie de
`../gru_boucle/S2/` (seule `modele.json` diffère). Ses pièces venaient de plusieurs runs : les GRU à 10 graines
fusionnaient deux entraînements de 5 graines (0–4 et 5–9), le world model physique NPWD2532 datait du 11 novembre,
les boostings et la recette B de l'installation de T3.

Remplacé par la sortie d'un seul `python train.py` (annee1 + annee2, Kaggle `zendarc147`, T4, 4936 s).

| Comparaison avec le nouveau `poids/` | Résultat |
|---|---|
| Prédictions `predict.py` sur `validation/test` et `validation_inverse/test` (final et chaque composant : GRU S, GRU à boucle, part réseau, boosting, mélange) | identiques à l'octet près |
| `recette/recette.pt`, `recette2/recette.pt`, `specialiste/b_NPWD2551/`, `marqueurs.json`, `ensemble.json` | identiques à l'octet près |
| `specialiste/physique/membre*/monde.pt` | 137 tenseurs identiques (écart max 0.0) ; seule la `config` diffère : quatre clés d'options retirées au nettoyage (`regime_*`, `lambda_regime`, `abandon_regime_contexte`) |
| `gbm/gbm_sacs.pkl.xz` | même contenu de prédiction ; l'ancien fichier gardait une clé `perte` retirée au nettoyage |

Verdict : ancien livré, gardé pour mémoire. Prédiction : `python predict.py --test_dir … --out … --poids poids_candidats/assemblage_s2`.
