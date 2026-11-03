# Ablations du world model

Configuration de base : `{"graines": [0, 1]}`

| Variante | RMSE directe | RMSE inverse | Écart vs complet |
|---|---:|---:|---:|
| complet | 0.0884 | 0.0978 | +0.0000 |
| sans éclipses | 0.0913 | 0.1014 | +0.0033 |
| sans saaf (Soleil) | 0.0904 | 0.1016 | +0.0029 |
| sans dmop (commandes) | 0.0900 | 0.0981 | +0.0009 |
| sans mémoire (ni encodeur ni GRU) | 0.0901 | 0.1059 | +0.0049 |
