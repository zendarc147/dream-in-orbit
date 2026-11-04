# Ablations du world model

Configuration de base : `{"graines": [0, 1]}`

| Variante | RMSE directe | RMSE inverse | Écart vs complet |
|---|---:|---:|---:|
| complet | 0.0864 | 0.0941 | +0.0000 |
| sans éclipses | 0.0879 | 0.0957 | +0.0016 |
| sans saaf (Soleil) | 0.0870 | 0.0989 | +0.0027 |
| sans dmop (commandes) | 0.0885 | 0.0958 | +0.0019 |
| sans mémoire (ni encodeur ni GRU) | 0.0891 | 0.1055 | +0.0071 |

Note : world model v1 seul (bruit 0.3, 600 pas, EMA 0.99, correction de niveau apprise),
2 graines par direction. « Sans mémoire » retire aussi la correction de niveau, qui lit
power_init. Avec la nouvelle configuration, les commandes dmop pèsent plus qu'avant
(+0.0019 contre +0.0009) et la mémoire encore plus (+0.0071 contre +0.0049).
