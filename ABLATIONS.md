# Ablations du world model

Configuration de base : `{"graines": [0, 1]}`

| Variante | RMSE directe | RMSE inverse | Écart vs complet |
|---|---:|---:|---:|
| complet | 0.0840 | 0.0918 | +0.0000 |
| sans éclipses | 0.0844 | 0.0910 | -0.0002 |
| sans saaf (Soleil) | 0.0865 | 0.0950 | +0.0028 |
| sans commandes AOOO/ATTT (branche vers NPWD2881) | 0.0857 | 0.0939 | +0.0019 |
| sans dmop (commandes) | 0.0885 | 0.0944 | +0.0035 |
| sans mémoire (ni encodeur ni GRU) | 0.0884 | 0.1032 | +0.0079 |

Note : world model v1 seul (bruit 0.3, 600 pas, EMA 0.99, commandes AOOO/ATTT dans une
branche vers NPWD2881, correction de niveau sur le résidu avec un terme rapide et un terme
lent), 2 graines par direction, sur GPU T4. « Sans mémoire » retire aussi la correction de
niveau, qui lit power_init. « Sans commandes AOOO/ATTT » retire seulement la branche
(+0.0019) ; « sans dmop » retire en plus les comptes de prepare.py (+0.0035 au total,
contre +0.0019 avant la branche). Les éclipses ne pèsent plus rien ici (−0.0002, sous le
bruit de graine) : la correction sur le résidu et la saison (saaf, ltdata) portent déjà
l'essentiel de ce qu'elles apportaient (+0.0016 dans la configuration précédente).
