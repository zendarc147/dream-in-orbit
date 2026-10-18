# Format des données : source publique ESA

Source probable de Dream in Orbit : **Mars Express Power Challenge**. Les cinq
types de contexte, les horodatages `ut_ms` et les 33 circuits `NPWD` correspondent
au sujet. L'identité du kit et son découpage ne sont pas confirmés.

## Fichiers vérifiés dans l'archive publique

CSV UTF-8, séparateur virgule, première ligne = noms des colonnes.

| Fichier renommé | Colonnes réelles | Contenu |
| --- | --- | --- |
| `power.csv` | `ut_ms`, 33 colonnes `NPWDxxxx` | Courants mesurés, cibles à prédire |
| `dmop.csv` | `ut_ms,subsystem` | Commandes des sous-systèmes |
| `ftl.csv` | `utb_ms,ute_ms,type,flagcomms` | Intervalles de pointage et communications |
| `saaf.csv` | `ut_ms,sa,sx,sy,sz` | Angles solaires |
| `evtf.csv` | `ut_ms,description` | Événements, dont éclipses |
| `ltdata.csv` | `ut_ms,sunmars_km,earthmars_km,sunmarsearthangle_deg,solarconstantmars,eclipseduration_min,occultationduration_min` | Contexte journalier |

Les timestamps sont des millisecondes Unix UTC. FTL utilise deux bornes :
`utb_ms` au début et `ute_ms` à la fin. `flagcomms` vaut `TRUE` ou `FALSE`.
Les valeurs numériques peuvent employer la notation scientifique.
La liste exacte des 33 circuits figure dans `power.csv` et `manifest.json`.

## Exemples inclus

`examples/esa_mars_express/` contient les 50 premières lignes de chacun des six
fichiers de la période publique 2008-08-22 / 2010-07-10. Ce sont de vrais extraits,
pas des données synthétiques. Les fichiers ont seulement été tronqués et renommés.

Chaque extrait commence indépendamment : 50 lignes ne représentent pas la même
durée selon le fichier. Ils ne forment donc pas une fenêtre temporelle alignée,
ni un dataset suffisant pour entraîner ou évaluer le modèle.

Ces fichiers permettent de développer les lecteurs CSV, vérifier les types et
préparer l'alignement horaire. Les données finales doivent provenir du kit autorisé.
`make_validation.py` et `evaluate.py` de la piscine ne sont pas disponibles ici.

## Provenance et téléchargement complet

- Documentation : https://kelvins.esa.int/mars-express-power-challenge/data/
- Dépôt officiel : https://doi.org/10.5281/zenodo.6327379
- Archive : https://zenodo.org/records/6327379
- Archive complète : 153 724 228 octets ; elle n'est pas incluse dans Git.
- Auteurs : Marcus Märtens et Dario Izzo, European Space Agency.
- Version : 1.0.0 ; licence CC BY 4.0, selon les métadonnées Zenodo.
- Licence : https://creativecommons.org/licenses/by/4.0/

Le manifeste conserve le chemin d'origine et le SHA-256 de chaque extrait.
La lecture partielle de l'archive a permis d'en vérifier les en-têtes réels.
Le checksum de l'archive entière n'a pas été vérifié.
