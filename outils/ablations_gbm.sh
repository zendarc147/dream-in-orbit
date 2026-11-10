#!/usr/bin/env bash
# Phase 2 : chaque variante du boosting B_V2 est entraînée et notée dans les deux directions
# (valider.py), puis mélangée aux deux groupes de graines du GRU de référence (melanger.py).
#   bash outils/ablations_gbm.sh NOM 'OPTIONS_JSON' [NOM 'OPTIONS_JSON' ...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while [ $# -ge 2 ]; do
  nom=$1 options=$2; shift 2
  if [ -f "runs/$nom/directe/gbm_sacs.pkl" ] && [ -f "runs/$nom/inverse/gbm_sacs.pkl" ]; then
    [ -f "runs/$nom/scores.json" ] || $PY outils/valider.py --nom "$nom" --sans-entrainement
  else
    $PY outils/valider.py --nom "$nom" --modele gbm_sacs --options "$options"
  fi
  $PY outils/melanger.py --gru runs/p1_gru_g04 runs/p1_gru_g59 --autre "runs/$nom" --nom "${nom}_mel" \
    --poids 0 0.1 0.2 0.3 0.4 0.5 | tail -4
done
