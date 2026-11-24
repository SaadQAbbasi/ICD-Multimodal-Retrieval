#!/usr/bin/env bash
# One command for all main experiments, chained with Slurm dependencies:
#   build_cache -> Round 4 (2 arms x 3 seeds) -> Round 6 (seed 42)
#   resize_images ----------------------------> Round 5 (4 arms, seed 42; after Round 4 SupCon s42)
# Run from the repository root after setup_weights.sh:   bash slurm/reproduce_all.sh
# Measured on one RTX 3090: resize 10 min (CPU), Round 4 6 x 1.3-1.7 h, Round 6 ~1.5 h, Round 5 4 x 0.4-1 h.
# The feature cache was not timed separately (the first job that built it plus 20 epochs took ~2 h).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p slurm_outputs
source configs/paths.env   # PYTHON, data, weights, SBATCH_ARGS (e.g. --nodelist=faretra)
A=(--parsable ${SBATCH_ARGS:-})
CACHE=$(sbatch "${A[@]}" slurm/build_cache.sbatch);  echo "cache:  $CACHE"
RESIZE=$(sbatch "${A[@]}" slurm/resize_images.sbatch); echo "resize: $RESIZE"
R4=$(DEPEND="afterok:$CACHE" bash slurm/submit_round4.sh); echo "$R4"
DEPEND="afterok:$CACHE" bash slurm/submit_round6.sh
SUP42=$(echo "$R4" | awk '/SupCon seed 42/ {print $NF}')
DEPEND="afterok:$RESIZE:$SUP42" bash slurm/submit_round5.sh
