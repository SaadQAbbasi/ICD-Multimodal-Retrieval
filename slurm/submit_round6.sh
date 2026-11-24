#!/usr/bin/env bash
# Round 6: hierarchy-aware SupCon (lambda 0.001), same schedule as Round 4, frozen features.
# Compare with outputs/experiment03_supcon_w0.001_e120_cos_s<seed>.  Run from the repository root:
#   bash slurm/submit_round6.sh            # seed 42 (SEEDS="42 43 44" for more)
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p slurm_outputs
source configs/paths.env   # PYTHON, data, weights, SBATCH_ARGS (e.g. --nodelist=faretra)
COMMON=(--parsable ${SBATCH_ARGS:-}); [ -n "${DEPEND:-}" ] && COMMON+=(--dependency="$DEPEND")
ENV="ALL,EPOCHS=120,EARLY_STOP_PATIENCE=20,LR_SCHEDULE=cosine,VAL_FULL_EVERY=10,SUPCON_WEIGHT=0.001"
for SEED in ${SEEDS:-42}; do
  J=$(sbatch "${COMMON[@]}" --job-name="task2_hsupcon_s$SEED" --export="$ENV,SEED=$SEED" slurm/experiment04_hier_supcon.sbatch)
  echo "round6 hierarchy SupCon seed $SEED: $J"
done
