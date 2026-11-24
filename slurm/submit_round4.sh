#!/usr/bin/env bash
# Round 4 (main experiment): BCE vs Jaccard-SupCon (lambda 0.001) on frozen features,
# 120 epochs, cosine LR 1e-3 -> 1e-5, early-stop patience 20, full validation every 10 epochs.
# Needs the feature cache (slurm/build_cache.sbatch). Run from the repository root:
#   bash slurm/submit_round4.sh                     # seeds 42 43 44  (1.3-1.7 h per job)
#   SEEDS=42 bash slurm/submit_round4.sh            # one seed
#   DEPEND=afterok:<jobid> bash slurm/submit_round4.sh
# Extra sbatch options (partition, node, account): SBATCH_ARGS in configs/paths.env
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p slurm_outputs
source configs/paths.env   # PYTHON, data, weights, SBATCH_ARGS (e.g. --nodelist=faretra)
COMMON=(--parsable ${SBATCH_ARGS:-}); [ -n "${DEPEND:-}" ] && COMMON+=(--dependency="$DEPEND")
ENV="ALL,EPOCHS=120,EARLY_STOP_PATIENCE=20,LR_SCHEDULE=cosine,VAL_FULL_EVERY=10"
for SEED in ${SEEDS:-42 43 44}; do
  J=$(sbatch "${COMMON[@]}" --job-name="task2_bce_s$SEED" --export="$ENV,SEED=$SEED" slurm/experiment01_textfix.sbatch)
  echo "round4 BCE    seed $SEED: $J"
  J=$(sbatch "${COMMON[@]}" --job-name="task2_supcon_s$SEED" --export="$ENV,SEED=$SEED,SUPCON_WEIGHT=0.001" slurm/experiment03_supcon.sbatch)
  echo "round4 SupCon seed $SEED: $J"
done
