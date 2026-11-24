#!/usr/bin/env bash
# Round 5: partial encoder fine-tuning (last 2 DINOv3 blocks / last 2 BERT layers), warm start
# from Round 4 SupCon of the same seed. Arms: frozen control / image / text / both (0.4-1 h each on an RTX 3090).
# Needs cache/images_256 (slurm/resize_images.sbatch) and outputs/experiment03_supcon_w0.001_e120_cos_s<seed>/best.pt.
# Run from the repository root:
#   SMOKE=1 bash slurm/submit_round5.sh     # 2-epoch smoke test on tiny subsets ("both")
#   bash slurm/submit_round5.sh             # 4 arms, seed 42
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p slurm_outputs
source configs/paths.env   # PYTHON, data, weights, SBATCH_ARGS (e.g. --nodelist=faretra)
COMMON=(--parsable ${SBATCH_ARGS:-}); [ -n "${DEPEND:-}" ] && COMMON+=(--dependency="$DEPEND")
if [ "${SMOKE:-0}" = 1 ]; then
  J=$(sbatch "${COMMON[@]}" --time=01:00:00 --job-name=task2_ft_smoke \
      --export="ALL,SMOKE=1,FT_IMAGE=1,FT_TEXT=1,SEED=42" slurm/experiment05_finetune.sbatch)
  echo "round5 smoke: $J"; exit 0
fi
for SEED in ${SEEDS:-42}; do
  for ARM in "frozen 0 0" "img 1 0" "txt 0 1" "both 1 1"; do
    set -- $ARM
    J=$(sbatch "${COMMON[@]}" --job-name="task2_ft_$1_s$SEED" \
        --export="ALL,FT_IMAGE=$2,FT_TEXT=$3,SEED=$SEED" slurm/experiment05_finetune.sbatch)
    echo "round5 $1 seed $SEED: $J"
  done
done
