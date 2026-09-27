#!/usr/bin/env bash
#SBATCH --job-name=task2_exp01
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --exclude=moro232
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/experiment01-%j.out

set -euo pipefail

TASK2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "$TASK2_DIR/../.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

cd "$TASK2_DIR"
mkdir -p slurm_outputs outputs/experiment01 cache .cache/huggingface .cache/torch .cache/wandb pretrained wandb

export PYTHONPATH="$TASK2_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$TASK2_DIR/.cache/huggingface"
export TORCH_HOME="$TASK2_DIR/.cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="$TASK2_DIR"
export WANDB_CACHE_DIR="$TASK2_DIR/.cache/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-mimic-icd-task2}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-experiment01-classification-${SLURM_JOB_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"

"$TASK2_DIR/setup_experiment01.sh"

BUILD_CACHE_ARGS=()
if [[ ! -f "$TASK2_DIR/cache/experiment01_backbone_embeddings.pt" ]]; then
  BUILD_CACHE_ARGS+=(--build-cache)
fi

"$PYTHON_BIN" -m experiments.experiment01 \
  "${BUILD_CACHE_ARGS[@]}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --min-label-frequency "${MIN_LABEL_FREQUENCY:-5}" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-mode "$WANDB_MODE"

