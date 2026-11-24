# Shared environment for every sbatch script (sourced, not run).
# Jobs must be submitted from the repository root (the folder holding README.md).
set -euo pipefail
TASK2_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$TASK2_DIR"
source "$TASK2_DIR/configs/paths.env"   # PYTHON, data, model weights, cache (edit that file)
mkdir -p slurm_outputs
export PYTHONPATH="$TASK2_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$TASK2_DIR/.cache/huggingface}"   # Bio_ClinicalBERT tokenizer/weights
export TORCH_HOME="${TORCH_HOME:-$TASK2_DIR/.cache/torch}"
export WANDB_DIR="$TASK2_DIR"
export TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-python}"   # set in configs/paths.env
printf 'Job %s on %s; GPU=%s; python=%s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-none}" "$PY"
