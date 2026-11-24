#!/usr/bin/env bash
#SBATCH --job-name=task2_finetune
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/task2_finetune-%j.out
# Round 5: partial encoder fine-tuning, warm start from Round 4 SupCon (same seed).
# Env: FT_IMAGE FT_TEXT (0/1) SEED EPOCHS FT_BLOCKS ENCODER_LR HEAD_LR SMOKE
source slurm/_env.sh
echo "FT_IMAGE=${FT_IMAGE:-1} FT_TEXT=${FT_TEXT:-1} SEED=${SEED:-42} SMOKE=${SMOKE:-0}"
"$PY" -u -m experiments.experiment05_finetune
