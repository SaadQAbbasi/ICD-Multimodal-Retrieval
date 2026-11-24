#!/usr/bin/env bash
#SBATCH --job-name=task2_bce
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/task2_bce-%j.out
# Exp1 BCE arm (frozen features). Env: EPOCHS SEED EARLY_STOP_PATIENCE LR_SCHEDULE VAL_FULL_EVERY
source slurm/_env.sh
"$PY" -u -m experiments.experiment01_textfix
OUT_DIR=$("$PY" -c "from experiments.experiment01_textfix import configuration; print(configuration().output_dir)")
"$PY" -u -m experiments.retrieval_eval "$OUT_DIR"    # adds retrieval.json
