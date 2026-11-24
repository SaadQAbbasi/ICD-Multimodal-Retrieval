#!/usr/bin/env bash
#SBATCH --job-name=task2_supcon
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/task2_supcon-%j.out
# Exp3 BCE + Jaccard SupCon (frozen features). Env: SUPCON_WEIGHT (required) + the Exp1 overrides
source slurm/_env.sh
: "${SUPCON_WEIGHT:?set SUPCON_WEIGHT}"
"$PY" -u -m experiments.experiment03_supcon
OUT_DIR=$("$PY" -c "from experiments.experiment03_supcon import configuration; print(configuration().output_dir)")
"$PY" -u -m experiments.retrieval_eval "$OUT_DIR"
