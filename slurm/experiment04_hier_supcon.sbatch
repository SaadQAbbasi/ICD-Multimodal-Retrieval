#!/usr/bin/env bash
#SBATCH --job-name=task2_hsupcon
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/task2_hsupcon-%j.out
# Exp4 BCE + hierarchy-aware SupCon (frozen features). Env: SUPCON_WEIGHT + the Exp1 overrides
source slurm/_env.sh
"$PY" -u -m experiments.experiment04_hier_supcon
OUT_DIR=$("$PY" -c "from experiments.experiment04_hier_supcon import configuration; print(configuration().output_dir)")
"$PY" -u -m experiments.retrieval_eval "$OUT_DIR"
