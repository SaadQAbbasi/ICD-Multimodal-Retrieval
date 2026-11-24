#!/usr/bin/env bash
#SBATCH --job-name=task2_build_cache
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/task2_build_cache-%j.out
# Step 1: frozen DINOv3 + MedCLIP features for every study -> cache/*_textfix.pt
source slurm/_env.sh
"$PY" -u -m experiments.build_cache
