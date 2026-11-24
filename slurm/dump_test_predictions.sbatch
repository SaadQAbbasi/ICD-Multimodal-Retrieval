#!/usr/bin/env bash
#SBATCH --job-name=task2_dump_preds
#SBATCH -N 1
#SBATCH --gpus=1                 # experiments used one RTX 3090 (24 GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/task2_dump_preds-%j.out
# Diagnostics prep: save test-set scores of Round 5 runs. Args: run directories.
source slurm/_env.sh
"$PY" -u -m experiments.dump_test_predictions "$@"
