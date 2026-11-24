#!/usr/bin/env bash
#SBATCH --job-name=task2_resize256
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/task2_resize256-%j.out
# Round 5 prep (CPU only): resize every retained JPG to 256x256 uint8 -> cache/images_256/ (~2.3 GB)
source slurm/_env.sh
"$PY" -u -m experiments.resize_images --workers "${SLURM_CPUS_PER_TASK:-8}"
