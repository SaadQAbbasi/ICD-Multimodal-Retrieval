#!/usr/bin/env bash
#SBATCH --job-name=task2_quick_test
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/task2_quick_test-%j.out
# End-to-end check of the main-experiment pipeline in ~2 min on one GPU (~6 min on CPU):
# random subset of 400 / 100 / 100 admissions -> feature cache -> BCE and BCE + Jaccard SupCon (2 epochs each)
# -> test metrics -> retrieval. The scores are NOT meaningful (tiny data); the check is that every step runs.
# Writes only inside the repository: cache/quick_test/ and outputs/*_e2_cos_s42/ (safe to delete).
#   source configs/paths.env && sbatch $SBATCH_ARGS slurm/quick_test.sbatch
source slurm/_env.sh
QT="$TASK2_DIR/cache/quick_test"
"$PY" -u -m experiments.make_quick_subset "$QT/splits" --dataset-dir "$DATASET_DIR" --train 400 --validation 100 --test 100
export DATASET_DIR="$QT/splits" CACHE_PATH="$QT/features.pt"            # the subset, never the real cache
export EPOCHS=2 EARLY_STOP_PATIENCE=0 LR_SCHEDULE=cosine VAL_FULL_EVERY=1 SEED=42 WANDB_MODE=disabled
"$PY" -u -m experiments.build_cache
"$PY" -u -m experiments.experiment01_textfix  > "$QT/train_bce.log"
SUPCON_WEIGHT=0.001 "$PY" -u -m experiments.experiment03_supcon > "$QT/train_supcon.log"
for RUN in outputs/experiment01_admission_unweighted_textfix_e2_cos_s42 outputs/experiment03_supcon_w0.001_e2_cos_s42; do
  "$PY" -u -m experiments.retrieval_eval "$RUN" > "$RUN/retrieval_stdout.log"
done
"$PY" - <<'PY'
import json
for name, run in [("BCE", "experiment01_admission_unweighted_textfix_e2_cos_s42"),
                  ("BCE + Jaccard SupCon", "experiment03_supcon_w0.001_e2_cos_s42")]:
    r = json.load(open(f"outputs/{run}/results.json")); q = json.load(open(f"outputs/{run}/retrieval.json"))
    f = r["test"]["modalities"]["fused"]
    print(f"{name:21s} epochs {len(r['history'])}  test fused micro AP {f['micro_ap']:.4f}  P@5 {f['precision_at_5']:.4f}  "
          f"retrieval Jaccard@10 {q['fused']['mean_jaccard']:.4f} (random {q['random']['mean_jaccard']:.4f})")
print("QUICK TEST PASSED: every step of the pipeline ran (scores on 400 training admissions are not meaningful).")
PY
