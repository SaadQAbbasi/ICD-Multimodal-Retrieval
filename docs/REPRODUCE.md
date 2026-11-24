# Task 2 — reproduction guide

Detailed companion to the [README](../README.md): every file's role, setup, and the exact commands behind each
reported result. All commands run from the repository root.

## 2. Repository structure

```
README.md                      overview and results
requirements.txt               pinned Python packages (Python 3.12; full freeze in requirements_full_freeze.txt)
setup_weights.sh               downloads the MedCLIP + DINOv3 weights and the pinned DINOv3 source code
configs/base_config.json       shared hyper-parameters of all frozen-encoder experiments
configs/paths.env              ALL paths (Python, MIMIC data, pretrained models, cache, sbatch options); the only file to edit
experiments/                   all code (run as modules from the repository root: python -m experiments.<name>)
slurm/                         sbatch job scripts + submit scripts for each round + reproduce_all.sh
notebooks/results_overview.ipynb   documented walkthrough of the results (reads results/)
docs/task2_presentation.pptx   slides of the oral presentation (11 slides, speaker notes included)
docs/task2_final_report.pdf    full written report (method, all experiments, diagnostics, limitations)
docs/REPRODUCE.md              this guide
docs/progression.png           results figure of the README
results/                       metrics of the runs in the README / report (copied from outputs/)
slurm_outputs/                 Slurm log folder (empty)
```
Created when you run the code: `pretrained/`, `vendor/dinov3/` (setup), `cache/` (features, resized images),
`outputs/<run>/` (one folder per run), `.cache/` (Hugging Face downloads).

### Role of each file

**Core code (`experiments/`)**

| file | role |
|---|---|
| `experiment01.py` | **Main library** used by every experiment. `ExperimentConfig` (all settings, relative paths), data loading (`load_admission_records`: groups the split rows into admissions and links reports and JPGs), frozen encoders (`_load_dinov3_image_encoder`, `MedCLIPTextEncoder` with the masked-pooling fix), feature cache (`build_backbone_cache`), model (`MultimodalICDClassifier`), losses (`classification_loss`, `jaccard_supcon_loss`, `hierarchy_pair_weights`), training loop with LR schedule and early stopping, evaluation (`multilabel_metrics`: micro/macro AP, F1, AUROC, P@k, recall@k; `_cm_hierarchy_metrics`: per ICD level), `run_experiment` (train → select best epoch on validation fused micro AP → test). |
| `build_cache.py` | **Step 1.** Encodes every study once with the frozen encoders → `cache/experiment01_admission_backbone_embeddings_textfix.pt`. |
| `experiment01_textfix.py` | **Exp1, BCE arm.** Loads `configs/base_config.json`, applies environment overrides (`EPOCHS`, `SEED`, `LR_SCHEDULE`, …) and trains. Also defines `base_configuration()` / `run_overrides()`, reused by Exp3 and Exp4. |
| `experiment03_supcon.py` | **Exp3, BCE + Jaccard-weighted SupCon** (`SUPCON_WEIGHT`, default temperature 0.1). |
| `experiment04_hier_supcon.py` | **Exp4, BCE + hierarchy-aware SupCon** (`supcon_positive_mode="hierarchy"`). |
| `make_quick_subset.py` | Writes a small random subset of the splits (only admissions the experiments keep) for a quick end-to-end test; used by `slurm/quick_test.sbatch` together with `CACHE_PATH`. |
| `compare_with_shipped.py` | Prints re-run results (`outputs/<run>/`) next to the shipped ones (`results/<run>/`): test fused micro AP, P@5, retrieval Jaccard@10, flag if the AP differs by more than 0.003. |
| `retrieval_eval.py` | Retrieval evaluation of a trained run (k = 10 nearest training admissions): mean Jaccard, nDCG, kNN-vote micro AP, random reference → `<run>/retrieval.json`. The sbatch scripts call it after training. |
| `resize_images.py` | Round 5 prep: resizes every JPG once to 256×256 uint8 (identical to the encoder preprocessing) → `cache/images_256/`. |
| `experiment05_finetune.py` | **Round 5.** Trains from images and report text, unfreezes the last `FT_BLOCKS` (2) DINOv3 blocks and/or BERT layers, warm-starts the head from the Round 4 SupCon run of the same seed, checks that step-0 features equal the cache. Arms are set by `FT_IMAGE` / `FT_TEXT`. |
| `frequency_baseline.py` | Frequency baseline (each code scored by its training prevalence) on exact codes and ICD levels, compared with one run. |
| `dump_test_predictions.py` | Saves per-admission test probabilities of Round 5 runs (`test_predictions.npz`, input of the diagnostics). |
| `diagnostic_code_groups.py` | Diagnostic 1: AP and lift over the baseline per code group (chapter, chest-visible set, procedures, training frequency). |
| `diagnostic_visibility_sensitivity.py` | Sensitivity check for Diagnostic 1: share and lift of chest-visible diagnoses with a narrow vs broad hand-picked list, and of the codes with no direct radiological sign. |
| `diagnostic_chest_findings.py` | Diagnostic 2: chest ICD categories vs CheXpert findings in the reports (agreement, exact-code vs category AP, "perfect report reader" reference). |

**Jobs and scripts (`slurm/`)**

| file | role |
|---|---|
| `_env.sh` | Shared job environment: loads `configs/paths.env`, sets the working dir, `PYTHONPATH`, HF cache, `PY=$PYTHON`. |
| `quick_test.sbatch` | ~2-minute end-to-end check: subset of 400/100/100 admissions → feature cache → BCE and SupCon (2 epochs) → test → retrieval; ends with `QUICK TEST PASSED`. |
| `build_cache.sbatch` | Runs `experiments.build_cache`. |
| `experiment01_textfix.sbatch`, `experiment03_supcon.sbatch`, `experiment04_hier_supcon.sbatch` | Train one frozen-encoder run, then `retrieval_eval`. |
| `resize_images.sbatch`, `experiment05_finetune.sbatch` | Round 5 prep and one fine-tuning arm. |
| `dump_test_predictions.sbatch` | Test predictions for the diagnostics. |
| `submit_round4.sh` / `submit_round5.sh` / `submit_round6.sh` | Submit all jobs of one round (seeds/arms as used in the report). |
| `reproduce_all.sh` | Submits everything with dependencies: cache → Round 4 → Round 6, resize + Round 4 SupCon s42 → Round 5. |

**Results (`results/`)**

| file | content |
|---|---|
| `<run>/results.json` | Per-epoch history (train losses, LR, validation metrics), best epoch, test metrics per modality and per ICD level. |
| `<run>/retrieval.json` | Retrieval scores. |
| `<run>/config.json` | Exact configuration of the run plus split statistics. |
| `all_results_comparison.csv` | One row per (run, modality, cohort, ICD level) for every run of the project, including earlier rounds and the baseline. |
| `diagnostics/*.md, *.json` | Output of the diagnostic scripts. |

Run folders: `experiment01_admission_unweighted_textfix_e120_cos_s{42,43,44}` (Round 4 BCE),
`experiment03_supcon_w0.001_e120_cos_s{42,43,44}` (Round 4 SupCon), `experiment05_ft_supcon_w0.001_{frozen,img,txt,both}_s42`
(Round 5), `experiment04_hsupcon_w0.001_e120_cos_s42` (Round 6). Model checkpoints and per-patient predictions are
**not** included, because the MIMIC data-use agreement forbids sharing patient-level data and the files are large.

---

## 3. Reproducing the experiments

Training runs as Python scripts (`experiments/`), submitted as Slurm GPU jobs (`slurm/`); the notebook only reads
results. **All paths are in one file, `configs/paths.env`**: Python interpreter, MIMIC data, the pretrained models and
the Slurm options. Every job and submit script loads it. Its defaults are the paths on the project machine (`faretra`),
so there nothing needs editing.

### 3.0 Quick start on the project machine (`faretra`)
```bash
git clone https://github.com/SaadQAbbasi/ICD-Multimodal-Retrieval.git && cd ICD-Multimodal-Retrieval
source configs/paths.env                     # faretra defaults: project venv, /datasets/..., downloaded models

# 1. end-to-end check of the pipeline, ~2 min on one GPU (400/100/100 admissions, 2 epochs, scores not meaningful)
sbatch $SBATCH_ARGS slurm/quick_test.sbatch
tail -3 slurm_outputs/task2_quick_test-*.out  # ends with "QUICK TEST PASSED ..."

# 2. main experiment (Round 4), one seed: feature cache, then BCE and BCE + Jaccard SupCon (~1.5 h each)
J=$(sbatch --parsable $SBATCH_ARGS slurm/build_cache.sbatch)
DEPEND=afterok:$J SEEDS=42 bash slurm/submit_round4.sh
squeue -u $USER                               # follow the jobs; logs in slurm_outputs/

# 3. compare with the shipped results (expected test fused micro AP ≈ 0.218 BCE / ≈ 0.220 SupCon)
$PYTHON -m experiments.compare_with_shipped
```
The models are read from the copies already downloaded on `faretra` (absolute paths in `configs/paths.env`), so
nothing is downloaded. To skip the feature-cache build as well, uncomment the `CACHE_PATH` line in `configs/paths.env`
(the project's existing cache is only read) and run step 2 without the dependency: `SEEDS=42 bash slurm/submit_round4.sh`.

### 3.1 Requirements
* Credentialed PhysioNet access to **MIMIC-IV** and **MIMIC-CXR-JPG / MIMIC-CXR reports**, plus the split files.
* One GPU with ≥ 12 GB (runs used an RTX 3090, 24 GB; Round 5 peaks at 11.5 GiB). Linux, Python 3.12.
* Slurm is optional. Every job is a plain `python -m ...` command (Section 3.4).

### 3.2 Setup on another machine
**Environment** (from the repository root):
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Paths: edit `configs/paths.env`.**

| variable | what to put | value on `faretra` (default) |
|---|---|---|
| `PYTHON` | interpreter of the environment above, e.g. `$PWD/.venv/bin/python` | `/home/icdmultimodalproject/project/.venv/bin/python` |
| `MIMIC_ROOT` | MIMIC-CXR-JPG folder (`files/pXX/pXXXXXXXX/sYYYYYYYY.txt` and `.../sYYYYYYYY/*.jpg`) | `/datasets/MIMIC-CXR` |
| `DATASET_DIR` | folder with `train.jsonl`, `validation.jsonl`, `test.jsonl` | `/datasets/MIMIC-GEN-RAG/edin_splits/mimic-iv_icd10/multimodal` |
| `MEDCLIP_WEIGHTS_DIR` | folder with MedCLIP's `pytorch_model.bin`, or empty = `pretrained/` | `/home/icdmultimodalproject/project/src/task2/pretrained` |
| `DINO_CHECKPOINT` | DINOv3 ViT-S/16 `.pth` file, or empty = `pretrained/dinov3-vits16/...` | `/home/icdmultimodalproject/project/src/task2/pretrained/dinov3-vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| `DINO_SOURCE_DIR` | Meta's DINOv3 code (folder with `dinov3/hub/backbones.py`), or empty = `vendor/dinov3` | `/home/icdmultimodalproject/project/src/task2/vendor/dinov3` |
| `CACHE_PATH` | optional existing feature cache; empty = `cache/experiment01_admission_backbone_embeddings_textfix.pt` (built by `build_cache.sbatch`) | empty |
| `SBATCH_ARGS` | extra `sbatch` options (partition, node, account) | `--nodelist=faretra` |

**Pretrained models: two options.**
* **(a) Use copies you already have:** put their absolute paths in `MEDCLIP_WEIGHTS_DIR`, `DINO_CHECKPOINT`,
  `DINO_SOURCE_DIR` (the default on `faretra`). Nothing is copied; the files are only read.
* **(b) Download them into the repository:** `bash setup_weights.sh`, then set those three variables to empty
  (`""`). The script downloads the MedCLIP checkpoint from its official bucket, the DINOv3 source from GitHub (pinned
  commit) and the DINOv3 ViT-S/16 weights from a Kaggle mirror. If the Kaggle download needs a login, request
  `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` from Meta (link below) and place it in `pretrained/dinov3-vits16/`.

Bio_ClinicalBERT (architecture and tokenizer of the MedCLIP text tower) is downloaded from Hugging Face on first use
in both cases.

| model | role | source | default location | variable |
|---|---|---|---|---|
| **MedCLIP** (ViT variant; only its text tower is used: Bio_ClinicalBERT + 768→512 projection) | report encoder | code: <https://github.com/RyanWangZf/MedCLIP>; checkpoint: <https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip> (unzipped → `pytorch_model.bin`) | `pretrained/pytorch_model.bin` | `MEDCLIP_WEIGHTS_DIR` (folder) |
| **Bio_ClinicalBERT** | backbone of the MedCLIP text tower (weights then replaced by MedCLIP's) | <https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT> (automatic) | Hugging Face cache | `HF_HOME` |
| **DINOv3 ViT-S/16** (LVD-1689M pretraining) | X-ray image encoder | weights: Meta, <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/> (also <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>); mirror used by `setup_weights.sh`: <https://www.kaggle.com/datasets/saadqasimabbasi/dino-v3-small-weights> | `pretrained/dinov3-vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | `DINO_CHECKPOINT` (file) |
| **DINOv3 source code** | model definition used to load the weights | <https://github.com/facebookresearch/dinov3>, commit [`6876159`](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) | `vendor/dinov3/` | `DINO_SOURCE_DIR` (folder) |

**Data format.** Each `.jsonl` row is one (admission, ICD code) pair with `subject_id`, `hadm_id`,
`icd_version_mapped`, `code_system_mapped`, `docid`, `icd_sequence` (code hierarchy) and the linked `reportids` /
`study_ids`. `diagnostic_chest_findings.py` also needs `$MIMIC_ROOT/mimic-cxr-2.0.0-chexpert.csv.gz`.

Logging to Weights & Biases is **off** by default. To enable it, set `WANDB_MODE=online` (and optionally `WANDB_ENTITY`, `WANDB_PROJECT`).

### 3.3 With Slurm (what we used)
Run from the repository root. The submit scripts and jobs load `configs/paths.env` themselves.
```bash
source configs/paths.env                  # only needed for $SBATCH_ARGS / $PYTHON in the lines below
sbatch $SBATCH_ARGS slurm/quick_test.sbatch   # optional: ~2 min end-to-end check (Section 3.0)
bash slurm/reproduce_all.sh               # everything below, chained with dependencies
```
or step by step:
```bash
J=$(sbatch --parsable $SBATCH_ARGS slurm/build_cache.sbatch)          # 1. feature cache
DEPEND=afterok:$J bash slurm/submit_round4.sh                          # 2. Round 4: BCE + SupCon, seeds 42 43 44 (6 jobs)
DEPEND=afterok:$J bash slurm/submit_round6.sh                          # 3. Round 6: hierarchy SupCon, seed 42
sbatch $SBATCH_ARGS slurm/resize_images.sbatch                         # 4a. Round 5 prep (CPU, ~10 min)
SMOKE=1 bash slurm/submit_round5.sh                                    # 4b. optional 2-epoch smoke test
bash slurm/submit_round5.sh                                            # 4c. Round 5: 4 arms (after Round 4 SupCon s42 has finished)
```
`SEEDS="42"` limits Round 4 to one seed. Measured times on one RTX 3090: Round 4 1.3–1.7 h per job, Round 6 ~1.5 h,
Round 5 0.4–1 h per arm. Logs go to `slurm_outputs/`, results to `outputs/<run>/`.

### 3.4 Without Slurm (same commands, one GPU)
```bash
source configs/paths.env && export PYTHONPATH=$PWD
$PYTHON -m experiments.build_cache

# Round 4 (main experiment), one seed; repeat with SEED=43, 44
export EPOCHS=120 EARLY_STOP_PATIENCE=20 LR_SCHEDULE=cosine VAL_FULL_EVERY=10 SEED=42
$PYTHON -m experiments.experiment01_textfix                                              # BCE
SUPCON_WEIGHT=0.001 $PYTHON -m experiments.experiment03_supcon                           # BCE + Jaccard SupCon
$PYTHON -m experiments.retrieval_eval outputs/experiment01_admission_unweighted_textfix_e120_cos_s42
$PYTHON -m experiments.retrieval_eval outputs/experiment03_supcon_w0.001_e120_cos_s42

# Round 6
SUPCON_WEIGHT=0.001 $PYTHON -m experiments.experiment04_hier_supcon
$PYTHON -m experiments.retrieval_eval outputs/experiment04_hsupcon_w0.001_e120_cos_s42
unset EPOCHS EARLY_STOP_PATIENCE LR_SCHEDULE VAL_FULL_EVERY

# Round 5 (needs the Round 4 SupCon run of the same seed)
$PYTHON -m experiments.resize_images
FT_IMAGE=0 FT_TEXT=0 SEED=42 $PYTHON -m experiments.experiment05_finetune    # frozen control
FT_IMAGE=1 FT_TEXT=0 SEED=42 $PYTHON -m experiments.experiment05_finetune    # image encoder
FT_IMAGE=0 FT_TEXT=1 SEED=42 $PYTHON -m experiments.experiment05_finetune    # text encoder
FT_IMAGE=1 FT_TEXT=1 SEED=42 $PYTHON -m experiments.experiment05_finetune    # both

# Baseline and diagnostics
$PYTHON -m experiments.frequency_baseline outputs/experiment03_supcon_w0.001_e120_cos_s42
$PYTHON -m experiments.dump_test_predictions outputs/experiment05_ft_supcon_w0.001_frozen_s42 outputs/experiment05_ft_supcon_w0.001_both_s42
$PYTHON -m experiments.diagnostic_code_groups outputs/experiment05_ft_supcon_w0.001_frozen_s42 outputs/experiment05_ft_supcon_w0.001_both_s42
$PYTHON -m experiments.diagnostic_visibility_sensitivity outputs/experiment05_ft_supcon_w0.001_both_s42
$PYTHON -m experiments.diagnostic_chest_findings outputs/experiment05_ft_supcon_w0.001_both_s42
```

### 3.5 Checking a rerun
Each run writes `outputs/<run>/results.json` (and `retrieval.json`). `$PYTHON -m experiments.compare_with_shipped`
prints every re-run folder next to the same folder in `results/` (test fused micro AP, P@5, retrieval Jaccard@10) and
flags differences above 0.003. You can also set `RESULTS_DIR = ROOT / "outputs"` in `results_overview.ipynb` to redraw
all tables and figures from your runs. Round 4 BCE seed 42 should give test fused micro AP ≈ 0.218,
SupCon ≈ 0.220. Expect small differences (third decimal) from GPU non-determinism and from rebuilding the
fp16 feature cache. The seed-to-seed std is 0.0003–0.0004. Round 5 also prints a check at start-up: its features must
match the cache (cosine ≥ 0.9999), and its epoch-0 validation AP must equal the Round 4 SupCon best validation AP.

---

## 4. Notes
* `configs/base_config.json` holds the shared settings: batch 16 admissions, AdamW LR 1e-3, weight decay 1e-4, dropout 0.1,
  embedding 512, AMP. A run's folder name records its overrides, e.g. `_e120_cos_s42` = 120 epochs, cosine LR, seed 42.
* Model selection always uses the **validation** fused micro AP. The test set is evaluated once, with the best checkpoint.
* F1 scores in `results.json` use a fixed threshold of 0.5 without tuning, so with 6,085 rare codes they are near 0.
  We report ranking metrics (AP, P@k) instead.
* `vendor/dinov3` (downloaded by the setup script) is Meta's code under the DINOv3 License. MedCLIP and Bio_ClinicalBERT keep their own licenses.
