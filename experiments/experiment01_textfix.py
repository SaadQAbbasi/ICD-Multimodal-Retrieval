"""Exp1 (BCE baseline): frozen DINOv3 + MedCLIP features, unweighted BCE.

This is the plain-BCE arm of every comparison in the project. All other
experiments (Exp3 SupCon, Exp4 hierarchy SupCon) start from
``base_configuration()`` and change only the loss.

Hyper-parameters come from ``configs/base_config.json``; file locations are the
relative defaults of ``ExperimentConfig`` (resolved against this folder) and
the dataset paths can be overridden with ``MIMIC_ROOT`` / ``DATASET_DIR``.

Optional environment overrides (used by the Slurm submit scripts):
    EPOCHS, SEED, EARLY_STOP_PATIENCE, LR_SCHEDULE (constant|cosine|plateau),
    LR_MIN_RATIO, VAL_FULL_EVERY, WANDB_MODE / WANDB_ENTITY / WANDB_PROJECT.

Round 4 BCE arm, seed 42 (main experiment):
    EPOCHS=120 EARLY_STOP_PATIENCE=20 LR_SCHEDULE=cosine VAL_FULL_EVERY=10 SEED=42 \\
        python -m experiments.experiment01_textfix
    -> outputs/experiment01_admission_unweighted_textfix_e120_cos_s42/

("textfix" = the MedCLIP text features use attention-masked mean pooling; the
original MedCLIP code also averaged padding tokens, which made a report's
embedding depend on the other reports in its batch.)
"""
import json
import os
from pathlib import Path

from experiments.experiment01 import ExperimentConfig, run_experiment

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/base_config.json"


def run_overrides():
    """Read the optional environment overrides.

    Returns (config overrides, output/run-name suffix). With none set the
    suffix is empty, e.g. EPOCHS=120 LR_SCHEDULE=cosine SEED=42 -> "_e120_cos_s42".
    """
    overrides, parts = {}, []
    if os.getenv("EPOCHS"):
        overrides["epochs"] = int(os.environ["EPOCHS"])
        parts.append(f"e{overrides['epochs']}")
    if os.getenv("EARLY_STOP_PATIENCE"):
        overrides["early_stopping_patience"] = int(os.environ["EARLY_STOP_PATIENCE"])
    if os.getenv("LR_SCHEDULE"):  # constant | cosine | plateau
        overrides["lr_schedule"] = os.environ["LR_SCHEDULE"]
        if overrides["lr_schedule"] != "constant":
            parts.append({"cosine": "cos"}.get(overrides["lr_schedule"], overrides["lr_schedule"]))
    if os.getenv("LR_MIN_RATIO"):
        overrides["lr_min_ratio"] = float(os.environ["LR_MIN_RATIO"])
    if os.getenv("VAL_FULL_EVERY"):  # cheap validation in between (metrics unchanged -> no suffix)
        overrides["validation_full_metrics_every"] = int(os.environ["VAL_FULL_EVERY"])
    if os.getenv("SEED"):
        overrides["seed"] = int(os.environ["SEED"])
        parts.append(f"s{overrides['seed']}")
    return overrides, "".join("_" + p for p in parts)


def base_configuration(**overrides) -> ExperimentConfig:
    """Shared configuration of all frozen-encoder experiments."""
    values = json.loads(BASE_CONFIG.read_text())
    values.update(
        task2_root=str(ROOT),
        wandb_entity=os.getenv("WANDB_ENTITY") or None,
        wandb_project=os.getenv("WANDB_PROJECT", values["wandb_project"]),
        wandb_mode=os.getenv("WANDB_MODE", "disabled"),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def configuration() -> ExperimentConfig:
    overrides, suffix = run_overrides()
    return base_configuration(
        **overrides,
        output_dir=str(ROOT / f"outputs/experiment01_admission_unweighted_textfix{suffix}"),
        wandb_run_name=os.getenv(
            "WANDB_RUN_NAME",
            f"experiment01-admission-unweighted-textfix{suffix.replace('_', '-')}",
        ),
    )


if __name__ == "__main__":
    run_experiment(configuration())
