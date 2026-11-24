"""Exp4: unweighted BCE + hierarchy-aware Jaccard SupCon.

Identical to experiment03_supcon (text-fixed cache, architecture, schedule)
except the SupCon positive weight between two admissions: instead of the
Jaccard overlap of their exact codes, it is the depth-weighted mean of the
Jaccard overlaps at each hierarchy level (L1 ... L5 and the exact code; level
weights 1..6, normalised). Diagnoses use the ICD-9 path, procedures the
ICD-10-PCS character prefixes (see experiment01.build_hierarchy_levels).

    SUPCON_WEIGHT=0.001 EPOCHS=120 LR_SCHEDULE=cosine SEED=42 python -m experiments.experiment04_hier_supcon
"""
import os
from pathlib import Path

from experiments.experiment01 import run_experiment
from experiments.experiment01_textfix import base_configuration, run_overrides

ROOT = Path(__file__).resolve().parents[1]


def configuration(weight: float | None = None, temperature: float | None = None):
    weight = float(os.getenv("SUPCON_WEIGHT", "0.001")) if weight is None else weight
    temperature = float(os.getenv("SUPCON_TEMPERATURE", "0.1")) if temperature is None else temperature
    overrides, suffix = run_overrides()
    tag = f"w{weight:g}{suffix}"
    return base_configuration(
        **overrides,
        supcon_weight=weight,
        supcon_temperature=temperature,
        supcon_positive_mode="hierarchy",
        output_dir=str(ROOT / f"outputs/experiment04_hsupcon_{tag}"),
        wandb_run_name=os.getenv("WANDB_RUN_NAME", f"experiment04-hsupcon-{tag.replace('_', '-')}"),
    )


if __name__ == "__main__":
    run_experiment(configuration())
