"""Exp3: unweighted BCE + Jaccard-weighted supervised contrastive loss.

Same data, cache (text-fixed), architecture and schedule as
experiment01_textfix. The only change is the extra loss term
``supcon_weight * jaccard_supcon_loss`` over the image, text and fused
admission embeddings (see experiments/experiment01.py).

    SUPCON_WEIGHT=0.1 python -m experiments.experiment03_supcon
"""
import os
from pathlib import Path

from experiments.experiment01 import run_experiment
from experiments.experiment01_textfix import base_configuration, run_overrides

ROOT = Path(__file__).resolve().parents[1]


def configuration(weight: float | None = None, temperature: float | None = None):
    weight = float(os.getenv("SUPCON_WEIGHT", "0.1")) if weight is None else weight
    temperature = (
        float(os.getenv("SUPCON_TEMPERATURE", "0.1"))
        if temperature is None
        else temperature
    )
    overrides, suffix = run_overrides()
    tag = f"w{weight:g}{suffix}"
    return base_configuration(
        **overrides,
        supcon_weight=weight,
        supcon_temperature=temperature,
        output_dir=str(ROOT / f"outputs/experiment03_supcon_{tag}"),
        wandb_run_name=os.getenv("WANDB_RUN_NAME", f"experiment03-supcon-{tag.replace('_', '-')}"),
    )


if __name__ == "__main__":
    run_experiment(configuration())
