"""Write a small random subset of the official splits, for a quick end-to-end test.

Keeps only admissions that the experiments would also keep (at least one local
JPG, see ``load_admission_records``), samples ``--train/--validation/--test``
of them per split (patients never cross splits, because the official splits are
already patient-disjoint), and copies *all* rows of the sampled admissions to
``<out_dir>/{train,validation,test}.jsonl``. Point ``DATASET_DIR`` at
``<out_dir>`` (and ``CACHE_PATH`` at a separate cache file) to run any
frozen-encoder experiment on the subset.

Used by slurm/quick_test.sbatch:
    python -m experiments.make_quick_subset cache/quick_test/splits --train 400 --validation 100 --test 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from experiments.experiment01 import SPLITS, load_admission_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-dir", default=os.getenv(
        "DATASET_DIR", "/datasets/MIMIC-GEN-RAG/edin_splits/mimic-iv_icd10/multimodal"))
    parser.add_argument("--mimic-root", default=os.getenv("MIMIC_ROOT", "/datasets/MIMIC-CXR"))
    args = parser.parse_args()

    loaded = load_admission_records(args.dataset_dir, args.mimic_root)
    sizes = {"train": args.train, "validation": args.validation, "test": args.test}
    rng = random.Random(args.seed)
    keep: set[int] = set()
    for split in SPLITS:
        ids = sorted(r.hadm_id for r in loaded.records if r.split == split)
        keep.update(rng.sample(ids, min(sizes[split], len(ids))))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = admissions = 0
        seen: set[int] = set()
        with open(Path(args.dataset_dir) / f"{split}.jsonl", encoding="utf-8") as src, \
                open(args.out_dir / f"{split}.jsonl", "w", encoding="utf-8") as dst:
            for line in src:
                hadm_id = int(json.loads(line)["hadm_id"])
                if hadm_id in keep:
                    dst.write(line)
                    rows += 1
                    if hadm_id not in seen:
                        seen.add(hadm_id)
                        admissions += 1
        print(f"{split:10s} {admissions:5d} admissions, {rows:6d} (admission, code) rows")
    print("subset written to", args.out_dir.resolve())


if __name__ == "__main__":
    main()
