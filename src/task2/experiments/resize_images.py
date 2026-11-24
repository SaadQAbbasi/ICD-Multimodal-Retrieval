"""Round 5 prep: resize every retained CXR JPG once to 256x256 uint8.

Reproduces exactly the first part of DINOv3ImageProcessor (experiment01):
PIL open -> convert("RGB") -> v2.ToImage -> v2.Resize((256, 256), antialias=True),
i.e. the uint8 tensor right before ToDtype/Normalize. MIMIC-CXR JPGs are
grayscale, so R == G == B after the resize and only one channel is stored
(the loader repeats it to 3 channels). Any image whose channels differ is
counted and reported (it would not be reproduced exactly).

Output (cache/images_256/):
  images_256.uint8.npy  [N, 256, 256] uint8 (np.load(..., mmap_mode="r"))
  index.json            {"paths": [...], "size": 256, "non_gray": [...]}

    python -m experiments.resize_images [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from experiments.experiment01 import ExperimentConfig, load_admission_records

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "cache/images_256"
SIZE = 256


class ResizeDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = v2.Compose(
            [v2.ToImage(), v2.Resize((SIZE, SIZE), antialias=True)]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            pixels = self.transform(image.convert("RGB"))  # uint8 [3, S, S]
        gray = bool(torch.equal(pixels[0], pixels[1]) and torch.equal(pixels[0], pixels[2]))
        return index, pixels[0].numpy(), gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    parser.add_argument("--workers", type=int, default=int(os.getenv("SLURM_CPUS_PER_TASK", "8")))
    args = parser.parse_args()

    config = ExperimentConfig(task2_root=str(ROOT)).resolve()
    loaded = load_admission_records(config.dataset_dir, config.mimic_root)
    paths = sorted({p for r in loaded.records for s in r.studies for p in s.image_paths})
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} unique images from {len(loaded.records)} admissions", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if not args.limit else f".limit{args.limit}"
    array_path = OUT_DIR / f"images_256.uint8{suffix}.npy"
    tmp_path = array_path.with_name(array_path.stem + ".tmp.npy")
    array = open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(len(paths), SIZE, SIZE))

    loader = DataLoader(
        ResizeDataset(paths), batch_size=64, num_workers=args.workers,
        collate_fn=lambda items: items,
    )
    non_gray, done, start = [], 0, time.time()
    for batch in loader:
        for index, pixels, gray in batch:
            array[index] = pixels
            if not gray:
                non_gray.append(paths[index])
        done += len(batch)
        if done % 2048 < len(batch):
            rate = done / (time.time() - start)
            print(f"{done}/{len(paths)}  {rate:.0f} img/s  eta {(len(paths)-done)/rate/60:.1f} min", flush=True)
    array.flush()
    del array
    os.replace(tmp_path, array_path)
    index_path = OUT_DIR / f"index{suffix}.json"
    index_path.write_text(json.dumps({"paths": paths, "size": SIZE, "non_gray": non_gray}))
    print(f"wrote {array_path} ({array_path.stat().st_size/1e9:.2f} GB), "
          f"non-gray images: {len(non_gray)}, {time.time()-start:.0f} s")


if __name__ == "__main__":
    main()
