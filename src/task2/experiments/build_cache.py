"""Step 1: encode every study once with the frozen backbones and cache the features.

For each retained admission's studies:
  * image study feature = L2-normalised mean of the L2-normalised DINOv3 ViT-S/16
    CLS tokens of its JPGs (384-d);
  * text study feature  = MedCLIP text embedding of the radiology report
    (attention-masked mean of BERT hidden states 1, 2, 12 -> 768->512 projection).

All frozen-encoder experiments (Exp1/3/4, Rounds 4 and 6) train only the small
heads on top of this cache (~6 s per epoch). Output:
    cache/experiment01_admission_backbone_embeddings_textfix.pt  (~70 MB, fp16)

Usage (from the repository root; GPU recommended; encodes every study of the retained admissions):
    python -m experiments.build_cache            # skip if the cache exists
    python -m experiments.build_cache --force    # rebuild
"""
import argparse

from experiments.experiment01 import build_backbone_cache, load_admission_records
from experiments.experiment01_textfix import base_configuration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()
    config = base_configuration().resolve()
    loaded = load_admission_records(config.dataset_dir, config.mimic_root)
    cache = build_backbone_cache(list(loaded.records), config, force=args.force)
    print("cache ready:", config.cache_path)
    print({k: cache["metadata"].get(k) for k in ("num_image_studies", "num_reports", "image_dim", "text_dim")})


if __name__ == "__main__":
    main()
