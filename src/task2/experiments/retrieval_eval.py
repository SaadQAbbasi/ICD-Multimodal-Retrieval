"""Retrieval evaluation of a trained admission encoder.

For every test admission, retrieve the k nearest train admissions by cosine
similarity of the admission embedding (image, text or fused) and score how
similar their ICD code sets are:

* mean_jaccard@k    - mean Jaccard overlap between query and neighbour codes
* ndcg@k            - nDCG with Jaccard overlap as graded relevance
* knn_micro_ap      - micro AP of k-NN label voting (similarity-weighted mean of
                      neighbour label vectors), a classifier-free probe

A random-neighbour reference is reported alongside.

    python -m experiments.retrieval_eval outputs/experiment03_supcon_w0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from experiments.experiment01 import (
    ExperimentConfig,
    MultimodalICDClassifier,
    _move_batch,
    create_cached_loader,
    load_admission_records,
    split_records,
)


@torch.inference_mode()
def embed(model, loader, device):
    model.eval()
    out = {"image": [], "text": [], "fused": []}
    labels = []
    for batch in loader:
        batch = _move_batch(batch, device)
        o = model(batch["image_features"], batch["image_mask"],
                  batch["text_features"], batch["text_mask"])
        for name in out:
            out[name].append(F.normalize(o[f"{name}_embedding"].float(), dim=-1).cpu())
        labels.append(batch["labels"].cpu())
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(labels)


def jaccard_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = a @ b.T
    union = a.sum(1, keepdim=True) + b.sum(1)[None, :] - inter
    return inter / union.clamp_min(1.0)


def score_neighbours(query_labels, train_labels, neighbour_idx, weights=None):
    k = neighbour_idx.shape[1]
    rows = []
    for start in range(0, query_labels.shape[0], 512):
        q = query_labels[start:start + 512]
        idx = neighbour_idx[start:start + 512]
        jac = jaccard_matrix(q, train_labels)
        rows.append(torch.gather(jac, 1, idx))
    rel = torch.cat(rows)  # [Q, k] Jaccard of each retrieved neighbour
    discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
    dcg = (rel * discounts).sum(1)
    # Ideal DCG: best k Jaccard values over the whole train set.
    ideal = []
    for start in range(0, query_labels.shape[0], 512):
        jac = jaccard_matrix(query_labels[start:start + 512], train_labels)
        ideal.append(jac.topk(k, dim=1).values)
    idcg = (torch.cat(ideal) * discounts).sum(1)
    ndcg = (dcg / idcg.clamp_min(1e-12))[idcg > 0]
    return {"mean_jaccard": float(rel.mean()), "ndcg": float(ndcg.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    label_keys = checkpoint["label_keys"]
    label_to_index = {key: i for i, key in enumerate(label_keys)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_admission_records(config.dataset_dir, config.mimic_root)
    cache = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    loaders = {
        split: create_cached_loader(
            split_records(loaded.records, split), cache, label_to_index,
            config, training=False,
        )
        for split in ("train", "test")
    }
    model = MultimodalICDClassifier(
        image_dim=int(cache["metadata"]["image_dim"]),
        text_dim=int(cache["metadata"]["text_dim"]),
        embedding_dim=config.embedding_dim,
        num_labels=len(label_keys),
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_emb, train_labels = embed(model, loaders["train"], device)
    test_emb, test_labels = embed(model, loaders["test"], device)
    keep = test_labels.sum(1) > 0  # drop queries with only unseen codes
    test_labels = test_labels[keep]
    k = args.k

    results = {"k": k, "queries": int(keep.sum()), "gallery": int(train_labels.shape[0])}
    generator = torch.Generator().manual_seed(0)
    random_idx = torch.randint(0, train_labels.shape[0], (test_labels.shape[0], k), generator=generator)
    results["random"] = score_neighbours(test_labels, train_labels, random_idx)

    valid_codes = (test_labels.sum(0) > 0).numpy()
    for name in ("image", "text", "fused"):
        query = test_emb[name][keep]
        sims = query @ train_emb[name].T
        top = sims.topk(k, dim=1)
        metrics = score_neighbours(test_labels, train_labels, top.indices)
        weights = torch.softmax(top.values / 0.1, dim=1)
        votes = (weights.unsqueeze(-1) * train_labels[top.indices]).sum(1).numpy()
        metrics["knn_micro_ap"] = float(
            average_precision_score(test_labels.numpy()[:, valid_codes].ravel(),
                                    votes[:, valid_codes].ravel())
        )
        results[name] = metrics

    print(json.dumps(results, indent=2))
    (output_dir / "retrieval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
