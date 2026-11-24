"""Dump per-admission test probabilities of a Round 5 (experiment05) checkpoint.

Writes <output_dir>/test_predictions.npz with hadm_ids, targets (uint8) and
image/text/fused probabilities (float16), for per-code / per-group analysis
(experiments/diagnostic_code_groups.py).

    python -m experiments.dump_test_predictions outputs/experiment05_ft_supcon_w0.001_both_s42 [...]
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.experiment01 import (
    ExperimentConfig,
    MultimodalICDClassifier,
    _configure_huggingface_auth,
    _load_dinov3_image_encoder,
    _load_medclip_text_encoder,
    load_admission_records,
    split_records,
)
from experiments.experiment05_finetune import (
    IMAGE_DIR,
    FinetuneAdmissions,
    FinetuneCollator,
    FinetuneModel,
    FinetuneSettings,
    build_frozen_table,
    predict,
)


def dump(out_dir: Path, device: torch.device) -> None:
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    known = {f.name for f in fields(FinetuneSettings)}
    s = FinetuneSettings(**{k: v for k, v in ckpt["finetune"].items() if k in known})
    label_keys = ckpt["label_keys"]
    label_to_index = {k: i for i, k in enumerate(label_keys)}

    loaded = load_admission_records(cfg.dataset_dir, cfg.mimic_root)
    test = split_records(list(loaded.records), "test")
    image_rows = {p: i for i, p in enumerate(json.loads((IMAGE_DIR / "index.json").read_text())["paths"])}
    images_path = IMAGE_DIR / "images_256.uint8.npy"

    _configure_huggingface_auth()
    dino, _ = _load_dinov3_image_encoder(cfg.dino_checkpoint_path, cfg.dino_source_dir, device)
    medclip, tokenizer = _load_medclip_text_encoder(cfg.medclip_weights_dir, device)
    studies = sorted({st.study_id: st for r in test for st in r.studies}.values(), key=lambda st: st.study_id)
    texts = [Path(st.report_path).read_text(encoding="utf-8", errors="replace") for st in studies]
    tokens = {st.study_id: ids for st, ids in zip(studies, tokenizer(texts, truncation=True, max_length=512)["input_ids"])}

    heads = MultimodalICDClassifier(384, 512, cfg.embedding_dim, len(label_keys), cfg.dropout)
    heads.load_state_dict(ckpt["heads_state_dict"])
    model = FinetuneModel(dino, medclip, heads, s).to(device)
    result = model.load_state_dict(ckpt["encoder_trainable_state"], strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys[:5])
    for kind, trainable in (("image", s.ft_image), ("text", s.ft_text)):
        if not trainable:
            index, table = build_frozen_table(model, kind, studies, image_rows, tokens, images_path, device)
            model.frozen_tables[kind] = (index, table.to(device))

    dataset = FinetuneAdmissions(test, label_to_index, image_rows, tokens, s, training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=FinetuneCollator(images_path, tokens, s.ft_image, s.ft_text))
    _, ids, targets, probs, _ = predict(model, loader, cfg, device)
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        hadm_ids=np.asarray(ids), targets=targets.astype(np.uint8),
        **{m: p.astype(np.float16) for m, p in probs.items()},
    )
    print(f"{out_dir.name}: {len(ids)} admissions, best epoch {ckpt['epoch']}", flush=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arg in sys.argv[1:]:
        dump(Path(arg).resolve(), device)
