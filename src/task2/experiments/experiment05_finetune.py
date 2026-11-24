"""Round 5: partial encoder fine-tuning (BCE + Jaccard SupCon lambda=0.001).

Trains end to end from the resized images (cache/images_256, see
experiments/resize_images.py) and the raw report text instead of the frozen
embedding cache. Everything after the encoders is identical to Exp3:

  image study vector = normalize(mean_i normalize(DINOv3 CLS_i))       (384)
  text  study vector = normalize(MedCLIP(masked mean of hs 1, 2, 12))  (512)
  -> MultimodalICDClassifier (projection heads, study mean, fusion, shared classifier)
  loss = 0.25 BCE_img + 0.25 BCE_txt + 0.5 BCE_fused + 0.001 * Jaccard SupCon

Unfrozen (per arm):
  image: DINOv3 blocks[-FT_BLOCKS:] + final norm
  text : BERT encoder.layer[-FT_BLOCKS:] + MedCLIP 768->512 projection
A frozen modality's study vectors are computed once at start-up (and checked
against the embedding cache). Both encoders always stay in eval mode (DINOv3
applies RoPE coordinate augmentation in train mode; no dropout in encoders),
so step-0 features equal the cache; gradients still flow through the unfrozen
layers. Heads (warm-started from the Round 4 SupCon checkpoint of the same
seed) train in train mode (dropout 0.1).

Env: FT_IMAGE=0/1 FT_TEXT=0/1 SEED=42 EPOCHS=20 FT_BLOCKS=2 ENCODER_LR=1e-5
     HEAD_LR=1e-4 SMOKE=0/1 (tiny subsets, 2 epochs, memory probe)

    FT_IMAGE=1 FT_TEXT=1 python -m experiments.experiment05_finetune
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.experiment01 import (
    ExperimentConfig,
    MultimodalICDClassifier,
    _cm_hierarchy_metrics,
    _configure_huggingface_auth,
    _flatten_wandb_metrics,
    _load_dinov3_image_encoder,
    _load_medclip_text_encoder,
    build_label_vocabulary,
    classification_loss,
    jaccard_supcon_loss,
    load_admission_records,
    multilabel_metrics,
    set_seed,
    split_records,
)
from experiments.retrieval_eval import score_neighbours

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "cache/images_256"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FinetuneSettings:
    seed: int = 42
    ft_image: bool = True
    ft_text: bool = True
    ft_blocks: int = 2
    epochs: int = 20
    batch_size: int = 16
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    lr_min_ratio: float = 0.01
    early_stopping_patience: int = 6
    val_full_every: int = 5
    max_studies_train: int = 16  # per admission and modality (random subset), eval uses all
    max_images_train: int = 40  # image budget per admission during training
    encode_chunk: int = 128
    num_workers: int = 4
    smoke: bool = False
    warm_start_dir: str = ""
    output_dir: str = ""
    wandb_run_name: str = ""

    @property
    def arm(self) -> str:
        return {(0, 0): "frozen", (1, 0): "img", (0, 1): "txt", (1, 1): "both"}[
            (int(self.ft_image), int(self.ft_text))
        ]


def settings_from_env() -> FinetuneSettings:
    env = os.environ
    s = FinetuneSettings(
        seed=int(env.get("SEED", 42)),
        ft_image=env.get("FT_IMAGE", "1") == "1",
        ft_text=env.get("FT_TEXT", "1") == "1",
        ft_blocks=int(env.get("FT_BLOCKS", 2)),
        epochs=int(env.get("EPOCHS", 20)),
        encoder_lr=float(env.get("ENCODER_LR", 1e-5)),
        head_lr=float(env.get("HEAD_LR", 1e-4)),
        early_stopping_patience=int(env.get("EARLY_STOP_PATIENCE", 6)),
        num_workers=int(env.get("SLURM_CPUS_PER_TASK", 4)),
        smoke=env.get("SMOKE", "0") == "1",
    )
    if s.smoke:
        s = replace(s, epochs=int(env.get("EPOCHS", 2)), val_full_every=2)
    tag = f"supcon_w0.001_{s.arm}"
    if s.ft_blocks != 2:
        tag += f"_b{s.ft_blocks}"
    tag += f"_s{s.seed}" + ("_smoke" if s.smoke else "")
    return replace(
        s,
        warm_start_dir=str(ROOT / f"outputs/experiment03_supcon_w0.001_e120_cos_s{s.seed}"),
        output_dir=str(ROOT / f"outputs/experiment05_ft_{tag}"),
        wandb_run_name=env.get("WANDB_RUN_NAME", f"experiment05-ft-{tag.replace('_', '-')}"),
    )


# --------------------------------------------------------------------------- data
class FinetuneAdmissions(Dataset):
    """Per admission: image studies (rows into the resized array), report token ids."""

    def __init__(self, records, label_to_index, image_rows, tokens, s: FinetuneSettings, training):
        self.records = [r for r in records if any(k in label_to_index for k in r.target_keys)]
        self.label_to_index = label_to_index
        self.image_rows = image_rows
        self.tokens = tokens
        self.s = s
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_studies = [st for st in record.studies if st.image_paths]
        text_studies = list(record.studies)
        if self.training:
            random.shuffle(image_studies)
            chosen, budget = [], 0
            for st in image_studies:
                if chosen and (len(chosen) >= self.s.max_studies_train
                               or budget + len(st.image_paths) > self.s.max_images_train):
                    continue
                chosen.append(st)
                budget += len(st.image_paths)
            image_studies = chosen
            random.shuffle(text_studies)
            text_studies = text_studies[: self.s.max_studies_train]
        labels = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for key in record.target_keys:
            if key in self.label_to_index:
                labels[self.label_to_index[key]] = 1.0
        return {
            "hadm_id": record.hadm_id,
            "image_study_ids": [st.study_id for st in image_studies],
            "image_rows": [[self.image_rows[p] for p in st.image_paths] for st in image_studies],
            "text_study_ids": [st.study_id for st in text_studies],
            "labels": labels,
            "complete_images": record.all_referenced_images_available,
        }


class FinetuneCollator:
    def __init__(self, images_path, tokens, load_pixels, load_text):
        self.images_path = images_path
        self.images = None  # opened lazily per worker
        self.tokens = tokens
        self.load_pixels = load_pixels
        self.load_text = load_text

    def __call__(self, items):
        batch = {
            "hadm_ids": [it["hadm_id"] for it in items],
            "complete_images": [it["complete_images"] for it in items],
            "labels": torch.stack([it["labels"] for it in items]),
        }
        for kind in ("image", "text"):
            adm, pos, ids = [], [], []
            for a, it in enumerate(items):
                for p, sid in enumerate(it[f"{kind}_study_ids"]):
                    adm.append(a)
                    pos.append(p)
                    ids.append(sid)
            batch[f"{kind}_study_adm"] = torch.tensor(adm)
            batch[f"{kind}_study_pos"] = torch.tensor(pos)
            batch[f"{kind}_study_ids"] = torch.tensor(ids)
        if self.load_pixels:
            if self.images is None:
                self.images = np.load(self.images_path, mmap_mode="r")
            rows, image_study, flat = [], [], 0
            for it in items:
                for study_rows in it["image_rows"]:
                    rows.extend(study_rows)
                    image_study.extend([flat] * len(study_rows))
                    flat += 1
            order = np.argsort(rows)  # sorted reads from the memmap
            pixels = np.empty((len(rows), *self.images.shape[1:]), dtype=np.uint8)
            pixels[order] = self.images[np.asarray(rows)[order]]
            batch["pixels"] = torch.from_numpy(pixels)
            batch["image_of_study"] = torch.tensor(image_study)
        if self.load_text:
            seqs = [self.tokens[sid] for sid in batch["text_study_ids"].tolist()]
            length = max(len(t) for t in seqs)
            ids = torch.zeros(len(seqs), length, dtype=torch.long)
            mask = torch.zeros(len(seqs), length, dtype=torch.long)
            for i, t in enumerate(seqs):
                ids[i, : len(t)] = torch.tensor(t)
                mask[i, : len(t)] = 1
            batch["input_ids"], batch["attention_mask"] = ids, mask
        return batch


# --------------------------------------------------------------------------- model
class FinetuneModel(torch.nn.Module):
    def __init__(self, dino, medclip, heads, s: FinetuneSettings):
        super().__init__()
        self.dino, self.medclip, self.heads, self.s = dino, medclip, heads, s
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.frozen_tables: dict[str, tuple[dict[int, int], torch.Tensor]] = {}
        for p in list(dino.parameters()) + list(medclip.parameters()):
            p.requires_grad_(False)
        k = s.ft_blocks
        if s.ft_image:
            for m in list(dino.blocks[-k:]) + [dino.norm]:
                for p in m.parameters():
                    p.requires_grad_(True)
        if s.ft_text:
            for m in list(medclip.model.encoder.layer[-k:]) + [medclip.projection_head]:
                for p in m.parameters():
                    p.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        self.medclip.eval()
        return self

    def encoder_parameters(self, which):
        module = self.dino if which == "image" else self.medclip
        return [p for p in module.parameters() if p.requires_grad]

    def encode_images(self, pixels_u8: torch.Tensor) -> torch.Tensor:
        out = []
        for chunk in pixels_u8.split(self.s.encode_chunk):
            x = chunk.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
            x = (x - self.mean) / self.std
            out.append(F.normalize(self.dino(x).float(), dim=-1))
        return torch.cat(out)

    def encode_reports(self, input_ids, attention_mask) -> torch.Tensor:
        out = []
        for ids, mask in zip(input_ids.split(self.s.encode_chunk), attention_mask.split(self.s.encode_chunk)):
            ids, mask = ids[:, : int(mask.sum(1).max())], mask[:, : int(mask.sum(1).max())]
            out.append(F.normalize(self.medclip(ids, mask).float(), dim=-1))
        return torch.cat(out)

    @staticmethod
    def study_mean(image_features, image_of_study, num_studies):
        summed = torch.zeros(num_studies, image_features.shape[1], device=image_features.device,
                             dtype=image_features.dtype).index_add_(0, image_of_study, image_features)
        return F.normalize(summed, dim=-1)  # normalize(sum) == normalize(mean)

    def study_vectors(self, batch, kind):
        if kind in self.frozen_tables:
            index, table = self.frozen_tables[kind]
            rows = torch.tensor([index[i] for i in batch[f"{kind}_study_ids"].tolist()], device=table.device)
            return table[rows]
        if kind == "image":
            feats = self.encode_images(batch["pixels"])
            return self.study_mean(feats, batch["image_of_study"], len(batch["image_study_ids"]))
        return self.encode_reports(batch["input_ids"], batch["attention_mask"])

    @staticmethod
    def pad(vectors, adm, pos, batch_size):
        maximum = int(pos.max()) + 1
        padded = vectors.new_zeros(batch_size, maximum, vectors.shape[1])
        mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=vectors.device)
        padded[adm, pos] = vectors
        mask[adm, pos] = True
        return padded, mask

    def forward(self, batch):
        b = batch["labels"].shape[0]
        img, img_mask = self.pad(self.study_vectors(batch, "image"), batch["image_study_adm"], batch["image_study_pos"], b)
        txt, txt_mask = self.pad(self.study_vectors(batch, "text"), batch["text_study_adm"], batch["text_study_pos"], b)
        return self.heads(img, img_mask, txt, txt_mask)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# --------------------------------------------------------------------------- helpers
@torch.inference_mode()
def build_frozen_table(model, kind, studies, image_rows, tokens, images_path, device, chunk=64):
    """Study vectors for every study with the (frozen) encoder."""
    studies = [st for st in studies if st.image_paths] if kind == "image" else list(studies)
    collate = FinetuneCollator(images_path, tokens, kind == "image", kind == "text")
    index, vectors = {}, []
    for start in range(0, len(studies), chunk):
        part = studies[start: start + chunk]
        item = {"hadm_id": 0, "labels": torch.zeros(1), "complete_images": True,
                "image_study_ids": [], "image_rows": [], "text_study_ids": []}
        item[f"{kind}_study_ids"] = [st.study_id for st in part]
        if kind == "image":
            item["image_rows"] = [[image_rows[p] for p in st.image_paths] for st in part]
        batch = move(collate([item]), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            vectors.append(model.study_vectors(batch, kind).float())
        for st in part:
            index[st.study_id] = len(index)
    return index, torch.cat(vectors)


def compare_with_cache(kind, index, table, cache):
    key = f"{kind}_feature"
    ids = [sid for sid in index if key in cache["studies"].get(sid, {})]
    ref = torch.stack([cache["studies"][sid][key].float() for sid in ids])
    ours = table[[index[sid] for sid in ids]].cpu()
    cos = (F.normalize(ref, dim=-1) * F.normalize(ours, dim=-1)).sum(-1)
    return {"studies": len(ids), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
            "max_abs_diff": float((ref - ours).abs().max())}


def lr_lambda_factory(s: FinetuneSettings, steps_per_epoch: int):
    total = s.epochs * steps_per_epoch
    warmup = max(1, int(s.warmup_epochs * steps_per_epoch))

    def factor(step):
        if step < warmup:
            return 0.01 + 0.99 * step / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return s.lr_min_ratio + (1 - s.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def grad_norm(params):
    grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(grads).norm()) if grads else 0.0


@torch.inference_mode()
def predict(model, loader, cfg, device, want_embeddings=False):
    model.eval()
    ids, labels, logits, total, n = [], [], {m: [] for m in ("image", "text", "fused")}, 0.0, 0
    emb = {m: [] for m in ("image", "text", "fused")}
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch)
        out = {k: v.float() for k, v in out.items()}
        loss, _ = classification_loss(out, batch["labels"], None, cfg)
        bs = batch["labels"].shape[0]
        total += loss.item() * bs
        n += bs
        ids.extend(batch["hadm_ids"])
        labels.append(batch["labels"].cpu())
        for m in logits:
            logits[m].append(out[f"{m}_logits"].cpu())
            if want_embeddings:
                emb[m].append(F.normalize(out[f"{m}_embedding"], dim=-1).cpu())
    targets = torch.cat(labels).numpy()
    probs = {m: torch.sigmoid(torch.cat(v)).numpy() for m, v in logits.items()}
    embeddings = {m: torch.cat(v) for m, v in emb.items()} if want_embeddings else None
    return total / max(n, 1), ids, targets, probs, embeddings


def evaluate(model, loader, cfg, device, label_keys, catalog, records_by_id, full):
    loss, ids, targets, probs, _ = predict(model, loader, cfg, device)
    return {
        "loss": loss,
        "sample_count": len(ids),
        "full_metrics": full,
        "modalities": {m: multilabel_metrics(targets, p, cfg.threshold, cfg.top_k, macro=full)
                       for m, p in probs.items()},
        "cm_hierarchy": _cm_hierarchy_metrics(ids, probs, label_keys, catalog, records_by_id, cfg) if full else {},
    }


def retrieval(model, train_loader, test_loader, cfg, device, k=10):
    from sklearn.metrics import average_precision_score

    _, _, train_targets, _, train_emb = predict(model, train_loader, cfg, device, want_embeddings=True)
    _, _, test_targets, _, test_emb = predict(model, test_loader, cfg, device, want_embeddings=True)
    train_labels, test_labels = torch.from_numpy(train_targets), torch.from_numpy(test_targets)
    keep = test_labels.sum(1) > 0
    test_labels = test_labels[keep]
    res = {"k": k, "queries": int(keep.sum()), "gallery": int(train_labels.shape[0])}
    gen = torch.Generator().manual_seed(0)
    res["random"] = score_neighbours(test_labels, train_labels,
                                     torch.randint(0, train_labels.shape[0], (test_labels.shape[0], k), generator=gen))
    valid = (test_labels.sum(0) > 0).numpy()
    for m in ("image", "text", "fused"):
        top = (test_emb[m][keep] @ train_emb[m].T).topk(k, dim=1)
        metrics = score_neighbours(test_labels, train_labels, top.indices)
        votes = (torch.softmax(top.values / 0.1, dim=1).unsqueeze(-1) * train_labels[top.indices]).sum(1).numpy()
        metrics["knn_micro_ap"] = float(average_precision_score(test_labels.numpy()[:, valid].ravel(), votes[:, valid].ravel()))
        res[m] = metrics
    return res


# --------------------------------------------------------------------------- main
def run(s: FinetuneSettings) -> dict[str, Any]:
    set_seed(s.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(s.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warm_dir = Path(s.warm_start_dir)
    cfg = ExperimentConfig(**json.loads((warm_dir / "manifest.json").read_text())["config"])
    cfg = replace(cfg, output_dir=str(out_dir), seed=s.seed, wandb_run_name=s.wandb_run_name)
    print("settings", json.dumps(asdict(s)), flush=True)

    loaded = load_admission_records(cfg.dataset_dir, cfg.mimic_root)
    records = list(loaded.records)
    records_by_id = {r.hadm_id: r for r in records}
    label_keys, label_to_index, _ = build_label_vocabulary(split_records(records, "train"), cfg.min_label_frequency)
    warm = torch.load(warm_dir / "best.pt", map_location="cpu", weights_only=False)
    if list(warm["label_keys"]) != label_keys:
        raise ValueError("label vocabulary differs from the warm-start checkpoint")

    index_meta = json.loads((IMAGE_DIR / "index.json").read_text())
    image_rows = {p: i for i, p in enumerate(index_meta["paths"])}
    images_path = IMAGE_DIR / "images_256.uint8.npy"

    _configure_huggingface_auth()
    dino, _ = _load_dinov3_image_encoder(cfg.dino_checkpoint_path, cfg.dino_source_dir, device)
    medclip, tokenizer = _load_medclip_text_encoder(cfg.medclip_weights_dir, device)
    studies = sorted({st.study_id: st for r in records for st in r.studies}.values(), key=lambda st: st.study_id)
    t0 = time.time()
    texts = [Path(st.report_path).read_text(encoding="utf-8", errors="replace") for st in studies]
    enc = tokenizer(texts, truncation=True, max_length=512)["input_ids"]
    tokens = {st.study_id: ids for st, ids in zip(studies, enc)}
    print(f"tokenized {len(tokens)} reports in {time.time()-t0:.0f}s", flush=True)

    heads = MultimodalICDClassifier(384, 512, cfg.embedding_dim, len(label_keys), cfg.dropout)
    heads.load_state_dict(warm["model_state_dict"])
    model = FinetuneModel(dino, medclip, heads, s).to(device)

    splits = {name: split_records(records, name) for name in ("train", "validation", "test")}
    if s.smoke:
        rng = random.Random(0)
        splits = {k: rng.sample(v, {"train": 256, "validation": 128, "test": 128}[k]) for k, v in splits.items()}
    complete_test = [r for r in splits["test"] if r.all_referenced_images_available]

    cache_check = {}
    cache = torch.load(cfg.cache_path, map_location="cpu", weights_only=False)
    needed = sorted({st.study_id: st for v in splits.values() for r in v for st in r.studies}.values(),
                    key=lambda st: st.study_id)
    for kind, trainable in (("image", s.ft_image), ("text", s.ft_text)):
        t0 = time.time()
        sample = needed if not trainable else needed[:: max(1, len(needed) // 512)]
        index, table = build_frozen_table(model, kind, sample, image_rows, tokens, images_path, device)
        cache_check[kind] = compare_with_cache(kind, index, table, cache)
        cache_check[kind]["seconds"] = time.time() - t0
        if not trainable:
            model.frozen_tables[kind] = (index, table.to(device))
        print(f"{kind} features vs cache ({'frozen table' if not trainable else 'sample'}):",
              cache_check[kind], flush=True)
    del cache

    load_pixels, load_text = s.ft_image, s.ft_text
    collate = FinetuneCollator(images_path, tokens, load_pixels, load_text)

    def loader(recs, training):
        ds = FinetuneAdmissions(recs, label_to_index, image_rows, tokens, s, training)
        return DataLoader(ds, batch_size=s.batch_size, shuffle=training, num_workers=s.num_workers,
                          collate_fn=collate, pin_memory=device.type == "cuda",
                          persistent_workers=s.num_workers > 0, drop_last=False)

    loaders = {"train": loader(splits["train"], True), "val": loader(splits["validation"], False),
               "test": loader(splits["test"], False), "test_complete": loader(complete_test, False),
               "train_eval": loader(splits["train"], False)}
    print({k: len(v.dataset) for k, v in loaders.items()}, flush=True)

    groups = [{"params": list(heads.parameters()), "lr": s.head_lr, "name": "heads"}]
    for kind, on in (("image", s.ft_image), ("text", s.ft_text)):
        if on:
            groups.append({"params": model.encoder_parameters(kind), "lr": s.encoder_lr, "name": f"{kind}_encoder"})
    optimizer = torch.optim.AdamW(groups, weight_decay=s.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(s, len(loaders["train"])))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    trainable = [p for g in groups for p in g["params"]]
    print("trainable params:", {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}, flush=True)

    wandb_run = None
    if os.getenv("WANDB_MODE", "disabled") != "disabled":
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=s.wandb_run_name,
                               dir=str(ROOT), config={"finetune": asdict(s), "base": asdict(cfg)},
                               tags=["experiment05", "finetune", s.arm, "jaccard-supcon", "admission-level"],
                               job_type="train", mode=os.getenv("WANDB_MODE", "online"))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"finetune": asdict(s), "config": asdict(cfg), "label_count": len(label_keys),
         "cache_check": cache_check, "sizes": {k: len(v.dataset) for k, v in loaders.items()},
         "warm_start_epoch": warm["epoch"]}, indent=2))
    (out_dir / "labels.json").write_text(json.dumps(label_keys))

    def ev(name, full):
        return evaluate(model, loaders[name], cfg, device, label_keys, loaded.target_catalog, records_by_id, full)

    def save_best(epoch, validation):
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad and not n.startswith("heads.")}
        state = {n: t.detach().cpu() for n, t in model.state_dict().items() if n in trainable_names}
        torch.save({"epoch": epoch, "heads_state_dict": heads.state_dict(), "encoder_trainable_state": state,
                    "finetune": asdict(s), "config": asdict(cfg), "label_keys": label_keys,
                    "validation": validation}, out_dir / "best.pt.tmp")
        os.replace(out_dir / "best.pt.tmp", out_dir / "best.pt")

    try:
        t0 = time.time()
        epoch0 = ev("val", full=False)
        print(f"epoch 0 (warm start) val fused AP {epoch0['modalities']['fused']['micro_ap']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if wandb_run:
            wandb_run.log({"epoch": 0, **_flatten_wandb_metrics("validation", epoch0)})
        best, since, history = epoch0["modalities"]["fused"]["micro_ap"], 0, []
        save_best(0, epoch0)
        if s.smoke:
            memory_probe(model, loaders["train"].dataset, collate, cfg, device)
        step = 0
        for epoch in range(1, s.epochs + 1):
            model.train()
            totals, t_epoch = Counter(), time.time()
            for batch in loaders["train"]:
                batch = move(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(batch)
                    bce, parts = classification_loss(out, batch["labels"], None, cfg)
                    supcon = jaccard_supcon_loss([out["image_embedding"], out["text_embedding"], out["fused_embedding"]],
                                                 batch["labels"], cfg.supcon_temperature)
                    loss = bce + cfg.supcon_weight * supcon
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norms = {f"grad_norm_{g['name']}": grad_norm(g["params"]) for g in groups}
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                bs = batch["labels"].shape[0]
                totals["samples"] += bs
                vals = {"loss": loss, "bce_total_loss": bce, "supcon_loss": supcon,
                        **{f"{k}_loss": v for k, v in parts.items()}}
                for k, v in vals.items():
                    totals[k] += float(v) * bs
                for k, v in norms.items():
                    totals[k] += v * bs
                if wandb_run and step % 50 == 0:
                    wandb_run.log({"step": step, **{f"train_step/{k}": float(v) for k, v in vals.items()},
                                   **{f"train_step/{k}": v for k, v in norms.items()},
                                   **{f"train_step/lr_{g['name']}": g["lr"] for g in optimizer.param_groups}})
            n = max(totals.pop("samples"), 1)
            train_metrics = {k: v / n for k, v in totals.items()}
            train_metrics.update({f"lr_{g['name']}": g["lr"] for g in optimizer.param_groups})
            train_metrics["seconds"] = time.time() - t_epoch
            if device.type == "cuda":
                train_metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
            full = epoch % s.val_full_every == 0 or epoch == s.epochs
            t_val = time.time()
            validation = ev("val", full)
            validation["seconds"] = time.time() - t_val
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
            score = validation["modalities"]["fused"]["micro_ap"]
            print(f"epoch {epoch}: train {train_metrics['seconds']:.0f}s loss {train_metrics['loss']:.4f} "
                  f"supcon {train_metrics['supcon_loss']:.3f} | val {validation['seconds']:.0f}s fused AP {score:.4f} "
                  f"(best {max(best, score):.4f}) | mem {train_metrics.get('peak_memory_gib', 0):.1f} GiB "
                  f"| grad norms " + " ".join(f"{k[10:]}={train_metrics[k]:.3g}" for k in norms), flush=True)
            if wandb_run:
                wandb_run.log({"epoch": epoch, **_flatten_wandb_metrics("train", train_metrics),
                               **_flatten_wandb_metrics("validation", validation)})
            if score > best:
                best, since = score, 0
                save_best(epoch, validation)
            else:
                since += 1
                if s.early_stopping_patience and since >= s.early_stopping_patience:
                    print(f"early stop at epoch {epoch} (best {best:.4f})", flush=True)
                    break

        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        heads.load_state_dict(ckpt["heads_state_dict"])
        missing = model.load_state_dict(ckpt["encoder_trainable_state"], strict=False)
        if missing.unexpected_keys:
            raise ValueError(f"unexpected keys in checkpoint: {missing.unexpected_keys[:5]}")
        results = {"history": history, "epoch0_validation": epoch0, "best_epoch": ckpt["epoch"],
                   "cache_check": cache_check,
                   "test": ev("test", True), "test_complete_images": ev("test_complete", True)}
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"best epoch {ckpt['epoch']}  TEST fused micro AP {results['test']['modalities']['fused']['micro_ap']:.4f}"
              f"  P@5 {results['test']['modalities']['fused']['precision_at_5']:.4f}", flush=True)
        ret = retrieval(model, loaders["train_eval"], loaders["test"], cfg, device)
        (out_dir / "retrieval.json").write_text(json.dumps(ret, indent=2))
        print("retrieval fused", ret["fused"], flush=True)
        if wandb_run:
            wandb_run.log(_flatten_wandb_metrics("test", results["test"]))
            wandb_run.log(_flatten_wandb_metrics("retrieval", ret))
            wandb_run.summary["best_epoch"] = ckpt["epoch"]
            wandb_run.summary["best_validation_fused_micro_ap"] = best
        return results
    finally:
        if wandb_run:
            wandb_run.finish()


def memory_probe(model, dataset, collate, cfg, device):
    """Smoke only: forward/backward on the 16 train admissions with the most images."""
    if device.type != "cuda":
        return
    order = sorted(range(len(dataset)), key=lambda i: -sum(len(st.image_paths) for st in dataset.records[i].studies))
    batch = move(collate([dataset[i] for i in order[: 16]]), device)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(batch)
        bce, _ = classification_loss(out, batch["labels"], None, cfg)
        loss = bce + cfg.supcon_weight * jaccard_supcon_loss(
            [out["image_embedding"], out["text_embedding"], out["fused_embedding"]], batch["labels"], cfg.supcon_temperature)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"memory probe: {len(batch.get('pixels', []))} images, {len(batch['text_study_ids'])} reports -> "
          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, {time.time()-t0:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    run(settings_from_env())
