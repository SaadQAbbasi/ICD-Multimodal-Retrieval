"""Experiment 01: multimodal ICD classification without metric learning.

The dataset unit is one MIMIC-CXR study. ICD JSONL rows are grouped by study,
the final node of each ICD hierarchy becomes a multilabel target, and the
official MIMIC split is retained. Frozen DINOv3 and MedCLIP features are cached
once; trainable projection, fusion, and classification heads are optimized with
binary cross-entropy for image, text, and fused representations.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SPLIT_ALIASES = {"train": "train", "val": "validate", "validate": "validate", "test": "test"}


@dataclass(frozen=True)
class ExperimentConfig:
    task2_root: str
    mimic_root: str = "/datasets/MIMIC-CXR"
    metadata_path: str = "/datasets/MIMIC-CXR/mimicxr_parsed_ds.jsonl"
    dino_checkpoint_path: str = (
        "pretrained/dinov3-vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    dino_source_dir: str = "vendor/dinov3"
    medclip_weights_dir: str = "pretrained"
    cache_path: str = "cache/experiment01_backbone_embeddings.pt"
    output_dir: str = "outputs/experiment01"
    embedding_dim: int = 512
    min_label_frequency: int = 5
    max_images_per_study: int | None = None
    batch_size: int = 16
    extraction_batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    image_loss_weight: float = 0.25
    text_loss_weight: float = 0.25
    fused_loss_weight: float = 0.50
    pos_weight_cap: float = 50.0
    threshold: float = 0.5
    top_k: int = 5
    seed: int = 42
    use_amp: bool = True
    wandb_project: str = "mimic-icd-task2"
    wandb_entity: str | None = None
    wandb_run_name: str = "experiment01-classification"
    wandb_mode: str = "online"

    def resolve(self) -> "ExperimentConfig":
        root = Path(self.task2_root).resolve()
        values = asdict(self)
        values["task2_root"] = str(root)
        for field in (
            "dino_checkpoint_path",
            "dino_source_dir",
            "medclip_weights_dir",
            "cache_path",
            "output_dir",
        ):
            path = Path(values[field])
            values[field] = str(path if path.is_absolute() else root / path)
        return ExperimentConfig(**values)


@dataclass(frozen=True)
class StudyRecord:
    subject_id: int
    study_id: int
    split: str
    report_path: str
    image_paths: tuple[str, ...]
    icd_codes: tuple[str, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _leaf_code(hierarchy: str) -> str | None:
    nodes = [node.strip() for node in hierarchy.split("|") if node.strip() and node != "None"]
    return nodes[-1] if nodes else None


def load_study_records(metadata_path: str | Path, mimic_root: str | Path) -> list[StudyRecord]:
    """Group ICD rows into one multimodal, multilabel record per study."""
    mimic_root = Path(mimic_root)
    grouped: dict[int, dict[str, Any]] = {}
    subject_splits: dict[int, str] = {}

    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = SPLIT_ALIASES[row["split"]]
            subject_id = int(row["subject_id"])
            study_id = int(row["study_id"])

            previous_split = subject_splits.setdefault(subject_id, split)
            if previous_split != split:
                raise ValueError(f"Patient {subject_id} occurs in both {previous_split} and {split}")

            record = grouped.setdefault(
                study_id,
                {
                    "subject_id": subject_id,
                    "split": split,
                    "reportid": row["reportid"],
                    "codes": set(),
                },
            )
            if record["subject_id"] != subject_id or record["split"] != split:
                raise ValueError(f"Inconsistent metadata for study {study_id}")
            code = _leaf_code(row["icd_hierarchy"])
            if code:
                record["codes"].add(code)

    records: list[StudyRecord] = []
    for study_id, item in grouped.items():
        patient_bucket, patient_id, study_name = item["reportid"].split(".")
        patient_dir = mimic_root / "files" / patient_bucket / patient_id
        report_path = patient_dir / f"{study_name}.txt"
        image_paths = tuple(str(path) for path in sorted((patient_dir / study_name).glob("*.jpg")))
        if report_path.is_file() and image_paths and item["codes"]:
            records.append(
                StudyRecord(
                    subject_id=item["subject_id"],
                    study_id=study_id,
                    split=item["split"],
                    report_path=str(report_path),
                    image_paths=image_paths,
                    icd_codes=tuple(sorted(item["codes"])),
                )
            )

    records.sort(key=lambda record: (record.split, record.study_id))
    return records


def split_records(records: Sequence[StudyRecord], split: str) -> list[StudyRecord]:
    canonical = SPLIT_ALIASES[split]
    return [record for record in records if record.split == canonical]


def build_label_vocabulary(
    train_records: Sequence[StudyRecord], min_frequency: int = 1
) -> tuple[list[str], dict[str, int], Counter[str]]:
    """Build the label space from training studies only."""
    frequency = Counter(code for record in train_records for code in set(record.icd_codes))
    codes = sorted(code for code, count in frequency.items() if count >= min_frequency)
    if not codes:
        raise ValueError(f"No ICD codes occur at least {min_frequency} times in the training split")
    return codes, {code: index for index, code in enumerate(codes)}, frequency


def summarize_records(records: Sequence[StudyRecord], label_to_index: dict[str, int] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("train", "validate", "test"):
        subset = split_records(records, split)
        retained = subset
        if label_to_index is not None:
            retained = [r for r in subset if any(code in label_to_index for code in r.icd_codes)]
        summary[split] = {
            "studies": len(subset),
            "patients": len({record.subject_id for record in subset}),
            "studies_with_in_vocab_label": len(retained),
            "images": sum(len(record.image_paths) for record in subset),
        }
    return summary


class RawStudyDataset(Dataset):
    def __init__(self, records: Sequence[StudyRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> StudyRecord:
        return self.records[index]


def _read_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class BackboneExtractionCollator:
    def __init__(self, image_processor: Any, tokenizer: Any):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, records: Sequence[StudyRecord]) -> dict[str, Any]:
        images: list[Image.Image] = []
        image_counts: list[int] = []
        reports: list[str] = []
        for record in records:
            study_images = [_read_rgb(path) for path in record.image_paths]
            images.extend(study_images)
            image_counts.append(len(study_images))
            reports.append(Path(record.report_path).read_text(encoding="utf-8", errors="replace"))

        image_inputs = self.image_processor(images=images, return_tensors="pt")
        text_inputs = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "records": records,
            "image_counts": image_counts,
            "pixel_values": image_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }


class DINOv3ImageProcessor:
    """Apply Meta's published LVD-1689M evaluation transform."""

    def __init__(self, resize_size: int = 256):
        from torchvision.transforms import v2

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((resize_size, resize_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(
        self, images: Sequence[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("DINOv3ImageProcessor only supports return_tensors='pt'")
        return {"pixel_values": torch.stack([self.transform(image) for image in images])}


class MedCLIPTextEncoder(nn.Module):
    """MedCLIP's published text architecture, isolated from its legacy package."""

    def __init__(self, bert_model: nn.Module):
        super().__init__()
        self.model = bert_model
        self.projection_head = nn.Linear(768, 512, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(
            [output.hidden_states[1], output.hidden_states[2], output.hidden_states[-1]]
        )
        pooled = hidden.permute(1, 0, 2, 3).mean(dim=2).mean(dim=1)
        return self.projection_head(pooled)


def _load_medclip_text_encoder(weights_dir: str | Path, device: torch.device) -> tuple[nn.Module, Any]:
    from transformers import AutoModel, AutoTokenizer

    weights_path = Path(weights_dir) / "pytorch_model.bin"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"MedCLIP checkpoint not found at {weights_path}. Run task2/setup_experiment01.sh first."
        )

    bert_id = "emilyalsentzer/Bio_ClinicalBERT"
    bert_model = AutoModel.from_pretrained(bert_id, output_hidden_states=True)
    text_encoder = MedCLIPTextEncoder(bert_model)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    text_state = {
        key.removeprefix("text_model."): value
        for key, value in checkpoint.items()
        if key.startswith("text_model.")
    }
    if not text_state:
        raise ValueError(f"No text_model weights found in {weights_path}")
    missing, unexpected = text_encoder.load_state_dict(text_state, strict=False)
    unexpected = [key for key in unexpected if key != "model.embeddings.position_ids"]
    if missing or unexpected:
        raise ValueError(f"MedCLIP text checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    text_encoder = text_encoder.to(device).eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(bert_id)
    return text_encoder, tokenizer


def _load_dinov3_image_encoder(
    checkpoint_path: str | Path,
    source_dir: str | Path,
    device: torch.device,
) -> tuple[nn.Module, DINOv3ImageProcessor]:
    """Load the official Meta ViT-S/16 architecture and a local raw state dict."""
    checkpoint_path = Path(checkpoint_path)
    source_dir = Path(source_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found at {checkpoint_path}. "
            "Run task2/setup_experiment01.sh first."
        )
    if not (source_dir / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(
            f"Official DINOv3 source not found at {source_dir}. "
            "Run task2/setup_experiment01.sh first."
        )

    source_string = str(source_dir.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    from dinov3.hub.backbones import dinov3_vits16

    image_encoder = dinov3_vits16(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    image_encoder.load_state_dict(state_dict, strict=True)
    image_encoder = image_encoder.to(device).eval()
    for parameter in image_encoder.parameters():
        parameter.requires_grad_(False)
    return image_encoder, DINOv3ImageProcessor()


def _configure_huggingface_auth() -> None:
    """Reuse the cluster credential without copying it into the Task 2 cache."""
    if os.environ.get("HF_TOKEN"):
        return
    token_path = Path(os.environ.get("HF_TOKEN_PATH", "/gringotts/hf_home/token"))
    if token_path.is_file():
        os.environ["HF_TOKEN"] = token_path.read_text(encoding="utf-8").strip()


@torch.inference_mode()
def build_backbone_cache(
    records: Sequence[StudyRecord],
    config: ExperimentConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Create reusable, frozen DINOv3 and MedCLIP embeddings for every study."""
    config = config.resolve()
    cache_path = Path(config.cache_path)
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    _configure_huggingface_auth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder, image_processor = _load_dinov3_image_encoder(
        config.dino_checkpoint_path, config.dino_source_dir, device
    )
    text_encoder, tokenizer = _load_medclip_text_encoder(config.medclip_weights_dir, device)

    loader = DataLoader(
        RawStudyDataset(records),
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=BackboneExtractionCollator(image_processor, tokenizer),
        pin_memory=torch.cuda.is_available(),
    )

    cached_studies: dict[int, dict[str, torch.Tensor]] = {}
    image_dim = text_dim = None
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = torch.float16

    for batch in tqdm(loader, desc="Caching DINOv3 + MedCLIP embeddings"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            image_features = F.normalize(image_encoder(pixel_values).float(), dim=-1)
            text_features = F.normalize(text_encoder(input_ids, attention_mask).float(), dim=-1)

        image_dim = image_features.shape[-1]
        text_dim = text_features.shape[-1]
        offset = 0
        for record, count, text_feature in zip(batch["records"], batch["image_counts"], text_features):
            cached_studies[record.study_id] = {
                "image_features": image_features[offset : offset + count].cpu().half(),
                "text_features": text_feature.cpu().half(),
            }
            offset += count

    payload = {
        "metadata": {
            "dino_checkpoint_path": str(Path(config.dino_checkpoint_path).resolve()),
            "dino_source_dir": str(Path(config.dino_source_dir).resolve()),
            "medclip_weights_dir": str(Path(config.medclip_weights_dir).resolve()),
            "image_dim": int(image_dim or 0),
            "text_dim": int(text_dim or 0),
            "num_studies": len(cached_studies),
        },
        "studies": cached_studies,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


class CachedICDStudyDataset(Dataset):
    def __init__(
        self,
        records: Sequence[StudyRecord],
        cache: dict[str, Any],
        label_to_index: dict[str, int],
        max_images_per_study: int | None = None,
        training: bool = False,
        seed: int = 42,
    ):
        available = cache["studies"]
        self.records = [
            record
            for record in records
            if record.study_id in available and any(code in label_to_index for code in record.icd_codes)
        ]
        self.cache = available
        self.label_to_index = label_to_index
        self.max_images_per_study = max_images_per_study
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cached = self.cache[record.study_id]
        image_features = cached["image_features"].float()
        if self.max_images_per_study and image_features.shape[0] > self.max_images_per_study:
            if self.training:
                generator = torch.Generator().manual_seed(torch.initial_seed() + index)
                selected = torch.randperm(image_features.shape[0], generator=generator)[: self.max_images_per_study]
                image_features = image_features[selected]
            else:
                image_features = image_features[: self.max_images_per_study]

        labels = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for code in record.icd_codes:
            label_index = self.label_to_index.get(code)
            if label_index is not None:
                labels[label_index] = 1.0

        return {
            "study_id": record.study_id,
            "subject_id": record.subject_id,
            "image_features": image_features,
            "text_features": cached["text_features"].float(),
            "labels": labels,
        }


def collate_cached_studies(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_images = max(item["image_features"].shape[0] for item in items)
    image_dim = items[0]["image_features"].shape[-1]
    image_features = torch.zeros(batch_size, max_images, image_dim, dtype=torch.float32)
    image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)
    for index, item in enumerate(items):
        count = item["image_features"].shape[0]
        image_features[index, :count] = item["image_features"]
        image_mask[index, :count] = True

    return {
        "study_ids": [item["study_id"] for item in items],
        "subject_ids": [item["subject_id"] for item in items],
        "image_features": image_features,
        "image_mask": image_mask,
        "text_features": torch.stack([item["text_features"] for item in items]),
        "labels": torch.stack([item["labels"] for item in items]),
    }


def create_cached_loaders(
    records: Sequence[StudyRecord],
    cache: dict[str, Any],
    label_to_index: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for requested_split, canonical_split in (("train", "train"), ("val", "validate"), ("test", "test")):
        dataset = CachedICDStudyDataset(
            split_records(records, canonical_split),
            cache,
            label_to_index,
            max_images_per_study=config.max_images_per_study,
            training=requested_split == "train",
            seed=config.seed,
        )
        loaders[requested_split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=requested_split == "train",
            num_workers=config.num_workers,
            collate_fn=collate_cached_studies,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )
    return loaders


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).to(values.dtype)
    denominator = expanded_mask.sum(dim=dim).clamp_min(1.0)
    return (values * expanded_mask).sum(dim=dim) / denominator


class MultimodalICDClassifier(nn.Module):
    """Shared ICD classifier over image, text, and image-text embeddings."""

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        embedding_dim: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_projection = ProjectionHead(image_dim, embedding_dim, dropout)
        self.text_projection = ProjectionHead(text_dim, embedding_dim, dropout)
        self.fusion_projection = ProjectionHead(2 * embedding_dim, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(
        self,
        image_features: torch.Tensor,
        image_mask: torch.Tensor,
        text_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        expanded_text = text_embeddings.unsqueeze(1).expand(-1, image_embeddings.shape[1], -1)
        fused_embeddings = self.fusion_projection(torch.cat([image_embeddings, expanded_text], dim=-1))

        image_logits = self.classifier(image_embeddings)
        text_logits = self.classifier(text_embeddings)
        fused_logits = self.classifier(fused_embeddings)

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "fused_embeddings": fused_embeddings,
            "image_logits": image_logits,
            "text_logits": text_logits,
            "fused_logits": fused_logits,
            "image_study_logits": masked_mean(image_logits, image_mask, dim=1),
            "fused_study_logits": masked_mean(fused_logits, image_mask, dim=1),
            "image_study_embeddings": F.normalize(masked_mean(image_embeddings, image_mask, dim=1), dim=-1),
            "fused_study_embeddings": F.normalize(masked_mean(fused_embeddings, image_mask, dim=1), dim=-1),
        }


def compute_pos_weight(dataset: CachedICDStudyDataset, cap: float = 50.0) -> torch.Tensor:
    positives = torch.zeros(len(dataset.label_to_index), dtype=torch.float64)
    for record in dataset.records:
        for code in set(record.icd_codes):
            index = dataset.label_to_index.get(code)
            if index is not None:
                positives[index] += 1
    negatives = len(dataset) - positives
    return (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=cap).float()


def classification_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    image_mask: torch.Tensor,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image_targets = labels.unsqueeze(1).expand_as(outputs["image_logits"])
    image_element_loss = F.binary_cross_entropy_with_logits(
        outputs["image_logits"], image_targets, pos_weight=pos_weight, reduction="none"
    ).mean(dim=-1)
    image_loss = ((image_element_loss * image_mask).sum(dim=1) / image_mask.sum(dim=1).clamp_min(1)).mean()

    text_loss = F.binary_cross_entropy_with_logits(
        outputs["text_logits"], labels, pos_weight=pos_weight
    )
    fused_element_loss = F.binary_cross_entropy_with_logits(
        outputs["fused_logits"], image_targets, pos_weight=pos_weight, reduction="none"
    ).mean(dim=-1)
    fused_loss = ((fused_element_loss * image_mask).sum(dim=1) / image_mask.sum(dim=1).clamp_min(1)).mean()

    total = (
        config.image_loss_weight * image_loss
        + config.text_loss_weight * text_loss
        + config.fused_loss_weight * fused_loss
    )
    return total, {"image": image_loss, "text": text_loss, "fused": fused_loss}


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: MultimodalICDClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = Counter()
    scaler = torch.amp.GradScaler(device.type, enabled=config.use_amp and device.type == "cuda")

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            outputs = model(batch["image_features"], batch["image_mask"], batch["text_features"])
            loss, parts = classification_loss(outputs, batch["labels"], batch["image_mask"], pos_weight, config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["labels"].shape[0]
        totals["samples"] += batch_size
        totals["loss"] += loss.item() * batch_size
        for name, value in parts.items():
            totals[f"{name}_loss"] += value.item() * batch_size

    samples = max(totals.pop("samples"), 1)
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def predict(
    model: MultimodalICDClassifier,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: dict[str, list[torch.Tensor]] = {"image": [], "text": [], "fused": []}
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = _move_batch(batch, device)
        outputs = model(batch["image_features"], batch["image_mask"], batch["text_features"])
        loss, _ = classification_loss(outputs, batch["labels"], batch["image_mask"], pos_weight, config)
        batch_size = batch["labels"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        labels.append(batch["labels"].cpu())
        logits["image"].append(outputs["image_study_logits"].cpu())
        logits["text"].append(outputs["text_logits"].cpu())
        logits["fused"].append(outputs["fused_study_logits"].cpu())

    targets = torch.cat(labels).numpy()
    probabilities = {
        name: torch.sigmoid(torch.cat(parts)).numpy() for name, parts in logits.items()
    }
    return total_loss / max(total_samples, 1), targets, probabilities


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    metrics = {
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_ap": float(average_precision_score(targets, probabilities, average="micro")),
    }

    valid = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < targets.shape[0])
    metrics["evaluated_label_fraction"] = float(valid.mean())
    if valid.any():
        metrics["macro_ap_valid"] = float(
            average_precision_score(targets[:, valid], probabilities[:, valid], average="macro")
        )
        metrics["macro_auroc_valid"] = float(
            roc_auc_score(targets[:, valid], probabilities[:, valid], average="macro")
        )
    else:
        metrics["macro_ap_valid"] = math.nan
        metrics["macro_auroc_valid"] = math.nan

    k = min(top_k, probabilities.shape[1])
    top_indices = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(targets, top_indices, axis=1).sum(axis=1)
    metrics[f"precision_at_{k}"] = float(np.mean(hits / k))
    metrics[f"recall_at_{k}"] = float(np.mean(hits / np.maximum(targets.sum(axis=1), 1)))
    return metrics


def evaluate_modalities(
    model: MultimodalICDClassifier,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    loss, targets, probabilities = predict(model, loader, pos_weight, config, device)
    return {
        "loss": loss,
        "modalities": {
            name: multilabel_metrics(targets, values, config.threshold, config.top_k)
            for name, values in probabilities.items()
        },
    }


def _flatten_wandb_metrics(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):
            flattened.update(_flatten_wandb_metrics(key, value))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flattened[key] = float(value)
    return flattened


def _initialize_wandb(config: ExperimentConfig) -> Any | None:
    if config.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. "
            "Install wandb or use --wandb-mode disabled."
        ) from error

    wandb_dir = Path(config.task2_root) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        dir=str(config.task2_root),
        config=asdict(config),
        tags=["experiment01", "classification-only", "dinov3", "medclip"],
        job_type="train",
    )


def save_checkpoint(
    path: str | Path,
    model: MultimodalICDClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ExperimentConfig,
    label_codes: Sequence[str],
    validation: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config.resolve()),
            "label_codes": list(label_codes),
            "validation": validation,
        },
        path,
    )


def run_experiment(
    config: ExperimentConfig,
    build_cache: bool = False,
    force_cache: bool = False,
) -> dict[str, Any]:
    config = config.resolve()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_study_records(config.metadata_path, config.mimic_root)
    train_records = split_records(records, "train")
    label_codes, label_to_index, frequency = build_label_vocabulary(
        train_records, config.min_label_frequency
    )

    cache_path = Path(config.cache_path)
    if build_cache or force_cache:
        cache = build_backbone_cache(records, config, force=force_cache)
    elif cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(
            f"Embedding cache not found at {cache_path}. Re-run with --build-cache."
        )

    loaders = create_cached_loaders(records, cache, label_to_index, config)
    if not all(len(loader.dataset) for loader in loaders.values()):
        raise ValueError("At least one split is empty after cache and label filtering")

    image_dim = int(cache["metadata"]["image_dim"])
    text_dim = int(cache["metadata"]["text_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalICDClassifier(
        image_dim=image_dim,
        text_dim=text_dim,
        embedding_dim=config.embedding_dim,
        num_labels=len(label_codes),
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    pos_weight = compute_pos_weight(loaders["train"].dataset, config.pos_weight_cap).to(device)

    manifest = {
        "config": asdict(config),
        "label_count": len(label_codes),
        "label_frequency": dict(frequency),
        "split_summary": summarize_records(records, label_to_index),
        "cache_metadata": cache["metadata"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(label_codes, indent=2), encoding="utf-8")

    wandb_run = _initialize_wandb(config)
    if wandb_run is not None:
        initial_metrics = {
            "data/label_count": len(label_codes),
            "cache/image_dim": image_dim,
            "cache/text_dim": text_dim,
            "cache/num_studies": int(cache["metadata"]["num_studies"]),
            "storage/cache_size_mib": cache_path.stat().st_size / (1024**2),
            "storage/free_gib": shutil.disk_usage(cache_path.parent).free / (1024**3),
        }
        initial_metrics.update(_flatten_wandb_metrics("data/splits", manifest["split_summary"]))
        wandb_run.log(initial_metrics, step=0)

    try:
        history: list[dict[str, Any]] = []
        best_score = -math.inf
        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, pos_weight, config, device
            )
            validation = evaluate_modalities(model, loaders["val"], pos_weight, config, device)
            epoch_result = {"epoch": epoch, "train": train_metrics, "validation": validation}
            history.append(epoch_result)
            print(json.dumps(epoch_result, indent=2))

            score = validation["modalities"]["fused"]["micro_ap"]
            if wandb_run is not None:
                epoch_metrics = {"epoch": epoch}
                epoch_metrics.update(_flatten_wandb_metrics("train", train_metrics))
                epoch_metrics.update(_flatten_wandb_metrics("validation", validation))
                wandb_run.log(epoch_metrics, step=epoch)

            if score > best_score:
                best_score = score
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    label_codes,
                    validation,
                )

        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_results = evaluate_modalities(model, loaders["test"], pos_weight, config, device)
        results = {"history": history, "best_epoch": checkpoint["epoch"], "test": test_results}
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        if wandb_run is not None:
            wandb_run.log(
                _flatten_wandb_metrics("test", test_results), step=config.epochs + 1
            )
            wandb_run.summary["best_epoch"] = checkpoint["epoch"]
            wandb_run.summary["best_validation_fused_micro_ap"] = best_score
            for name, value in _flatten_wandb_metrics("test", test_results).items():
                wandb_run.summary[name] = value
        return results
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mimic-root", type=str, default="/datasets/MIMIC-CXR")
    parser.add_argument("--metadata-path", type=str, default="/datasets/MIMIC-CXR/mimicxr_parsed_ds.jsonl")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-label-frequency", type=int, default=5)
    parser.add_argument("--max-images-per-study", type=int, default=None)
    parser.add_argument(
        "--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "mimic-icd-task2")
    )
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.getenv("WANDB_RUN_NAME", "experiment01-classification"),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = ExperimentConfig(
        task2_root=args.task2_root,
        mimic_root=args.mimic_root,
        metadata_path=args.metadata_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_label_frequency=args.min_label_frequency,
        max_images_per_study=args.max_images_per_study,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    run_experiment(configuration, build_cache=args.build_cache, force_cache=args.force_cache)
