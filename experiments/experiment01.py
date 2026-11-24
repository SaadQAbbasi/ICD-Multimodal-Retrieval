"""Experiment 01: admission-level multimodal ICD classification.

The official Edin MIMIC-IV ICD-10 multimodal splits are grouped by hospital
admission. Each retained admission has at least one locally available CXR JPG.
Frozen DINOv3 image features and frozen MedCLIP report features are cached once,
pooled within the admission, projected into one shared space, and classified
with admission-level multilabel binary cross-entropy. No metric-learning loss
is used in this baseline.
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
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validate": "validation",
    "validation": "validation",
    "test": "test",
}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ExperimentConfig:
    task2_root: str
    # Dataset locations; override with the MIMIC_ROOT / DATASET_DIR environment variables.
    mimic_root: str = os.getenv("MIMIC_ROOT", "/datasets/MIMIC-CXR")
    dataset_dir: str = os.getenv(
        "DATASET_DIR", "/datasets/MIMIC-GEN-RAG/edin_splits/mimic-iv_icd10/multimodal"
    )
    # Backbone weights (relative = inside this repository, as placed by setup_weights.sh).
    # DINO_CHECKPOINT / DINO_SOURCE_DIR / MEDCLIP_WEIGHTS_DIR override them, e.g. with absolute
    # paths to an existing copy of the weights.
    dino_checkpoint_path: str = os.getenv(
        "DINO_CHECKPOINT", "pretrained/dinov3-vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    dino_source_dir: str = os.getenv("DINO_SOURCE_DIR", "vendor/dinov3")  # Meta's DINOv3 code
    medclip_weights_dir: str = os.getenv("MEDCLIP_WEIGHTS_DIR", "pretrained")  # folder with MedCLIP pytorch_model.bin
    # Frozen-backbone feature cache, built once by experiments/build_cache.py.
    # CACHE_PATH overrides it (configs/paths.env; slurm/quick_test.sbatch uses it for its subset cache).
    cache_path: str = os.getenv("CACHE_PATH", "cache/experiment01_admission_backbone_embeddings_textfix.pt")
    output_dir: str = "outputs/experiment01_admission"
    embedding_dim: int = 512
    min_label_frequency: int = 1
    max_image_studies_per_admission: int | None = None
    max_text_reports_per_admission: int | None = None
    batch_size: int = 16
    extraction_batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    early_stopping_patience: int = 0  # 0 = disabled; else stop after N epochs w/o val fused micro AP gain
    lr_schedule: str = "constant"  # constant | cosine | plateau; stepped once per epoch (round 4)
    lr_min_ratio: float = 0.01  # floor LR = learning_rate * lr_min_ratio (cosine eta_min / plateau min_lr)
    plateau_patience: int = 5  # plateau: epochs w/o val fused micro AP gain before LR *= plateau_factor
    plateau_factor: float = 0.5
    validation_full_metrics_every: int = 1  # full (macro AP/AUROC + hierarchy) validation every N epochs
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    image_loss_weight: float = 0.25
    text_loss_weight: float = 0.25
    fused_loss_weight: float = 0.50
    pos_weight_cap: float = 50.0
    use_positive_weights: bool = True
    threshold: float = 0.5
    top_k: int = 5
    supcon_weight: float = 0.0
    supcon_temperature: float = 0.1
    supcon_positive_mode: str = "jaccard"  # jaccard (exact-code Jaccard) | hierarchy (depth-weighted mean of per-level Jaccard, round 6)
    seed: int = 42
    use_amp: bool = True
    wandb_project: str = "mimic-icd-task2"
    wandb_entity: str | None = None
    wandb_run_name: str = "experiment01-admission-classification"
    wandb_mode: str = os.getenv("WANDB_MODE", "disabled")  # online | offline | disabled

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
        values["mimic_root"] = str(Path(values["mimic_root"]).resolve())
        values["dataset_dir"] = str(Path(values["dataset_dir"]).resolve())
        return ExperimentConfig(**values)


@dataclass(frozen=True)
class StudyResource:
    study_id: int
    report_id: str
    report_path: str
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class AdmissionRecord:
    subject_id: int
    hadm_id: int
    split: str
    studies: tuple[StudyResource, ...]
    target_keys: tuple[str, ...]
    all_referenced_images_available: bool


@dataclass(frozen=True)
class LoadedAdmissions:
    records: tuple[AdmissionRecord, ...]
    target_catalog: dict[str, dict[str, Any]]
    availability_audit: dict[str, dict[str, int]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_key(row: dict[str, Any]) -> str:
    return f"{int(row['icd_version_mapped'])}:{row['code_system_mapped']}:{row['docid']}"


def _clean_path(value: Any) -> tuple[str, ...]:
    return tuple(
        node.strip()
        for node in str(value).split("|")
        if node.strip() and node.strip() != "None"
    )


def _report_paths(report_id: str, mimic_root: Path) -> tuple[Path, Path]:
    bucket, patient, study = report_id.split(".")
    patient_dir = mimic_root / "files" / bucket / patient
    return patient_dir / f"{study}.txt", patient_dir / study


def load_admission_records(
    dataset_dir: str | Path,
    mimic_root: str | Path,
) -> LoadedAdmissions:
    """Group official split rows into admission-level multimodal records.

    Admissions with no locally available JPG are excluded. Reports from every
    referenced study are retained, including studies whose JPG directory is
    absent, because all reports are available in the current dataset mount.
    """
    dataset_dir = Path(dataset_dir)
    mimic_root = Path(mimic_root)
    grouped: dict[int, dict[str, Any]] = {}
    subject_splits: dict[int, str] = {}
    mutable_catalog: dict[str, dict[str, Any]] = {}

    for split in SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"Official split file not found: {split_path}")
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] != split:
                    raise ValueError(
                        f"Row split {row['split']!r} disagrees with file {split_path.name}"
                    )
                subject_id = int(row["subject_id"])
                hadm_id = int(row["hadm_id"])
                report_ids = tuple(row.get("reportids") or ())
                study_ids = tuple(int(value) for value in (row.get("study_ids") or ()))
                if not report_ids or len(report_ids) != len(study_ids):
                    raise ValueError(f"Invalid report/study references for admission {hadm_id}")

                previous_split = subject_splits.setdefault(subject_id, split)
                if previous_split != split:
                    raise ValueError(
                        f"Patient {subject_id} occurs in both {previous_split} and {split}"
                    )

                admission = grouped.setdefault(
                    hadm_id,
                    {
                        "subject_id": subject_id,
                        "split": split,
                        "report_ids": report_ids,
                        "study_ids": study_ids,
                        "target_keys": set(),
                    },
                )
                stable = (
                    admission["subject_id"],
                    admission["split"],
                    admission["report_ids"],
                    admission["study_ids"],
                )
                if stable != (subject_id, split, report_ids, study_ids):
                    raise ValueError(f"Inconsistent repeated rows for admission {hadm_id}")

                key = _target_key(row)
                admission["target_keys"].add(key)
                leaf_to_root = _clean_path(row["icd_sequence"])
                if not leaf_to_root:
                    raise ValueError(f"Empty ICD sequence for target {key}")
                docid_leaf_to_root = _clean_path(row["docid"].replace("-", "|"))
                candidate = {
                    "mapped_version": int(row["icd_version_mapped"]),
                    "code_system": row["code_system_mapped"],
                    "docid": str(row["docid"]),
                    "path_leaf_to_root": list(leaf_to_root),
                    "path_root_to_leaf": list(reversed(leaf_to_root)),
                    "docid_root_to_leaf": list(reversed(docid_leaf_to_root)),
                    "mapped_codes": set(),
                    "original_codes": set(),
                }
                metadata = mutable_catalog.setdefault(key, candidate)
                for field in (
                    "mapped_version",
                    "code_system",
                    "docid",
                    "path_leaf_to_root",
                    "path_root_to_leaf",
                    "docid_root_to_leaf",
                ):
                    if metadata[field] != candidate[field]:
                        raise ValueError(f"Target {key} has inconsistent {field}")
                metadata["mapped_codes"].add(str(row["icd_code"]))
                metadata["original_codes"].add(str(row["icd_code_original"]))

    audit = {
        split: {
            "official_admissions": 0,
            "retained_admissions": 0,
            "dropped_no_local_image": 0,
            "complete_image_admissions": 0,
            "partial_image_admissions": 0,
            "referenced_studies": 0,
            "available_image_studies": 0,
            "local_jpgs": 0,
        }
        for split in SPLITS
    }
    records: list[AdmissionRecord] = []
    for hadm_id, item in grouped.items():
        split = item["split"]
        audit[split]["official_admissions"] += 1
        studies: list[StudyResource] = []
        for report_id, study_id in zip(item["report_ids"], item["study_ids"]):
            parsed_study_id = int(report_id.rsplit(".", 1)[-1].removeprefix("s"))
            if parsed_study_id != study_id:
                raise ValueError(f"Study ID mismatch: {report_id} versus {study_id}")
            report_path, image_dir = _report_paths(report_id, mimic_root)
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"Referenced report is unavailable for admission {hadm_id}: {report_path}"
                )
            image_paths = tuple(str(path) for path in sorted(image_dir.glob("*.jpg")))
            studies.append(
                StudyResource(
                    study_id=study_id,
                    report_id=report_id,
                    report_path=str(report_path),
                    image_paths=image_paths,
                )
            )

        audit[split]["referenced_studies"] += len(studies)
        available_studies = sum(bool(study.image_paths) for study in studies)
        audit[split]["available_image_studies"] += available_studies
        audit[split]["local_jpgs"] += sum(len(study.image_paths) for study in studies)
        if available_studies == 0:
            audit[split]["dropped_no_local_image"] += 1
            continue

        complete = available_studies == len(studies)
        audit[split]["retained_admissions"] += 1
        audit[split][
            "complete_image_admissions" if complete else "partial_image_admissions"
        ] += 1
        records.append(
            AdmissionRecord(
                subject_id=item["subject_id"],
                hadm_id=hadm_id,
                split=split,
                studies=tuple(studies),
                target_keys=tuple(sorted(item["target_keys"])),
                all_referenced_images_available=complete,
            )
        )

    target_catalog: dict[str, dict[str, Any]] = {}
    for key, metadata in mutable_catalog.items():
        cleaned = dict(metadata)
        cleaned["mapped_codes"] = sorted(metadata["mapped_codes"])
        cleaned["original_codes"] = sorted(metadata["original_codes"])
        target_catalog[key] = cleaned

    records.sort(key=lambda record: (SPLITS.index(record.split), record.hadm_id))
    return LoadedAdmissions(tuple(records), target_catalog, audit)


def split_records(records: Sequence[AdmissionRecord], split: str) -> list[AdmissionRecord]:
    canonical = SPLIT_ALIASES[split]
    return [record for record in records if record.split == canonical]


def build_label_vocabulary(
    train_records: Sequence[AdmissionRecord], min_frequency: int = 1
) -> tuple[list[str], dict[str, int], Counter[str]]:
    """Build the target vocabulary from unique training admissions only."""
    frequency = Counter(
        key for record in train_records for key in set(record.target_keys)
    )
    keys = sorted(key for key, count in frequency.items() if count >= min_frequency)
    if not keys:
        raise ValueError(
            f"No ICD targets occur at least {min_frequency} times in the training split"
        )
    return keys, {key: index for index, key in enumerate(keys)}, frequency


def summarize_records(
    loaded: LoadedAdmissions,
    label_to_index: dict[str, int] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLITS:
        subset = split_records(loaded.records, split)
        retained = subset
        if label_to_index is not None:
            retained = [
                record
                for record in subset
                if any(key in label_to_index for key in record.target_keys)
            ]
        summary[split] = {
            **loaded.availability_audit[split],
            "patients_retained": len({record.subject_id for record in subset}),
            "admissions_with_in_vocab_target": len(retained),
            "unique_targets_present": len(
                {key for record in subset for key in record.target_keys}
            ),
        }
    return summary


def _unique_studies(records: Sequence[AdmissionRecord]) -> list[StudyResource]:
    resources: dict[int, StudyResource] = {}
    for record in records:
        for study in record.studies:
            previous = resources.setdefault(study.study_id, study)
            if previous != study:
                raise ValueError(f"Study {study.study_id} has inconsistent resources")
    return sorted(resources.values(), key=lambda study: study.study_id)


def _read_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class ImageStudyDataset(Dataset):
    def __init__(self, studies: Sequence[StudyResource]):
        self.studies = [study for study in studies if study.image_paths]

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> StudyResource:
        return self.studies[index]


class TextStudyDataset(Dataset):
    def __init__(self, studies: Sequence[StudyResource]):
        self.studies = list(studies)

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> StudyResource:
        return self.studies[index]


class ImageExtractionCollator:
    def __init__(self, image_processor: Any):
        self.image_processor = image_processor

    def __call__(self, studies: Sequence[StudyResource]) -> dict[str, Any]:
        images: list[Image.Image] = []
        counts: list[int] = []
        for study in studies:
            study_images = [_read_rgb(path) for path in study.image_paths]
            images.extend(study_images)
            counts.append(len(study_images))
        return {
            "studies": studies,
            "image_counts": counts,
            "pixel_values": self.image_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ],
        }


class TextExtractionCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, studies: Sequence[StudyResource]) -> dict[str, Any]:
        reports = [
            Path(study.report_path).read_text(encoding="utf-8", errors="replace")
            for study in studies
        ]
        inputs = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "studies": studies,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
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
        # Average over real tokens only (attention-mask weighted). The original
        # MedCLIP code averaged padding tokens too, which made each report's
        # embedding depend on the longest report in its extraction batch.
        hidden = hidden.permute(1, 0, 2, 3)  # [batch, layers, tokens, dim]
        mask = attention_mask[:, None, :, None].to(hidden.dtype)
        token_mean = (hidden * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        pooled = token_mean.mean(dim=1)
        return self.projection_head(pooled)


def _load_medclip_text_encoder(
    weights_dir: str | Path, device: torch.device
) -> tuple[nn.Module, Any]:
    from transformers import AutoModel, AutoTokenizer

    weights_path = Path(weights_dir) / "pytorch_model.bin"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"MedCLIP checkpoint not found at {weights_path}. "
            "Run setup_weights.sh first."
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
        raise ValueError(
            f"MedCLIP text checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )

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
    checkpoint_path = Path(checkpoint_path)
    source_dir = Path(source_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found at {checkpoint_path}. "
            "Run setup_weights.sh first."
        )
    if not (source_dir / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(
            f"Official DINOv3 source not found at {source_dir}. "
            "Run setup_weights.sh first."
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
    if os.environ.get("HF_TOKEN"):
        return
    token_path = Path(os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache/huggingface/token"))
    if token_path.is_file():
        os.environ["HF_TOKEN"] = token_path.read_text(encoding="utf-8").strip()


@torch.inference_mode()
def build_backbone_cache(
    records: Sequence[AdmissionRecord],
    config: ExperimentConfig,
    force: bool = False,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Cache one DINO vector per study and one MedCLIP vector per report."""
    config = config.resolve()
    cache_path = Path(config.cache_path)
    if cache_path.exists() and not force:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("metadata", {}).get("dataset_dir") != config.dataset_dir:
            raise ValueError("Existing cache belongs to a different dataset; use --force-cache")
        return cache

    _configure_huggingface_auth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder, image_processor = _load_dinov3_image_encoder(
        config.dino_checkpoint_path, config.dino_source_dir, device
    )
    text_encoder, tokenizer = _load_medclip_text_encoder(config.medclip_weights_dir, device)
    studies = _unique_studies(records)

    study_cache: dict[int, dict[str, torch.Tensor]] = {
        study.study_id: {} for study in studies
    }
    amp_enabled = config.use_amp and device.type == "cuda"

    image_loader = DataLoader(
        ImageStudyDataset(studies),
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=ImageExtractionCollator(image_processor),
        pin_memory=torch.cuda.is_available(),
    )
    image_dim = 0
    processed_images = 0
    for batch_index, batch in enumerate(
        tqdm(image_loader, desc="Caching DINOv3 study embeddings"), start=1
    ):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            features = F.normalize(image_encoder(pixels).float(), dim=-1)
        image_dim = int(features.shape[-1])
        offset = 0
        for study, count in zip(batch["studies"], batch["image_counts"]):
            study_feature = F.normalize(
                features[offset : offset + count].mean(dim=0), dim=-1
            )
            study_cache[study.study_id]["image_feature"] = study_feature.cpu().half()
            offset += count
        processed_images += sum(batch["image_counts"])
        if wandb_run is not None:
            wandb_run.log(
                {
                    "cache/image_batches_completed": batch_index,
                    "cache/image_batches_total": len(image_loader),
                    "cache/images_processed": processed_images,
                }
            )

    text_loader = DataLoader(
        TextStudyDataset(studies),
        batch_size=config.extraction_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=TextExtractionCollator(tokenizer),
        pin_memory=torch.cuda.is_available(),
    )
    text_dim = 0
    for batch_index, batch in enumerate(
        tqdm(text_loader, desc="Caching MedCLIP report embeddings"), start=1
    ):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            features = F.normalize(
                text_encoder(input_ids, attention_mask).float(), dim=-1
            )
        text_dim = int(features.shape[-1])
        for study, feature in zip(batch["studies"], features):
            study_cache[study.study_id]["text_feature"] = feature.cpu().half()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "cache/text_batches_completed": batch_index,
                    "cache/text_batches_total": len(text_loader),
                    "cache/reports_processed": min(
                        batch_index * config.extraction_batch_size, len(studies)
                    ),
                }
            )

    payload = {
        "metadata": {
            "dataset_dir": config.dataset_dir,
            "mimic_root": config.mimic_root,
            "dino_checkpoint_path": str(Path(config.dino_checkpoint_path).resolve()),
            "dino_source_dir": str(Path(config.dino_source_dir).resolve()),
            "medclip_weights_dir": str(Path(config.medclip_weights_dir).resolve()),
            "image_dim": image_dim,
            "text_dim": text_dim,
            "num_admissions": len(records),
            "num_studies": len(studies),
            "num_image_studies": sum("image_feature" in value for value in study_cache.values()),
            "num_reports": sum("text_feature" in value for value in study_cache.values()),
            "num_jpgs": processed_images,
        },
        "studies": study_cache,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, cache_path)
    return payload


class CachedICDAdmissionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[AdmissionRecord],
        cache: dict[str, Any],
        label_to_index: dict[str, int],
        max_image_studies: int | None = None,
        max_text_reports: int | None = None,
        training: bool = False,
    ):
        available = cache["studies"]
        self.records = []
        for record in records:
            has_label = any(key in label_to_index for key in record.target_keys)
            image_count = sum(
                "image_feature" in available.get(study.study_id, {})
                for study in record.studies
            )
            text_count = sum(
                "text_feature" in available.get(study.study_id, {})
                for study in record.studies
            )
            if has_label and image_count and text_count:
                self.records.append(record)
        self.cache = available
        self.label_to_index = label_to_index
        self.max_image_studies = max_image_studies
        self.max_text_reports = max_text_reports
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def _limit(self, values: torch.Tensor, maximum: int | None) -> torch.Tensor:
        if maximum is None or values.shape[0] <= maximum:
            return values
        if self.training:
            indices = torch.randperm(values.shape[0])[:maximum]
            return values[indices]
        return values[:maximum]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_features = torch.stack(
            [
                self.cache[study.study_id]["image_feature"].float()
                for study in record.studies
                if "image_feature" in self.cache[study.study_id]
            ]
        )
        text_features = torch.stack(
            [
                self.cache[study.study_id]["text_feature"].float()
                for study in record.studies
                if "text_feature" in self.cache[study.study_id]
            ]
        )
        image_features = self._limit(image_features, self.max_image_studies)
        text_features = self._limit(text_features, self.max_text_reports)

        labels = torch.zeros(len(self.label_to_index), dtype=torch.float32)
        for key in record.target_keys:
            label_index = self.label_to_index.get(key)
            if label_index is not None:
                labels[label_index] = 1.0

        return {
            "hadm_id": record.hadm_id,
            "subject_id": record.subject_id,
            "image_features": image_features,
            "text_features": text_features,
            "labels": labels,
            "complete_images": record.all_referenced_images_available,
        }


def _pad_sequences(
    values: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(values)
    maximum = max(value.shape[0] for value in values)
    feature_dim = values[0].shape[-1]
    padded = torch.zeros(batch_size, maximum, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, maximum, dtype=torch.bool)
    for index, value in enumerate(values):
        count = value.shape[0]
        padded[index, :count] = value
        mask[index, :count] = True
    return padded, mask


def collate_cached_admissions(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    image_features, image_mask = _pad_sequences(
        [item["image_features"] for item in items]
    )
    text_features, text_mask = _pad_sequences(
        [item["text_features"] for item in items]
    )
    return {
        "hadm_ids": [item["hadm_id"] for item in items],
        "subject_ids": [item["subject_id"] for item in items],
        "complete_images": [item["complete_images"] for item in items],
        "image_features": image_features,
        "image_mask": image_mask,
        "text_features": text_features,
        "text_mask": text_mask,
        "labels": torch.stack([item["labels"] for item in items]),
    }


def create_cached_loader(
    records: Sequence[AdmissionRecord],
    cache: dict[str, Any],
    label_to_index: dict[str, int],
    config: ExperimentConfig,
    training: bool,
) -> DataLoader:
    dataset = CachedICDAdmissionDataset(
        records,
        cache,
        label_to_index,
        max_image_studies=config.max_image_studies_per_admission,
        max_text_reports=config.max_text_reports_per_admission,
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        collate_fn=collate_cached_admissions,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def create_cached_loaders(
    records: Sequence[AdmissionRecord],
    cache: dict[str, Any],
    label_to_index: dict[str, int],
    config: ExperimentConfig,
) -> dict[str, DataLoader]:
    return {
        requested: create_cached_loader(
            split_records(records, canonical),
            cache,
            label_to_index,
            config,
            training=requested == "train",
        )
        for requested, canonical in (
            ("train", "train"),
            ("val", "validation"),
            ("test", "test"),
        )
    }


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
    """Shared admission-level classifier over image, text, and fused embeddings."""

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
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_studies = self.image_projection(image_features)
        text_reports = self.text_projection(text_features)
        image_embedding = F.normalize(
            masked_mean(image_studies, image_mask, dim=1), dim=-1
        )
        text_embedding = F.normalize(
            masked_mean(text_reports, text_mask, dim=1), dim=-1
        )
        fused_embedding = self.fusion_projection(
            torch.cat([image_embedding, text_embedding], dim=-1)
        )
        return {
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "fused_embedding": fused_embedding,
            "image_logits": self.classifier(image_embedding),
            "text_logits": self.classifier(text_embedding),
            "fused_logits": self.classifier(fused_embedding),
        }


def compute_pos_weight(
    dataset: CachedICDAdmissionDataset, cap: float = 50.0
) -> torch.Tensor:
    positives = torch.zeros(len(dataset.label_to_index), dtype=torch.float64)
    for record in dataset.records:
        for key in set(record.target_keys):
            index = dataset.label_to_index.get(key)
            if index is not None:
                positives[index] += 1
    negatives = len(dataset) - positives
    return (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=cap).float()


def classification_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {
        name: F.binary_cross_entropy_with_logits(
            outputs[f"{name}_logits"], labels,
            pos_weight=pos_weight if config.use_positive_weights else None
        )
        for name in ("image", "text", "fused")
    }
    total = (
        config.image_loss_weight * losses["image"]
        + config.text_loss_weight * losses["text"]
        + config.fused_loss_weight * losses["fused"]
    )
    return total, losses


def build_hierarchy_levels(
    label_keys: Sequence[str], target_catalog: dict[str, dict[str, Any]], depth: int = 5
) -> dict[str, Any]:
    """Per-level label -> class index maps for hierarchy-aware SupCon.

    Diagnoses (CM) use cumulative prefixes of the ICD-9 path (L1 chapter ...
    L5); procedures (PCS) use character prefixes of length 1..5 (section, body
    system, operation, body part, approach). The exact code is the last level.
    Level weights grow with depth (1, 2, ..., depth + 1), normalised.
    """
    def ancestors(key: str) -> list[str]:
        meta = target_catalog[key]
        if meta["code_system"] == "PCS":
            code = meta["path_root_to_leaf"][-1]
            return [f"PCS|{code[:n]}" for n in range(1, depth + 1)]
        path = meta["path_root_to_leaf"]
        return ["CM|" + "|".join(path[: min(n, len(path))]) for n in range(1, depth + 1)]

    per_key = [ancestors(key) for key in label_keys]
    indices = []
    for level in range(depth):
        classes = {name: i for i, name in enumerate(sorted({a[level] for a in per_key}))}
        indices.append(torch.tensor([classes[a[level]] for a in per_key], dtype=torch.long))
    weights = torch.arange(1, depth + 2, dtype=torch.float32)
    return {"indices": indices, "num_classes": [int(i.max()) + 1 for i in indices],
            "level_weights": weights / weights.sum()}


def _jaccard(sets: torch.Tensor) -> torch.Tensor:
    intersection = sets @ sets.T
    counts = sets.sum(dim=1)
    union = counts[:, None] + counts[None, :] - intersection
    return intersection / union.clamp_min(1.0)


def hierarchy_pair_weights(labels: torch.Tensor, levels: dict[str, Any]) -> torch.Tensor:
    """Depth-weighted mean over hierarchy levels (and the exact code) of the
    Jaccard overlap between two admissions' code sets at that level."""
    labels = labels.float()
    weights = levels["level_weights"].to(labels.device)
    total = weights[-1] * _jaccard(labels)
    for weight, index, size in zip(weights[:-1], levels["indices"], levels["num_classes"]):
        index = index.to(labels.device)
        sets = labels.new_zeros(labels.shape[0], size).index_add_(1, index, labels).clamp_(max=1.0)
        total = total + weight * _jaccard(sets)
    return total


def jaccard_supcon_loss(
    views: Sequence[torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 0.1,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft multilabel supervised contrastive loss with Jaccard positive weights.

    ``views`` holds L2-normalised admission embeddings of shape [B, D] (e.g.
    image, text and fused). All views of all admissions are contrasted against
    each other. The positive weight between two views is the Jaccard overlap of
    their admissions' label sets (1 for two views of the same admission); each
    anchor's own view is excluded. ``pair_weights`` [B, B] replaces the Jaccard
    matrix (e.g. hierarchy_pair_weights).
    """
    num_views = len(views)
    embeddings = torch.cat([view.float() for view in views], dim=0)
    jaccard = _jaccard(labels.float()) if pair_weights is None else pair_weights.float()
    weights = jaccard.repeat(num_views, num_views)
    self_mask = torch.eye(weights.shape[0], dtype=torch.bool, device=weights.device)
    weights = weights.masked_fill(self_mask, 0.0)

    similarity = embeddings @ embeddings.T / temperature
    similarity = similarity.masked_fill(self_mask, float("-inf"))
    log_prob = torch.log_softmax(similarity, dim=1).masked_fill(self_mask, 0.0)
    weight_sum = weights.sum(dim=1)
    valid = weight_sum > 0
    per_anchor = -(weights * log_prob).sum(dim=1) / weight_sum.clamp_min(1e-12)
    return per_anchor[valid].mean()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: MultimodalICDClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
    supcon_levels: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.train()
    totals = Counter()
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.use_amp and device.type == "cuda"
    )

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            outputs = model(
                batch["image_features"],
                batch["image_mask"],
                batch["text_features"],
                batch["text_mask"],
            )
            loss, parts = classification_loss(
                outputs, batch["labels"], pos_weight, config
            )
            if config.supcon_weight > 0:
                supcon = jaccard_supcon_loss(
                    [
                        outputs["image_embedding"],
                        outputs["text_embedding"],
                        outputs["fused_embedding"],
                    ],
                    batch["labels"],
                    config.supcon_temperature,
                    pair_weights=hierarchy_pair_weights(batch["labels"], supcon_levels)
                    if config.supcon_positive_mode == "hierarchy"
                    else None,
                )
                parts["supcon"] = supcon
                parts["bce_total"] = loss
                loss = loss + config.supcon_weight * supcon
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
) -> tuple[float, list[int], np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    admission_ids: list[int] = []
    labels: list[torch.Tensor] = []
    logits: dict[str, list[torch.Tensor]] = {
        "image": [],
        "text": [],
        "fused": [],
    }
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = _move_batch(batch, device)
        outputs = model(
            batch["image_features"],
            batch["image_mask"],
            batch["text_features"],
            batch["text_mask"],
        )
        loss, _ = classification_loss(outputs, batch["labels"], pos_weight, config)
        batch_size = batch["labels"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        admission_ids.extend(batch["hadm_ids"])
        labels.append(batch["labels"].cpu())
        for name in logits:
            logits[name].append(outputs[f"{name}_logits"].cpu())

    targets = torch.cat(labels).numpy()
    probabilities = {
        name: torch.sigmoid(torch.cat(parts)).numpy()
        for name, parts in logits.items()
    }
    return total_loss / max(total_samples, 1), admission_ids, targets, probabilities


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 5,
    macro: bool = True,
) -> dict[str, float]:
    """Multilabel metrics. ``macro=False`` skips the per-label macro AP/AUROC
    (the dominant cost: ~5 min per validation pass at 6k labels) and reports
    NaN for them; everything else is unchanged."""
    predictions = probabilities >= threshold
    metrics = {
        "micro_f1": float(
            f1_score(targets, predictions, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "micro_ap": float(
            average_precision_score(targets, probabilities, average="micro")
        ),
    }

    valid = (targets.sum(axis=0) > 0) & (targets.sum(axis=0) < targets.shape[0])
    metrics["evaluated_label_fraction"] = float(valid.mean())
    if macro and valid.any():
        metrics["macro_ap_valid"] = float(
            average_precision_score(
                targets[:, valid], probabilities[:, valid], average="macro"
            )
        )
        metrics["macro_auroc_valid"] = float(
            roc_auc_score(
                targets[:, valid], probabilities[:, valid], average="macro"
            )
        )
    else:
        metrics["macro_ap_valid"] = math.nan
        metrics["macro_auroc_valid"] = math.nan

    k = min(top_k, probabilities.shape[1])
    top_indices = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(targets, top_indices, axis=1).sum(axis=1)
    metrics[f"precision_at_{k}"] = float(np.mean(hits / k))
    metrics[f"recall_at_{k}"] = float(
        np.mean(hits / np.maximum(targets.sum(axis=1), 1))
    )
    return metrics


def _cm_hierarchy_metrics(
    admission_ids: Sequence[int],
    probabilities: dict[str, np.ndarray],
    label_keys: Sequence[str],
    target_catalog: dict[str, dict[str, Any]],
    records_by_id: dict[int, AdmissionRecord],
    config: ExperimentConfig,
) -> dict[str, Any]:
    cm_output_indices = [
        index
        for index, key in enumerate(label_keys)
        if target_catalog[key]["code_system"] == "CM"
    ]
    if not cm_output_indices:
        return {}
    max_depth = max(
        len(target_catalog[label_keys[index]]["path_root_to_leaf"])
        for index in cm_output_indices
    )
    results: dict[str, Any] = {}

    for level in range(1, max_depth + 1):
        prefix_to_leaf_indices: dict[tuple[Any, ...], list[int]] = {}
        for output_index in cm_output_indices:
            key = label_keys[output_index]
            metadata = target_catalog[key]
            path = metadata["path_root_to_leaf"]
            prefix = tuple(path[: min(level, len(path))])
            class_key = (
                metadata["mapped_version"],
                metadata["code_system"],
                *prefix,
            )
            prefix_to_leaf_indices.setdefault(class_key, []).append(output_index)
        class_keys = sorted(prefix_to_leaf_indices, key=lambda value: tuple(map(str, value)))
        class_to_index = {key: index for index, key in enumerate(class_keys)}

        coarse_targets = np.zeros(
            (len(admission_ids), len(class_keys)), dtype=np.float32
        )
        for row_index, hadm_id in enumerate(admission_ids):
            for target_key in records_by_id[hadm_id].target_keys:
                metadata = target_catalog[target_key]
                if metadata["code_system"] != "CM":
                    continue
                path = metadata["path_root_to_leaf"]
                prefix = tuple(path[: min(level, len(path))])
                class_key = (
                    metadata["mapped_version"],
                    metadata["code_system"],
                    *prefix,
                )
                class_index = class_to_index.get(class_key)
                if class_index is not None:
                    coarse_targets[row_index, class_index] = 1.0

        level_results: dict[str, Any] = {"class_count": len(class_keys)}
        for modality, leaf_probabilities in probabilities.items():
            coarse_probabilities = np.stack(
                [
                    leaf_probabilities[:, prefix_to_leaf_indices[key]].max(axis=1)
                    for key in class_keys
                ],
                axis=1,
            )
            level_results[modality] = multilabel_metrics(
                coarse_targets,
                coarse_probabilities,
                config.threshold,
                config.top_k,
            )
        results[f"level_{level}"] = level_results
    return results


def evaluate_modalities(
    model: MultimodalICDClassifier,
    loader: DataLoader,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
    label_keys: Sequence[str],
    target_catalog: dict[str, dict[str, Any]],
    records_by_id: dict[int, AdmissionRecord],
    full: bool = True,
) -> dict[str, Any]:
    """``full=False`` = light pass: micro metrics + P@k/R@k only (macro AP/AUROC
    NaN, no cm_hierarchy). Used for intermediate validation epochs when
    ``config.validation_full_metrics_every > 1``."""
    loss, admission_ids, targets, probabilities = predict(
        model, loader, pos_weight, config, device
    )
    return {
        "loss": loss,
        "sample_count": len(admission_ids),
        "full_metrics": full,
        "modalities": {
            name: multilabel_metrics(
                targets, values, config.threshold, config.top_k, macro=full
            )
            for name, values in probabilities.items()
        },
        "cm_hierarchy": _cm_hierarchy_metrics(
            admission_ids,
            probabilities,
            label_keys,
            target_catalog,
            records_by_id,
            config,
        )
        if full
        else {},
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: ExperimentConfig
) -> Any | None:
    """Per-epoch LR scheduler selected by ``config.lr_schedule``."""
    min_lr = config.learning_rate * config.lr_min_ratio
    if config.lr_schedule == "constant":
        return None
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=min_lr
        )
    if config.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=min_lr,
        )
    raise ValueError(f"unknown lr_schedule {config.lr_schedule!r}")


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

    (Path(config.task2_root) / "wandb").mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        dir=config.task2_root,
        config=asdict(config),
        tags=[
            "experiment01" if config.supcon_weight <= 0 else "experiment03",
            "classification-only" if config.supcon_weight <= 0 else "jaccard-supcon",
            "admission-level",
            "dinov3",
            "medclip",
        ],
        job_type="train",
    )


def save_checkpoint(
    path: str | Path,
    model: MultimodalICDClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ExperimentConfig,
    label_keys: Sequence[str],
    validation: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config.resolve()),
            "label_keys": list(label_keys),
            "validation": validation,
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def run_experiment(
    config: ExperimentConfig,
    build_cache: bool = False,
    force_cache: bool = False,
) -> dict[str, Any]:
    config = config.resolve()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_admission_records(config.dataset_dir, config.mimic_root)
    records = list(loaded.records)
    train_records = split_records(records, "train")
    label_keys, label_to_index, frequency = build_label_vocabulary(
        train_records, config.min_label_frequency
    )
    split_summary = summarize_records(loaded, label_to_index)
    records_by_id = {record.hadm_id: record for record in records}

    wandb_run = _initialize_wandb(config)
    if wandb_run is not None:
        metrics = {"data/label_count": len(label_keys), "stage/cache_started": 1}
        metrics.update(_flatten_wandb_metrics("data/splits", split_summary))
        wandb_run.log(metrics)

    cache_path = Path(config.cache_path)
    if build_cache or force_cache:
        cache = build_backbone_cache(
            records, config, force=force_cache, wandb_run=wandb_run
        )
    elif cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("metadata", {}).get("dataset_dir") != config.dataset_dir:
            raise ValueError("Embedding cache belongs to a different dataset; use --force-cache")
        if wandb_run is not None:
            wandb_run.log({"cache/reused": 1})
    else:
        raise FileNotFoundError(
            f"Embedding cache not found at {cache_path}. Re-run with --build-cache."
        )

    loaders = create_cached_loaders(records, cache, label_to_index, config)
    if not all(len(loader.dataset) for loader in loaders.values()):
        raise ValueError("At least one split is empty after cache and target filtering")

    complete_test_records = [
        record
        for record in split_records(records, "test")
        if record.all_referenced_images_available
    ]
    complete_test_loader = create_cached_loader(
        complete_test_records,
        cache,
        label_to_index,
        config,
        training=False,
    )

    image_dim = int(cache["metadata"]["image_dim"])
    text_dim = int(cache["metadata"]["text_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalICDClassifier(
        image_dim=image_dim,
        text_dim=text_dim,
        embedding_dim=config.embedding_dim,
        num_labels=len(label_keys),
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config)
    if config.supcon_positive_mode not in ("jaccard", "hierarchy"):
        raise ValueError(f"unknown supcon_positive_mode {config.supcon_positive_mode!r}")
    supcon_levels = (
        build_hierarchy_levels(label_keys, loaded.target_catalog)
        if config.supcon_positive_mode == "hierarchy"
        else None
    )
    pos_weight = compute_pos_weight(
        loaders["train"].dataset, config.pos_weight_cap
    ).to(device)

    manifest = {
        "config": asdict(config),
        "label_count": len(label_keys),
        "label_frequency": dict(frequency),
        "split_summary": split_summary,
        "cache_metadata": cache["metadata"],
        "complete_image_test_admissions": len(complete_test_loader.dataset),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "labels.json").write_text(
        json.dumps(label_keys, indent=2), encoding="utf-8"
    )
    (output_dir / "target_catalog.json").write_text(
        json.dumps(loaded.target_catalog, indent=2), encoding="utf-8"
    )

    if wandb_run is not None:
        wandb_run.log(
            {
                "stage/cache_complete": 1,
                "cache/image_dim": image_dim,
                "cache/text_dim": text_dim,
                "cache/num_studies": int(cache["metadata"]["num_studies"]),
                "cache/num_image_studies": int(
                    cache["metadata"]["num_image_studies"]
                ),
                "cache/num_reports": int(cache["metadata"]["num_reports"]),
                "storage/cache_size_mib": cache_path.stat().st_size / (1024**2),
                "storage/free_gib": shutil.disk_usage(cache_path.parent).free / (1024**3),
            }
        )

    try:
        history: list[dict[str, Any]] = []
        best_score = -math.inf
        epochs_since_best = 0
        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, pos_weight, config, device,
                supcon_levels=supcon_levels,
            )
            train_metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
            full_validation = (
                config.validation_full_metrics_every <= 1
                or epoch % config.validation_full_metrics_every == 0
                or epoch == config.epochs
            )
            validation = evaluate_modalities(
                model,
                loaders["val"],
                pos_weight,
                config,
                device,
                label_keys,
                loaded.target_catalog,
                records_by_id,
                full=full_validation,
            )
            epoch_result = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
            history.append(epoch_result)
            print(json.dumps(epoch_result, indent=2))

            score = validation["modalities"]["fused"]["micro_ap"]
            if wandb_run is not None:
                epoch_metrics = {"epoch": epoch}
                epoch_metrics.update(_flatten_wandb_metrics("train", train_metrics))
                epoch_metrics.update(_flatten_wandb_metrics("validation", validation))
                wandb_run.log(epoch_metrics)

            if score > best_score:
                best_score = score
                epochs_since_best = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    config,
                    label_keys,
                    validation,
                )
            else:
                epochs_since_best += 1
                if (
                    config.early_stopping_patience > 0
                    and epochs_since_best >= config.early_stopping_patience
                ):
                    print(
                        f"Early stopping at epoch {epoch}: no validation fused "
                        f"micro AP gain for {epochs_since_best} epochs "
                        f"(best {best_score:.4f})."
                    )
                    break
            if scheduler is not None:
                if config.lr_schedule == "plateau":
                    scheduler.step(score)
                else:
                    scheduler.step()

        checkpoint = torch.load(
            output_dir / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_results = evaluate_modalities(
            model,
            loaders["test"],
            pos_weight,
            config,
            device,
            label_keys,
            loaded.target_catalog,
            records_by_id,
        )
        complete_test_results = evaluate_modalities(
            model,
            complete_test_loader,
            pos_weight,
            config,
            device,
            label_keys,
            loaded.target_catalog,
            records_by_id,
        )
        results = {
            "history": history,
            "best_epoch": checkpoint["epoch"],
            "test": test_results,
            "test_complete_images": complete_test_results,
        }
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        if wandb_run is not None:
            wandb_run.log(_flatten_wandb_metrics("test", test_results))
            wandb_run.log(
                _flatten_wandb_metrics(
                    "test_complete_images", complete_test_results
                )
            )
            wandb_run.summary["best_epoch"] = checkpoint["epoch"]
            wandb_run.summary["best_validation_fused_micro_ap"] = best_score
            for name, value in _flatten_wandb_metrics(
                "test", test_results
            ).items():
                wandb_run.summary[name] = value
        return results
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task2-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--mimic-root", type=str, default=ExperimentConfig.mimic_root)
    parser.add_argument("--dataset-dir", type=str, default=ExperimentConfig.dataset_dir)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    parser.add_argument("--min-label-frequency", type=int, default=1)
    parser.add_argument("--max-image-studies-per-admission", type=int, default=None)
    parser.add_argument("--max-text-reports-per-admission", type=int, default=None)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "mimic-icd-task2"),
    )
    parser.add_argument("--wandb-entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.getenv(
            "WANDB_RUN_NAME", "experiment01-admission-classification"
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "disabled"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = ExperimentConfig(
        task2_root=args.task2_root,
        mimic_root=args.mimic_root,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        extraction_batch_size=args.extraction_batch_size,
        min_label_frequency=args.min_label_frequency,
        max_image_studies_per_admission=args.max_image_studies_per_admission,
        max_text_reports_per_admission=args.max_text_reports_per_admission,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    run_experiment(
        configuration,
        build_cache=args.build_cache,
        force_cache=args.force_cache,
    )
