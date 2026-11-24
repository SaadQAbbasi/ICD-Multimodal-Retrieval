#!/usr/bin/env bash
set -euo pipefail

TASK2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAINED_DIR="$TASK2_DIR/pretrained"
VENDOR_DIR="$TASK2_DIR/vendor"

MEDCLIP_ARCHIVE="$PRETRAINED_DIR/medclip-vit-pretrained.zip"
MEDCLIP_URL="https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip"
MEDCLIP_CHECKPOINT="$PRETRAINED_DIR/pytorch_model.bin"

DINO_ARCHIVE="$PRETRAINED_DIR/dinov3-small-weights.zip"
DINO_DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/saadqasimabbasi/dino-v3-small-weights"
DINO_WEIGHTS_DIR="$PRETRAINED_DIR/dinov3-vits16"
DINO_CHECKPOINT="$DINO_WEIGHTS_DIR/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

DINO_SOURCE_COMMIT="6876159a11b4df116f30f667f8c9888617df0751"
DINO_SOURCE_ARCHIVE="$VENDOR_DIR/dinov3-$DINO_SOURCE_COMMIT.zip"
DINO_SOURCE_URL="https://github.com/facebookresearch/dinov3/archive/$DINO_SOURCE_COMMIT.zip"
DINO_SOURCE_DIR="$VENDOR_DIR/dinov3"
DINO_EXTRACTED_DIR="$VENDOR_DIR/dinov3-$DINO_SOURCE_COMMIT"

mkdir -p "$TASK2_DIR/.cache/huggingface" "$PRETRAINED_DIR" "$VENDOR_DIR"

if [[ ! -f "$MEDCLIP_CHECKPOINT" ]]; then
  echo "Downloading the official MedCLIP-ViT checkpoint..."
  curl --fail --location --retry 3 --continue-at - --output "$MEDCLIP_ARCHIVE" "$MEDCLIP_URL"
  unzip -q -o "$MEDCLIP_ARCHIVE" -d "$PRETRAINED_DIR"
fi

if [[ ! -f "$DINO_CHECKPOINT" ]]; then
  echo "Downloading the DINOv3 ViT-S/16 checkpoint from the Task 2 Kaggle dataset..."
  mkdir -p "$DINO_WEIGHTS_DIR"
  curl --fail --location --retry 3 --continue-at - --output "$DINO_ARCHIVE" "$DINO_DATASET_URL"
  unzip -q -o "$DINO_ARCHIVE" -d "$DINO_WEIGHTS_DIR"
fi

if [[ ! -f "$DINO_SOURCE_DIR/dinov3/hub/backbones.py" ]]; then
  echo "Downloading the pinned official Meta DINOv3 source..."
  curl --fail --location --retry 3 --continue-at - --output "$DINO_SOURCE_ARCHIVE" "$DINO_SOURCE_URL"
  unzip -q -o "$DINO_SOURCE_ARCHIVE" -d "$VENDOR_DIR"
  mv "$DINO_EXTRACTED_DIR" "$DINO_SOURCE_DIR"
fi

for checkpoint in "$MEDCLIP_CHECKPOINT" "$DINO_CHECKPOINT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Expected checkpoint not found: $checkpoint" >&2
    exit 1
  fi
done

if [[ ! -f "$DINO_SOURCE_DIR/dinov3/hub/backbones.py" ]]; then
  echo "Expected DINOv3 source not found: $DINO_SOURCE_DIR" >&2
  exit 1
fi

rm -f -- "$MEDCLIP_ARCHIVE" "$DINO_ARCHIVE" "$DINO_SOURCE_ARCHIVE"

echo "MedCLIP checkpoint ready at $MEDCLIP_CHECKPOINT"
echo "DINOv3 checkpoint ready at $DINO_CHECKPOINT"
echo "DINOv3 source ready at $DINO_SOURCE_DIR"
