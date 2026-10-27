#!/usr/bin/env bash
# Piper VITS Voice Adaptation Recipe
# Prerequisites:
#   pip install piper-train
#   Base model checkpoint: en_US-ryan-medium.ckpt

set -euo pipefail

DATASET_DIR="${1:-./dataset}"
OUTPUT_DIR="${2:-./training_dir}"
BASE_CHECKPOINT="${3:-./checkpoints/en_US-ryan-medium.ckpt}"
BATCH_SIZE="${4:-16}"
MAX_EPOCHS="${5:-500}"

echo "=== 1. Building Phoneme Alignment Cache ==="
python3 -m piper_train.preprocess \
    --language en-us \
    --input-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}/cache" \
    --dataset-format ljspeech \
    --sample-rate 22050

echo "=== 2. Launching VITS Fine-Tuning ==="
python3 -m piper_train \
    --dataset-dir "${OUTPUT_DIR}/cache" \
    --checkpoint-path "${BASE_CHECKPOINT}" \
    --accelerator "gpu" \
    --devices 1 \
    --batch-size "${BATCH_SIZE}" \
    --validation-split 0.05 \
    --max_epochs "${MAX_EPOCHS}" \
    --checkpoint-epochs 10 \
    --precision 32 \
    --default_root_dir "${OUTPUT_DIR}"

echo "=== Training Complete ==="
