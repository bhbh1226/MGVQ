#!/usr/bin/env bash

# ===== model zoo =====
# 사용법: MODEL=my-model ./eval_recon.sh
MODEL="${MODEL:-mgvq-f16c32}"
CKPT="${CKPT:-/mgvq_f16c32_g8.pt}"
DS_RATE="${DS_RATE:-16}"

# ===== knobs =====
CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"      # 16384 | 32768
CODEBOOK_GROUPS="${CODEBOOK_GROUPS:-8}"      # 4 | 8
GROUPS_TO_USE="${GROUPS_TO_USE:-4}"          # <= CODEBOOK_GROUPS
DATASET="${DATASET:-imagenet256p}"           # imagenet256p | UHDBench2k | dicom | dicom512
DATASET_ROOT="${DATASET_ROOT:-/path/to/dataset}"      # space-separated multiple paths

OUT="${OUT:-/path/to/results}"

echo "Running with MODEL=$MODEL, CKPT=$CKPT ..."

# for imagenet 256p reconstruction evaluation
python eval_recon.py \
  --vq-model "$MODEL" \
  --vq-ckpt "$CKPT" \
  --codebook-size "$CODEBOOK_SIZE" \
  --codebook-groups "$CODEBOOK_GROUPS" \
  --groups-to-use "$GROUPS_TO_USE" \
  --eval-dataset "$DATASET" \
  --ds-rate "$DS_RATE" \
  --path-to-save "$OUT" \
  --dataset-root $DATASET_ROOT \
  --eval-fid