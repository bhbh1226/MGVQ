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
DATASET="${DATASET:-imagenet256p}"           # imagenet256p | UHDBench2k | dicom | dicom512 | nifti | nifti256 | nifti512 | tensor | tensor256 | tensor512
DATASET_ROOT="${DATASET_ROOT:-/path/to/dataset}"      # space-separated multiple paths

OUT="${OUT:-/path/to/results}"

RESIZE_MODE="${RESIZE_MODE:-center_crop}"  # center_crop | fit_pad | stretch

# ===== DICOM options =====
DICOM_WINDOW="${DICOM_WINDOW:-}"             # lung | soft_tissue | bone | brain | liver | mediastinum | abdomen | default
DICOM_WINDOW_MIN="${DICOM_WINDOW_MIN:-}"     # custom HU window min
DICOM_WINDOW_MAX="${DICOM_WINDOW_MAX:-}"     # custom HU window max
NO_WINDOWING="${NO_WINDOWING:-true}"  # true | false

# ===== NIfTI/Tensor volumetric options =====
SLICE_AXIS="${SLICE_AXIS:-1}"                # 0=sagittal, 1=coronal, 2=axial
SLICE_IDX="${SLICE_IDX:-}"                   # specific slice index (optional)
SLICE_RANGE="${SLICE_RANGE:-}"               # slice range, format: 'start:end' or 'start:end:step' (e.g., '50:150' or '50:150:2')
HU_WINDOW="${HU_WINDOW:-}"                   # lung | soft_tissue | bone | brain | liver | mediastinum | abdomen
HU_WINDOW_MIN="${HU_WINDOW_MIN:-}"           # custom HU window min for NIfTI/Tensor
HU_WINDOW_MAX="${HU_WINDOW_MAX:-}"           # custom HU window max for NIfTI/Tensor

echo "Running with MODEL=$MODEL, CKPT=$CKPT, DATASET=$DATASET ..."

# Build optional arguments
OPTIONAL_ARGS=""

# DICOM options
if [ -n "$DICOM_WINDOW" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --dicom-window $DICOM_WINDOW"
fi
if [ -n "$DICOM_WINDOW_MIN" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --dicom-window-min $DICOM_WINDOW_MIN"
fi
if [ -n "$DICOM_WINDOW_MAX" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --dicom-window-max $DICOM_WINDOW_MAX"
fi
if [ "$NO_WINDOWING" = "true" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --no-windowing"
fi

# NIfTI/Tensor volumetric options
if [ -n "$SLICE_AXIS" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --slice-axis $SLICE_AXIS"
fi
if [ -n "$SLICE_IDX" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --slice-idx $SLICE_IDX"
fi
if [ -n "$SLICE_RANGE" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --slice-range $SLICE_RANGE"
fi
if [ -n "$HU_WINDOW" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --hu-window $HU_WINDOW"
fi
if [ -n "$HU_WINDOW_MIN" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --hu-window-min $HU_WINDOW_MIN"
fi
if [ -n "$HU_WINDOW_MAX" ]; then
  OPTIONAL_ARGS="$OPTIONAL_ARGS --hu-window-max $HU_WINDOW_MAX"
fi

# Run evaluation
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
  --resize-mode "$RESIZE_MODE" \
  $OPTIONAL_ARGS \
  --eval-fid