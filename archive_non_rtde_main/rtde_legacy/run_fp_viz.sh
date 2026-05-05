#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/handover.yaml}"
YOLO_MODEL="${YOLO_MODEL:-yoloe-26x-seg.pt}"

exec "${PYTHON_BIN}" visualize_pose_tracking.py \
  --config "${CONFIG_PATH}" \
  --model "${YOLO_MODEL}" \
  --select-mode highest_score \
  "$@"
