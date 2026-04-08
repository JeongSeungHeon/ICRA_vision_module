#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/handover.yaml}"
YOLO_MODEL="${YOLO_MODEL:-yoloe-26x-seg.pt}"

echo "[INFO] Starting dual RealSense + scale-adaptive FoundationPose pipeline"
echo "[INFO] Config: ${CONFIG_PATH}"
echo "[INFO] Model:  ${YOLO_MODEL}"

if [ ! -f "${YOLO_MODEL}" ]; then
  echo "[WARN] YOLO weights not found at ${YOLO_MODEL}"
fi

if [ ! -d "FoundationPose/weights" ]; then
  echo "[WARN] FoundationPose weights directory is missing: FoundationPose/weights"
  echo "[WARN] Pose tracker backend may stay unavailable until weights are installed."
fi

exec "${PYTHON_BIN}" robot_control_rtde_v2.py \
  --config "${CONFIG_PATH}" \
  --model "${YOLO_MODEL}" \
  --select-mode highest_score \
  "$@"
