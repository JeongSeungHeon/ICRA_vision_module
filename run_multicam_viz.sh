#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/handover.yaml}"
PRINT_EVERY="${PRINT_EVERY:-5}"

export PYTHONPATH="${SCRIPT_DIR}/archive_non_rtde_main:${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" archive_non_rtde_main/tools/visualize_handover_perception.py \
  --config "${CONFIG_PATH}" \
  --show-2d \
  --print-every "${PRINT_EVERY}" \
  "$@"
