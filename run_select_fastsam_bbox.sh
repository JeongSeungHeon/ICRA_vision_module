#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
main_python="${HANDOVER_PYTHON:-/home/ur5/miniforge3/envs/handover_ros2/bin/python}"

exec "$main_python" "$repo_root/tools/select_fastsam_bbox.py" \
  --config "$repo_root/configs/handover.yaml" \
  --output "$repo_root/output/sam3d/fastsam_bbox.yaml" \
  --repo-root "$repo_root"
