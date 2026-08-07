#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sam3d_python="${SAM3D_PYTHON:-/home/ur5/miniforge3/envs/sam3d-objects/bin/python}"

exec "$sam3d_python" "$repo_root/tools/sam3d_model_server.py" \
  --config "$repo_root/configs/handover.yaml" \
  --sam3d-repo "$repo_root/external/sam-3d-objects" \
  --socket /tmp/handover_sam3d_stage1.sock
