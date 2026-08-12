#!/usr/bin/env bash
set -euo pipefail

env_prefix="${1:-/home/ur5/miniforge3/envs/hoi_detr}"
repo_root="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
env_python="${env_prefix}/bin/python"
hoi_repo="${repo_root}/external/HOI-DETR"

if [[ ! -x "${env_python}" ]]; then
  echo "HOI-DETR environment Python not found: ${env_python}" >&2
  exit 1
fi
if [[ ! -f "${hoi_repo}/mmdet/__init__.py" || ! -f "${hoi_repo}/projects/__init__.py" ]]; then
  echo "HOI-DETR checkout is incomplete: ${hoi_repo}" >&2
  exit 1
fi

"${env_python}" -m pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
"${env_python}" -m pip install \
  mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
"${env_python}" -m pip install -e "${hoi_repo}"

PYTHONPATH="${hoi_repo}${PYTHONPATH:+:${PYTHONPATH}}" "${env_python}" -c \
  'import mmcv, torch; import mmdet; import projects; print("torch", torch.__version__, "mmcv", mmcv.__version__, "cuda", torch.cuda.is_available())'
