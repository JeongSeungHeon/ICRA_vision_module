#!/usr/bin/env bash
set -euo pipefail

env_prefix="${1:-/home/ur5/miniforge3/envs/hands23_ros2}"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  echo "Hands23 environment Python not found: ${env_prefix}/bin/python" >&2
  exit 1
fi
if [[ ! -x "${env_prefix}/bin/nvcc" ]]; then
  echo "Hands23 CUDA compiler not found: ${env_prefix}/bin/nvcc" >&2
  exit 1
fi

# Detectron2 v0.6 imports torch from setup.py, so build isolation must be
# disabled. FORCE_CUDA also makes the build deterministic on headless setup
# hosts, while the architecture list targets the RTX 4090 (Ada, sm_89).
CUDA_HOME="${env_prefix}" \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST="8.9" \
MAX_JOBS=8 \
PATH="${env_prefix}/bin:${PATH}" \
LD_LIBRARY_PATH="${env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONNOUSERSITE=1 \
"${env_prefix}/bin/python" -m pip install \
  --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@v0.6"
