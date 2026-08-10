#!/usr/bin/env bash
set -euo pipefail

# sudo chmod a+rw /dev/ttyACM0
# Ablation examples (all extra arguments are forwarded to the Python entry point):
#   bash run_fitting_final.sh --ablation shape-fitting
#   bash run_fitting_final.sh --ablation tactile-sensing
#   bash run_fitting_final.sh --ablation silhouette-scaling
# Place without grasp-offset XY compensation:
#   bash run_fitting_final.sh --disable-place-grasp-offset-xy
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
main_python="${HANDOVER_PYTHON:-python}"

exec "$main_python" "$repo_root/robot_control_rtde_fitting_final.py" \
 --config "$repo_root/configs/handover.yaml" \
 --object-backend sam3d \
 --enable-follow --follow-z \
 --enable-pre-release-descend-before-open \
 "$@"
#  --profile-runtime
 #--record-video \
 #--debug-tactile
#  --3d-debug \
#  --save-image \
