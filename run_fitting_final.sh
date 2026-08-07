#!/usr/bin/env bash
set -euo pipefail

# sudo chmod a+rw /dev/ttyACM0
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
