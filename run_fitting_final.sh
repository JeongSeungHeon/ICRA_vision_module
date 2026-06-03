# sudo chmod a+rw /dev/ttyACM0

python robot_control_rtde_fitting_final.py   \
 --select-mode highest_score \
 --enable-follow --follow-z \
 --enable-pre-release-descend-before-open \
 --debug-tactile
#  --profile-runtime
 #--record-video \
 #--debug-tactile \
#  --save-image \
#  --3d-debug \
