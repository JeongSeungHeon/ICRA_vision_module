This folder contains files moved out of the main workspace while keeping
`robot_control_rtde.py` and its runtime dependencies in their original paths.

Moved here:
- alternative entry points and runners
- debug/visualization tools
- demo scripts
- unused wrapper modules
- unused calibration backups and extra artifacts
- archived output files

Main runtime files remain in the repository root packages:
- `robot_control_rtde.py`
- `calibration/`
- `camera_parameters/` required by `configs/handover.yaml`
- `configs/`
- `handpose3d/hand_pose_6d.py`
- `object_pt_extraction/` runtime modules
- `perception/` runtime modules
- `robot/` RTDE runtime modules
- `system/dual_sensor_hub.py`
- `system/shared_state.py`
- `utils/` runtime modules
