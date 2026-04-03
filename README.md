# ICRA Vision Module

This repository is organized around one main entry point:

```bash
python robot_control_rtde.py --prompt cup --select-mode highest_score --enable-follow --follow-z
```

The code runs a dual-RealSense perception pipeline, estimates hand and object state, builds a grasp target, and controls a UR robot through RTDE.

## Main Entry Point

- Main script: [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py)
- Main config: [`configs/handover.yaml`](/home/ur5/ICRA_vision_module/configs/handover.yaml)
- Convenience launcher: [`run.sh`](/home/ur5/ICRA_vision_module/run.sh)

## Active Repository Layout

These paths are kept in the main workspace because `robot_control_rtde.py` depends on them at runtime:

- [`robot_control_rtde.py`](/home/ur5/ICRA_vision_module/robot_control_rtde.py)
- [`configs/`](/home/ur5/ICRA_vision_module/configs)
- [`camera_parameters/`](/home/ur5/ICRA_vision_module/camera_parameters)
- [`calibration/`](/home/ur5/ICRA_vision_module/calibration)
- [`system/`](/home/ur5/ICRA_vision_module/system)
- [`perception/`](/home/ur5/ICRA_vision_module/perception)
- [`object_pt_extraction/`](/home/ur5/ICRA_vision_module/object_pt_extraction)
- [`robot/`](/home/ur5/ICRA_vision_module/robot)
- [`utils/`](/home/ur5/ICRA_vision_module/utils)
- [`handpose3d/`](/home/ur5/ICRA_vision_module/handpose3d)

Files and folders not required by the main RTDE workflow were moved to:

- [`archive_non_rtde_main/`](/home/ur5/ICRA_vision_module/archive_non_rtde_main)

That archive contains older demos, visualization tools, alternate runners, unused wrappers, backup calibration files, and old outputs.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Hardware/runtime dependencies:

- 2 RealSense cameras
- UR robot reachable over RTDE
- Robotiq gripper if gripper control is enabled
- YOLOE weights file available locally

The repository currently keeps these local weights in the root:

- `yoloe-26l-seg.pt`
- `yoloe-26x-seg.pt`

## Configuration

Most runtime behavior is controlled from [`configs/handover.yaml`](/home/ur5/ICRA_vision_module/configs/handover.yaml):

- camera serials, resolution, fps, and depth filters
- calibration chain files
- hand/object perception thresholds
- grasp target settings
- UR RTDE IP and motion parameters
- workspace and safety bounds

Calibration files used by the current config:

- `camera_parameters/c0_to_robot.pckl`
- `camera_parameters/rot_trans_c1.dat`

## Running

Basic example:

```bash
python robot_control_rtde.py --prompt cup --select-mode highest_score --enable-follow --follow-z
```

Using the helper script:

```bash
bash run.sh
```

Useful options:

- `--prompt`: segmentation target classes
- `--config`: alternate YAML config path
- `--robot-ip`: override robot IP from config
- `--enable-follow`: enable live follow mode
- `--follow-z` or `--no-follow-z`: toggle Z-axis following
- `--open-gripper`: open gripper during initialization

To see the full CLI:

```bash
python robot_control_rtde.py --help
```

## Notes

- The script assumes package imports resolve from the repository root.
- Some dependencies such as `pyrealsense2`, `mediapipe`, `ultralytics`, and `ur-rtde` are optional at import time but required for real hardware execution.
- If you change calibration or config paths, update [`configs/handover.yaml`](/home/ur5/ICRA_vision_module/configs/handover.yaml) first rather than hard-coding paths in the Python files.
