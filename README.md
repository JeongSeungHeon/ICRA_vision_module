# ICRA Vision Module

This repository is currently organized around one active handover pipeline:

```bash
bash run_fitting_final.sh
```

The main script runs dual RealSense perception, object segmentation, hand pose lifting, point-cloud fusion, template shape fitting, fill-level estimation, grasp target generation, UR RTDE control, Robotiq gripper control, metadata logging, and web-assisted video recording.

## Main Entry Point

- Main script: `robot_control_rtde_fitting_final.py`
- Main config: `configs/handover.yaml`
- Convenience launcher: `run_fitting_final.sh`

The launcher currently runs:

```bash
python robot_control_rtde_fitting_final.py \
  --select-mode highest_score \
  --enable-follow --follow-z
```

## Active Layout

These paths are part of the current runtime path:

- `robot_control_rtde_fitting_final.py`: top-level perception and robot-control loop
- `configs/handover.yaml`: active dual-camera, perception, grasp, RTDE, and workspace config
- `calibration/`, `camera_parameters/`: camera-to-robot calibration chain
- `system/`: dual RealSense frame hub and shared state dataclasses
- `perception/`: object, hand, fusion, grasp, shape fitting, fill-level, and FDCT helpers
- `object_pt_extraction/`: segmentation and point-cloud utilities
- `robot/`: RTDE and Robotiq gripper controller code
- `utils/`: RealSense, metadata, preprocessing, and depth helpers
- `handpose3d/`: hand pose lifting support
- `shape_fitting/`: template point clouds used by `ShapeFittingTracker`
- `video_record.py`: local web UI and recording service for handover videos
- `FDCT/`: optional depth-completion model assets
- `benchmarks/`: benchmark and metadata helper scripts

Legacy runners and older visualization scripts have been moved under:

```text
archive_non_rtde_main/
```

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Expected hardware/runtime pieces:

- Two Intel RealSense cameras
- UR robot reachable over RTDE
- Robotiq gripper daemon/socket access if gripper control is enabled
- Local YOLOE weights, for example `yoloe-26l-seg.pt`
- Open3D for shape fitting
- Optional FDCT checkpoint at `FDCT/TransCG.tar` if FDCT depth completion is enabled

## Configuration

The active config is `configs/handover.yaml`.

Important sections:

- `cameras`: RealSense serials, resolution, fps, and depth filters
- `calibration`: cam0-to-base and cam1-to-cam0 transform files
- `perception.depth_completion.fdct`: optional FDCT depth completion
- `perception.object`: YOLOE segmentation and object point-cloud parameters
- `perception.hand`: MediaPipe hand detection and depth lifting parameters
- `perception.fusion`: temporal filtering and hand-approach activation
- `perception.shape_fitting`: template library, clustering, scale init, ICP, and output downsampling
- `perception.fill_level_estimation`: cup/glass fill-level estimator parameters
- `grasp`: grasp target selection and hand-relative dropout fallback
- `home_pose`: startup and post-task home pose
- `robot`: RTDE, gripper, grasp verification, and frame mapping
- `safety.workspace_bounds_m`: workspace clamp used by the final script

Calibration files referenced by the active config:

```text
camera_parameters/c0_to_robot.pckl
camera_parameters/rot_trans_c1.dat
```

Shape templates referenced by the active config:

```text
shape_fitting/template.npy
shape_fitting/wine_glass.npy
```

## Running

Start the current final pipeline:

```bash
bash run_fitting_final.sh
```

Or call the script directly:

```bash
python robot_control_rtde_fitting_final.py \
  --config configs/handover.yaml \
  --select-mode highest_score \
  --enable-follow \
  --follow-z
```

Useful options:

- `--config`: alternate YAML config path
- `--model`: YOLOE model or local weights path
- `--prompt`: segmentation prompt classes
- `--select-mode`: instance selection policy
- `--robot-ip`: override `robot.rtde.robot_ip`
- `--enable-follow`: enable RTDE follow thread
- `--follow-z` / `--no-follow-z`: toggle Z-axis following
- `--open-gripper`: open gripper during startup
- `--show-depth`: show a cam0 depth preview window
- `--enable-fdct-depth` / `--disable-fdct-depth`: override FDCT setting from config

For the full CLI:

```bash
python robot_control_rtde_fitting_final.py --help
```

## Runtime Flow

At a high level, the final script does this:

1. Starts the dual RealSense hub from `configs/handover.yaml`.
2. Runs object segmentation on both cameras.
3. Builds per-camera object point clouds and merges them in robot base frame.
4. Runs hand pose detection and selects the active hand.
5. Fits the configured object template with `perception.shape_fitting_tracker_v2`.
6. Estimates fill level and updates handover metadata.
7. Computes a grasp target from fitted object geometry and hand state.
8. Follows the target with UR RTDE until the direct grasp trigger fires.
9. Closes the gripper, verifies grasp, records contact timing, and saves grasp offset.
10. Returns to the home/object placement area, opens the gripper, and records delivery metadata.
11. Keeps a local video recorder UI available for saving/discarding task videos.

## Keyboard Controls

While the OpenCV windows are focused:

- `f`: toggle follow mode
- `r`: reset system to startup state
- `s`: stop/finalize the current handover metadata and video recording
- `q` or `Esc`: quit

## Notes

- Run commands from the repository root so relative config, calibration, model, and template paths resolve correctly.
- `FoundationPose/` is not required by `robot_control_rtde_fitting_final.py`; older FoundationPose runners are archived.
- Some tests still cover older experimental modules. Treat the active runtime path as the final script plus the modules listed in the Active Layout section.
