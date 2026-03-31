# Robot Control Implementation Plan

## Goal

Build a dual-camera single-PC receive-and-place system that runs:

- object segmentation and point cloud extraction from two cameras
- hand 3D pose estimation from two cameras
- merged object point cloud generation
- UR5 robot control over RTDE

without ROS, while keeping perception and robot control decoupled enough for stable runtime behavior.

This document is a step-by-step plan for turning the current prototype scripts into a runnable integrated system.

## Fixed Design Decisions

The following details are now fixed for the first implementation:

- use two cameras
- merge object point clouds from both cameras
- do not fuse hand poses across cameras
- if a right hand is detected, use camera 0 hand pose
- if a left hand is detected, use camera 1 hand pose
- start the robot only after the hand approaches the object and the object is lifted
- define lift using the configured height axis, not necessarily base-frame `z`
- use a lift threshold of at least `5 cm`
- default height axis to base-frame `y`, because it commonly represents the up/down direction in this setup
- compute the grasp target near the merged object centroid while keeping at least `3 cm` clearance from the selected hand along the height axis
- keep the robot end-effector orientation fixed and move position only
- remember the initial object centroid and place the object back there after the robot receives it
- verify grasp using RTDE force/current signals

## Scope

Target capabilities:

- read RGB and depth from two cameras
- estimate object geometry from each camera and merge object point clouds
- estimate hand pose from each camera independently
- convert all relevant outputs into the robot base frame
- detect hand approach to the object
- detect object lift along the configured height axis
- activate the robot after approach plus lift
- compute a robot grasp position from the merged object point cloud and selected hand pose
- keep robot orientation fixed during the receive motion
- place the object back near the memorized initial centroid

Out of scope for the first version:

- ROS/ROS2 integration
- hand pose fusion across cameras
- dynamic end-effector orientation planning
- advanced motion planning with MoveIt

## Key Task Assumptions

This plan assumes the following task flow:

- the object starts on a support surface
- the human hand approaches the object first
- the human lifts the object by at least `5 cm` along the configured height axis
- after that lift event, the robot begins the receive routine
- the robot grasps the object using a fixed end-effector orientation
- grasp success is checked with RTDE force/current feedback
- after the robot securely receives the object, it returns to the memorized initial object centroid and releases the object there

If any of these assumptions change, update the state machine before implementation begins.

## System Design Summary

Recommended process split:

1. `Dual Sensor Hub`
2. `Object Worker Cam0`
3. `Object Worker Cam1`
4. `Object Point Cloud Merger`
5. `Hand Worker Cam0`
6. `Hand Worker Cam1`
7. `Hand Selector`
8. `Task Manager`
9. `RTDE Controller`

Core rules:

- perception may be slower
- robot control must not block on perception
- object point clouds are merged
- hand poses are not fused
- the selected hand source depends on handedness

Recommended update rates:

- dual-camera capture: `30 Hz`
- hand pose per camera: `15-30 Hz`
- object perception per camera: `5-15 Hz`
- object point cloud merge: `5-15 Hz`
- task manager: `20-30 Hz`
- robot control: `50-125 Hz`

## Proposed Repository Additions

Recommended new modules:

- `system/runner.py`
- `system/shared_state.py`
- `system/dual_sensor_hub.py`
- `system/task_manager.py`
- `robot/rtde_controller.py`
- `robot/robot_commands.py`
- `robot/safety.py`
- `perception/object_worker_cam0.py`
- `perception/object_worker_cam1.py`
- `perception/object_merger.py`
- `perception/hand_worker_cam0.py`
- `perception/hand_worker_cam1.py`
- `perception/hand_selector.py`
- `perception/fusion.py`
- `calibration/extrinsics.py`
- `configs/handover.yaml`

Recommended reuse points from current code:

- `object_pt_extraction/realsense_yoloe_seg_demo.py`
- `object_pt_extraction/pointcloud_utils.py`
- `handpose3d/handpose3d.py`

## Shared Coordinate Design

Required transforms:

- `T_base_cam0`
- `T_base_cam1`

Provided calibration chain in this repository:

- `camera_parameters/c0_to_robot.pckl` stores the cam0-to-robot-base extrinsic used as `T_base_cam0`
- `camera_parameters/rot_trans_c1.dat` stores the cam1-to-cam0 extrinsic used as `T_cam0_cam1`
- derive `T_base_cam1 = T_base_cam0 @ T_cam0_cam1` so cam1 data can also be transformed into robot base

Design rules:

- convert all object point clouds into the robot base frame before merging
- convert the selected hand pose into the robot base frame before grasp computation
- define the upward/downward direction explicitly as a configurable height axis
- do all event detection and robot target generation in base frame only

Recommended convention:

- default `height_axis = y`
- use positive height-axis motion to represent upward lift
- allow the axis to be overridden in configuration if your base frame differs

## Implementation Steps

## Step 1. Freeze the MVP requirements

Define the exact first demo target before editing code.

Minimum viable receive-and-place demo:

- detect one known object class from two cameras
- compute object point clouds from both cameras and merge them
- estimate hand pose independently from both cameras
- use right-hand pose from camera 0 and left-hand pose from camera 1
- detect that a hand approaches the object
- detect that the object lifts by at least `5 cm` along the configured height axis
- start robot action only after that event
- compute a fixed-orientation grasp point using the merged object point cloud and selected hand pose
- verify grasp with RTDE force/current feedback
- move to the memorized initial object centroid
- release the object there

Deliverables:

- one agreed object class for testing
- camera 0 and camera 1 identifiers
- calibration data for both cameras
- one UR5 connection method and IP
- one gripper control interface

Done criteria:

- the team can describe the first demo in one paragraph without ambiguity

## Step 2. Centralize ownership of both cameras

Current prototype scripts likely open a camera inside each demo script. That is acceptable for demos but risky for integration.

Implementation tasks:

1. Create `system/dual_sensor_hub.py`
2. Move both camera initializations into that module
3. Publish only the latest frame bundle for each camera:
   - `rgb`
   - `depth`
   - `intrinsics`
   - `timestamp`
   - `frame_id`
4. Ensure each physical camera is opened only once
5. Expose a synchronized or near-synchronized dual-frame snapshot API

Design rules:

- do not let object and hand code open cameras independently
- do not accumulate deep frame queues
- always overwrite with the latest frame
- each downstream worker reads from the shared dual-camera hub

Suggested API:

```python
class DualSensorHub:
    def start(self): ...
    def get_latest_cam0(self): ...
    def get_latest_cam1(self): ...
    def get_latest_pair(self): ...
    def stop(self): ...
```

Done criteria:

- object and hand modules both consume frames from `DualSensorHub`
- the two cameras are opened only once each

## Step 3. Define shared state contracts

Before integrating modules, define what data each module publishes.

Implementation tasks:

1. Create `system/shared_state.py`
2. Define typed containers or dataclasses for:
   - sensor state
   - per-camera object state
   - merged object state
   - per-camera hand state
   - selected hand state
   - grasp target state
   - task state
   - robot state
3. Add `timestamp`, `frame_id`, `camera_id`, and `valid` to every state where relevant
4. Add RTDE force/current fields to robot state so grasp verification can happen in task logic

Recommended fields:

```python
MergedObjectState:
  centroid_base: np.ndarray shape (3,)
  initial_centroid_base: np.ndarray shape (3,)
  object_lifted: bool
  lift_height_delta_m: float
  height_axis_name: str
  valid: bool

SelectedHandState:
  selected_camera: int | None
  handedness: str | None
  palm_center_base: np.ndarray shape (3,) | None
  palm_normal_base: np.ndarray shape (3,) | None
  valid: bool

GraspTargetState:
  position_base: np.ndarray shape (3,) | None
  fixed_orientation_base: np.ndarray shape (3,) | None
  hand_height_clearance_m: float | None
  valid: bool

RobotState:
  actual_tcp_force_base: np.ndarray shape (3,) | None
  tcp_force_norm_n: float | None
  joint_currents_a: np.ndarray shape (6,) | None
  mean_joint_current_a: float | None
  grasp_verified_force_current: bool
```

Design rules:

- use latest-state semantics
- never let the robot controller depend on a backlog of old frames
- store the memorized initial object centroid explicitly
- keep force/current feedback available to task logic and safety logic

Done criteria:

- all major modules read and write the same structured state definitions

## Step 4. Refactor object perception into per-camera workers

Convert object perception from a standalone visualization/demo script into reusable workers.

Implementation tasks:

1. Create `perception/object_worker_cam0.py`
2. Create `perception/object_worker_cam1.py`
3. Extract the useful parts from `object_pt_extraction/realsense_yoloe_seg_demo.py`
4. Keep the workers focused on perception output, not UI
5. Reuse functions from `object_pt_extraction/pointcloud_utils.py`
6. Output per camera:
   - object centroid in base frame
   - object point cloud in base frame
   - object confidence
   - object validity flag

Recommended internal stages per camera:

1. consume latest RGB/depth
2. run segmentation
3. generate masked point cloud
4. transform point cloud from camera frame to base frame
5. remove outliers and downsample
6. publish per-camera object state

Notes:

- keep visualization optional behind a debug flag
- separate point cloud cleanup parameters from demo defaults
- point clouds should be transformed to base frame before merging

Done criteria:

- each object worker runs headless
- each object worker can be called continuously without opening new windows

## Step 5. Merge object point clouds

The robot should use one merged object point cloud built from both cameras.

Implementation tasks:

1. Create `perception/object_merger.py`
2. Read both per-camera object states
3. Merge point clouds in base frame
4. clean and downsample the merged cloud
5. compute a merged centroid
6. memorize the initial centroid when the object is first stably observed on the support surface
7. detect object lift by comparing current centroid position against the memorized centroid along the configured height axis

Recommended lift detection logic:

- object is considered lifted when the merged centroid moves upward by at least `0.05 m` along the configured height axis
- use `height_axis = y` by default
- require stable detection over several frames
- only accept lift after a valid hand-approach event

Recommended outputs:

- `merged_points_base`
- `centroid_base`
- `initial_centroid_base`
- `object_lifted`
- `lift_height_delta_m`
- `height_axis_name`
- `valid`

Done criteria:

- one stable merged object point cloud is available in base frame
- initial object centroid is memorized before the lift event
- the lift event fires only after a `5 cm` upward displacement on the configured height axis

## Step 6. Refactor hand pose into per-camera workers

Convert hand pose estimation into reusable runtime workers.

Implementation tasks:

1. Create `perception/hand_worker_cam0.py`
2. Create `perception/hand_worker_cam1.py`
3. Move reusable logic out of `handpose3d/handpose3d.py`
4. Publish handover-relevant features, not just raw landmarks
5. transform each hand pose into base frame using its camera transform

Minimum useful outputs per camera:

- `hand_detected`
- `handedness`
- `palm_center_base`
- `palm_normal_base`
- `wrist_base`
- `hand_velocity_base`

Recommended internal stages per camera:

1. consume latest RGB/depth
2. run hand pose estimation
3. estimate handedness
4. estimate palm center
5. estimate palm orientation
6. transform pose into base frame
7. smooth over a short temporal window
8. publish per-camera hand state

Important note:

Do not fuse hand poses from the two cameras. Each camera publishes independently.

Done criteria:

- both hand workers publish stable hand states in base frame
- no direct robot logic is embedded in the hand modules

## Step 7. Implement handedness-based hand selection

The runtime should select one hand source instead of fusing them.

Selection rule:

- if the detected hand is the right hand, use camera 0 hand pose
- if the detected hand is the left hand, use camera 1 hand pose

Implementation tasks:

1. Create `perception/hand_selector.py`
2. Read both per-camera hand states
3. If camera 0 reports a valid right hand, select camera 0
4. If camera 1 reports a valid left hand, select camera 1
5. If neither condition is satisfied, publish an invalid selected-hand state
6. Add a short hysteresis rule so the selected source does not flicker frame to frame

Recommended behavior for ambiguous cases:

- if both valid conditions hold, prefer the last selected source unless confidence drops strongly
- if a camera detects the wrong handedness for its assigned role, ignore it

Done criteria:

- exactly one selected hand state is available for task logic
- no stereo fusion is performed on the hand pose

## Step 8. Add temporal filtering and event detection

Raw perception will be too noisy for safe robot behavior.

Implementation tasks:

1. Add filtering utilities in `perception/fusion.py` or a dedicated utility module
2. Smooth:
   - merged object centroid
   - selected hand palm center
   - selected hand palm normal
3. Add validity timeout logic
4. Add stable-frame rules before event detection
5. Detect hand approach to object
6. Detect object lift after the hand approaches

Recommended event logic:

- `hand_approach=True` when hand-object distance falls below a threshold and stays stable
- `object_lifted=True` when object centroid rises by at least `0.05 m` along the configured height axis after `hand_approach=True`
- robot activation condition is `hand_approach=True` and `object_lifted=True`

Recommended policies:

- mark data invalid if older than `0.2-0.5 s`
- require `3-5` stable frames before robot activation
- reject large frame-to-frame jumps

Done criteria:

- perception outputs are stable enough to trigger robot activation reliably

## Step 9. Implement camera-to-robot calibration

This is mandatory. Without it, object and hand positions cannot be used for robot motion.

Implementation tasks:

1. Create `calibration/extrinsics.py`
2. Store `T_base_cam0` and `T_base_cam1`
3. Add utilities to transform points and poses from each camera frame to UR5 base frame
4. Decide calibration workflow:
   - manual measured transform
   - checkerboard or AprilTag based calibration
   - hand-eye calibration
5. Load camera intrinsics and extrinsics from configuration or existing calibration files
6. Compose the provided transforms so `T_base_cam1 = T_base_cam0 @ T_cam0_cam1`

Recommended first version:

- store camera transforms in `configs/handover.yaml`
- provide a simple verification script to overlay transformed test points from both cameras in base frame

Core APIs:

```python
def transform_point_cam0_to_base(p_cam, T_base_cam0): ...
def transform_point_cam1_to_base(p_cam, T_base_cam1): ...
def transform_pose_cam0_to_base(pose_cam, T_base_cam0): ...
def transform_pose_cam1_to_base(pose_cam, T_base_cam1): ...
```

Done criteria:

- merged object point cloud from both cameras aligns in base frame
- selected hand pose is available in base frame

## Step 10. Compute the robot grasp point

The grasp point should be computed from the merged object point cloud and the selected hand pose.

Required constraints:

- grasp point should be close to the object centroid
- grasp point should be sufficiently far from the selected hand along the configured height axis
- minimum height-axis separation from the hand should be at least `3 cm`
- robot end-effector orientation stays fixed
- robot motion changes position only

Implementation tasks:

1. Create grasp-point selection logic in `perception/fusion.py` or `system/task_manager.py`
2. Generate grasp candidates from the merged object point cloud
3. score candidates by distance to the merged object centroid
4. reject candidates whose height-axis distance to the selected hand is less than `0.03 m`
5. choose the highest-scoring valid candidate
6. output only target position plus one fixed end-effector orientation loaded from config

Recommended first version:

- use the configured height axis, with `y` as the default
- prefer candidates nearest to the merged centroid after applying the `3 cm` hand-clearance rule
- if no candidate passes the clearance rule, do not command grasp

Recommended output:

```python
GraspTargetState:
  position_base: np.ndarray shape (3,)
  fixed_orientation_base: np.ndarray shape (3,) | np.ndarray shape (4,)
  hand_height_clearance_m: float | None
  valid: bool
```

Done criteria:

- the system can compute a valid grasp target from merged object geometry and selected hand state

## Step 11. Build the task manager state machine

The task manager should turn perception into robot intentions, not direct motor commands.

Implementation tasks:

1. Create `system/task_manager.py`
2. Implement the state machine
3. Read merged object state, selected hand state, grasp target state, and robot state
4. Output high-level robot commands and target poses

Recommended states:

- `IDLE`
- `OBSERVE_INITIAL_OBJECT`
- `WAIT_FOR_HAND_APPROACH`
- `WAIT_FOR_OBJECT_LIFT`
- `ACTIVATE_ROBOT`
- `APPROACH_GRASP_POINT`
- `RECEIVE_OBJECT`
- `VERIFY_GRASP`
- `MOVE_TO_INITIAL_PLACE`
- `RELEASE`
- `RETREAT`
- `FAIL_SAFE`

Recommended transition examples:

- `IDLE -> OBSERVE_INITIAL_OBJECT` when merged object is valid
- `OBSERVE_INITIAL_OBJECT -> WAIT_FOR_HAND_APPROACH` after initial centroid is memorized
- `WAIT_FOR_HAND_APPROACH -> WAIT_FOR_OBJECT_LIFT` when selected hand approaches object
- `WAIT_FOR_OBJECT_LIFT -> ACTIVATE_ROBOT` when object lift is detected
- `ACTIVATE_ROBOT -> APPROACH_GRASP_POINT` when a valid grasp target exists
- `APPROACH_GRASP_POINT -> RECEIVE_OBJECT` when the robot reaches the grasp point
- `RECEIVE_OBJECT -> VERIFY_GRASP` when the gripper closes
- `VERIFY_GRASP -> MOVE_TO_INITIAL_PLACE` only when RTDE force/current verification succeeds
- `MOVE_TO_INITIAL_PLACE -> RELEASE` when the robot reaches the memorized initial centroid position
- `RELEASE -> RETREAT` after gripper opens
- any state -> `FAIL_SAFE` on timeout or safety violation

Done criteria:

- all robot motion decisions come from the task manager
- the memorized initial centroid is used as the placement target
- grasp verification is based on RTDE force/current
- perception modules do not command the robot directly

## Step 12. Implement the RTDE robot controller

The RTDE controller should translate task commands into robot actions.

Implementation tasks:

1. Create `robot/rtde_controller.py`
2. Connect to the UR5 over RTDE
3. Read current robot state continuously
4. Expose a small command interface:
   - move to fixed-orientation Cartesian position
   - servo to fixed-orientation Cartesian position
   - stop safely
   - gripper open or close
5. Publish force/current feedback into robot state
6. Add watchdog behavior if task data becomes stale

Design rules:

- the controller should not parse images
- the controller should not decide task transitions
- the controller should be able to hold last safe pose if perception drops out
- orientation should remain fixed unless the configuration explicitly changes it

Recommended command abstraction:

```python
RobotCommand:
  type: str
  target_position_base: np.ndarray | None
  fixed_orientation_base: np.ndarray | None
  gripper_action: str | None
```

Recommended RTDE feedback used for grasp verification:

- TCP force vector or force norm
- joint current vector or mean current
- a configurable rule that can require one or both signals

Done criteria:

- robot can be commanded from synthetic fixed-orientation targets before any perception is connected
- RTDE feedback required for grasp verification is available in robot state

## Step 13. Add safety guards

Safety must be explicit in code, not assumed.

Implementation tasks:

1. Create `robot/safety.py`
2. Add checks for:
   - workspace bounds
   - max translation step
   - fixed orientation validity
   - max robot speed
   - stale perception timeout
   - invalid calibration
3. Prevent receive or release if:
   - selected hand is lost
   - merged object confidence is poor
   - target position leaves workspace
   - no valid grasp target exists
   - RTDE grasp verification has not succeeded when required

Recommended fail-safe actions:

- hold current pose
- retreat to a safe standby pose
- retreat to home pose
- full motion stop for severe faults

Done criteria:

- every outgoing motion command passes through safety validation

## Step 14. Build the system runner

Create one entry point that starts all processes cleanly.

Implementation tasks:

1. Create `system/runner.py`
2. Initialize shared state and configuration
3. Start:
   - dual sensor hub
   - object worker cam0
   - object worker cam1
   - object point cloud merger
   - hand worker cam0
   - hand worker cam1
   - hand selector
   - task manager
   - RTDE controller
4. Add shutdown handling and heartbeat logging

Recommended config sections in `configs/handover.yaml`:

- camera 0 settings
- camera 1 settings
- segmentation thresholds
- point cloud parameters
- hand smoothing parameters
- handedness selection policy
- calibration matrices or calibration file paths
- height-axis definition
- hand-approach and lift thresholds
- grasp-point selection thresholds
- fixed end-effector orientation
- RTDE grasp-verification thresholds
- robot IP and control gains
- safety bounds

Done criteria:

- one command launches the full stack
- one command shuts it down cleanly

## Step 15. Test bottom-up before full handover

Do not jump directly to full human receive-and-place behavior.

Testing order:

1. dual sensor only
2. object worker cam0 only
3. object worker cam1 only
4. merged object point cloud verification
5. hand worker cam0 only
6. hand worker cam1 only
7. handedness-based hand selection
8. calibration verification for both cameras
9. lift-event detection without robot motion
10. RTDE controller with fixed synthetic targets
11. RTDE force/current readback verification
12. robot move to computed grasp point
13. robot move back to memorized initial centroid
14. full receive and release logic

Recommended test gates:

- Gate A: object point clouds from both cameras align in base frame
- Gate B: merged object centroid is stable
- Gate C: camera 0 right-hand selection works
- Gate D: camera 1 left-hand selection works
- Gate E: robot activation occurs only after hand approach plus at least `5 cm` object lift on the configured height axis
- Gate F: grasp point respects centroid proximity and `3 cm` height-axis clearance from hand
- Gate G: RTDE force/current feedback is usable for grasp verification
- Gate H: robot returns to the initial centroid and releases correctly

Done criteria:

- each gate passes before moving to the next

## Step 16. Add logging and replay support

Debugging integrated perception-control systems is difficult without logs.

Implementation tasks:

1. log timestamps for all module outputs
2. log hand selection decisions
3. log state machine transitions
4. log robot commands and robot feedback
5. log grasp-verification inputs from RTDE force/current
6. optionally save synchronized dual-camera RGB/depth snapshots during test runs

Recommended fields to log:

- frame ids for both cameras
- timestamps
- merged object centroid
- initial object centroid
- selected hand source and handedness
- selected hand palm center
- grasp target position
- lift height delta on the configured height axis
- task state
- robot actual pose
- TCP force norm
- mean joint current
- safety flags

Done criteria:

- any failure can be traced after the run

## Step 17. Tune thresholds for receive-and-place

Once the full pipeline runs, tune the runtime policy.

Tuneable values:

- hand-object approach threshold
- object-lift threshold, starting at `0.05 m`
- hand stability duration
- object confidence threshold
- hand selection hysteresis
- minimum hand-height clearance for grasp target
- RTDE force threshold
- RTDE current threshold
- max robot approach speed
- placement tolerance around initial centroid

Recommended tuning strategy:

1. tune merged object stability first
2. tune hand selection stability second
3. tune hand-approach and lift detection third
4. tune grasp-point selection fourth
5. tune RTDE grasp verification fifth
6. tune receive and release last

Done criteria:

- robot activation and receive behavior feel deliberate rather than reactive

## Recommended First Coding Order

If implementation starts immediately, use this exact order:

1. create `configs/handover.yaml`
2. create `system/shared_state.py`
3. create `system/dual_sensor_hub.py`
4. create `perception/object_worker_cam0.py`
5. create `perception/object_worker_cam1.py`
6. create `perception/object_merger.py`
7. create `perception/hand_worker_cam0.py`
8. create `perception/hand_worker_cam1.py`
9. create `perception/hand_selector.py`
10. create `calibration/extrinsics.py`
11. create grasp-point selection logic
12. create `system/task_manager.py`
13. create `robot/rtde_controller.py`
14. create `robot/safety.py`
15. create `system/runner.py`

## Acceptance Checklist

The first integrated version is acceptable when:

- both cameras are opened once and shared across all modules
- object point clouds from both cameras are merged in base frame
- hand poses are not fused and are selected by handedness rule
- robot activates only after hand approach plus at least `5 cm` object lift on the configured height axis
- grasp target is chosen near the merged centroid while respecting at least `3 cm` height-axis clearance from the hand
- RTDE controller runs independently of perception latency
- grasp success is verified using RTDE force/current
- after receive, the robot returns to the memorized initial object centroid and releases there
- stale or invalid perception does not cause unsafe motion
- a simple dual-camera receive-and-place demo runs end-to-end on one PC

## Remaining Open Questions

These points are still worth finalizing before full integration:

- what hand-object distance should define a valid approach event in your setup
- whether placement should use exactly the memorized centroid or a small fixed release-height offset above it
- what precise RTDE force/current thresholds best indicate a secure grasp for your gripper and payload

## Notes

- Keep visualization optional and disabled by default in the integrated system.
- Prefer `multiprocessing` over one giant sequential script.
- Use latest-state sharing, not frame backlogs.
- Treat dual-camera calibration as a first-class deliverable, not cleanup work.
- Keep the robot controller simple and conservative in the first version.
