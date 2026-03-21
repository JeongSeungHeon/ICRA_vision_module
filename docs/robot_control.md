# Robot Control Implementation Plan

## Goal

Build a single-PC handover system that runs:

- object segmentation and point cloud extraction
- hand 3D pose estimation
- UR5 robot control over RTDE

without ROS, while keeping perception and robot control decoupled enough for stable runtime behavior.

This document is a step-by-step implementation plan for turning the current prototype scripts into a runnable integrated system.

## Scope

Target capabilities:

- read RGB and depth from one RealSense camera
- estimate object geometry from segmentation and point cloud
- estimate hand pose and handover-relevant hand features
- fuse both signals in one robot base coordinate frame
- run a task state machine for grasp, present, approach, release, and retreat
- command UR5 through RTDE safely

Out of scope for the first version:

- ROS/ROS2 integration
- multi-camera fusion
- force-torque based human intent estimation
- advanced motion planning with MoveIt

## System Design Summary

Recommended process split:

1. `Sensor Hub`
2. `Object Worker`
3. `Hand Worker`
4. `Task Manager`
5. `RTDE Controller`

Core rule:

- perception may be slower
- robot control must not block on perception

Recommended update rates:

- sensor capture: `30 Hz`
- hand pose: `15-30 Hz`
- object perception: `5-15 Hz`
- task manager: `20-30 Hz`
- robot control: `50-125 Hz`

## Proposed Repository Additions

Recommended new modules:

- `system/runner.py`
- `system/shared_state.py`
- `system/sensor_hub.py`
- `system/task_manager.py`
- `robot/rtde_controller.py`
- `robot/robot_commands.py`
- `robot/safety.py`
- `perception/object_worker.py`
- `perception/hand_worker.py`
- `perception/fusion.py`
- `calibration/extrinsics.py`
- `configs/handover.yaml`

Recommended reuse points from current code:

- `object_pt_extraction/realsense_yoloe_seg_demo.py`
- `object_pt_extraction/pointcloud_utils.py`
- `handpose3d/handpose3d.py`

## Implementation Steps

## Step 1. Freeze the MVP requirements

Define the exact first demo target before editing code.

Minimum viable handover demo:

- detect one known object class
- compute object point cloud and object center
- estimate one hand pose
- transform object and hand to robot base frame
- move robot to present pose
- move toward hand when hand is stable and close enough
- open gripper when release conditions are met

Deliverables:

- one agreed object class for testing
- one RealSense camera configuration
- one UR5 connection method and IP
- one chosen gripper release API

Done criteria:

- the team can describe the first demo in one paragraph without ambiguity

## Step 2. Centralize camera ownership

Current prototype scripts likely open RealSense inside each demo script. That is acceptable for demos but risky for integration.

Implementation tasks:

1. Create `system/sensor_hub.py`
2. Move RealSense initialization into that module
3. Publish only the latest frame bundle:
   - `rgb`
   - `depth`
   - `intrinsics`
   - `timestamp`
   - `frame_id`
4. Ensure only one process opens `pyrealsense2.pipeline`

Design rules:

- do not let object and hand code open the camera independently
- do not accumulate deep frame queues
- always overwrite with the latest frame

Suggested API:

```python
class SensorHub:
    def start(self): ...
    def get_latest(self): ...
    def stop(self): ...
```

Done criteria:

- object and hand modules both consume frames from `SensorHub`
- the camera is opened only once

## Step 3. Define shared state contracts

Before integrating modules, define what data each module publishes.

Implementation tasks:

1. Create `system/shared_state.py`
2. Define typed containers or dataclasses for:
   - `SensorState`
   - `ObjectState`
   - `HandState`
   - `TaskState`
   - `RobotState`
3. Add `timestamp`, `frame_id`, and `valid` to every state

Recommended fields:

```python
ObjectState:
  object_detected: bool
  label: str
  confidence: float
  centroid_cam: np.ndarray shape (3,)
  grasp_pose_cam: np.ndarray shape (6,)
  valid: bool

HandState:
  hand_detected: bool
  palm_center_cam: np.ndarray shape (3,)
  palm_normal_cam: np.ndarray shape (3,)
  wrist_cam: np.ndarray shape (3,)
  hand_velocity_cam: np.ndarray shape (3,)
  intent_score: float
  valid: bool

TaskState:
  mode: str
  target_pose_base: np.ndarray shape (6,)
  gripper_cmd: str
  safety_ok: bool
```

Design rules:

- use latest-state semantics
- never let the robot controller depend on a backlog of old frames

Done criteria:

- all major modules read and write the same structured state definitions

## Step 4. Refactor object perception into a worker

Convert object perception from a standalone visualization/demo script into a reusable worker.

Implementation tasks:

1. Create `perception/object_worker.py`
2. Extract the useful parts from `object_pt_extraction/realsense_yoloe_seg_demo.py`
3. Keep the worker focused on perception output, not UI
4. Reuse functions from `object_pt_extraction/pointcloud_utils.py`
5. Output:
   - object centroid in camera frame
   - grasp candidate in camera frame
   - object confidence
   - object validity flag

Recommended internal stages:

1. consume latest RGB/depth
2. run segmentation
3. generate masked point cloud
4. remove outliers and downsample
5. compute centroid and grasp-relevant geometry
6. publish `ObjectState`

Notes:

- keep visualization optional behind a debug flag
- separate point cloud cleanup parameters from demo defaults
- consider separate `voxel_size` values for fusion and grasp geometry

Done criteria:

- worker runs headless
- worker can be called continuously without opening new windows

## Step 5. Refactor hand pose into a worker

Convert hand pose estimation into a reusable runtime worker.

Implementation tasks:

1. Create `perception/hand_worker.py`
2. Move reusable logic out of `handpose3d/handpose3d.py`
3. Publish handover-relevant features, not just raw landmarks

Minimum useful outputs:

- `palm_center_cam`
- `palm_normal_cam`
- `wrist_cam`
- `hand_velocity_cam`
- `intent_score`
- `hand_detected`

Recommended internal stages:

1. consume latest RGB/depth
2. run hand pose estimation
3. estimate palm center
4. estimate palm orientation
5. smooth over a short temporal window
6. publish `HandState`

Important note:

If the current hand pose module outputs only relative 3D skeleton coordinates, add a depth-based or camera-based conversion path so the hand can be localized in the same coordinate system as the object.

Done criteria:

- worker outputs stable handover features
- no direct robot logic is embedded in the hand module

## Step 6. Add temporal filtering and confidence handling

Raw perception will be too noisy for safe robot behavior.

Implementation tasks:

1. Add filtering utilities in `perception/fusion.py` or a dedicated utility module
2. Smooth:
   - object centroid
   - hand palm center
   - hand palm normal
3. Add validity timeout logic
4. Add minimum stable-frame rules before state transitions

Recommended policies:

- mark data invalid if older than `0.2-0.5 s`
- require `3-5` stable frames before approach
- reject large frame-to-frame jumps

Done criteria:

- perception outputs no longer jitter enough to trigger false robot actions

## Step 7. Implement camera-to-robot calibration

This is mandatory. Without it, object and hand positions cannot be used for robot motion.

Implementation tasks:

1. Create `calibration/extrinsics.py`
2. Store `T_base_camera`
3. Add utilities to transform points and poses from camera frame to UR5 base frame
4. Decide calibration workflow:
   - manual measured transform
   - checkerboard/AprilTag based calibration
   - hand-eye calibration

Recommended first version:

- store `T_base_camera` in `configs/handover.yaml`
- provide a simple verification script to overlay transformed test points

Core APIs:

```python
def transform_point_cam_to_base(p_cam, T_base_camera): ...
def transform_pose_cam_to_base(pose_cam, T_base_camera): ...
```

Done criteria:

- object centroid and hand palm center can both be expressed in `base` frame

## Step 8. Build the task manager state machine

The task manager should turn perception into robot intentions, not direct motor commands.

Implementation tasks:

1. Create `system/task_manager.py`
2. Implement the state machine
3. Read filtered `ObjectState`, `HandState`, and `RobotState`
4. Output high-level robot commands and target poses

Recommended states:

- `IDLE`
- `ACQUIRE_OBJECT`
- `GRASPING`
- `LIFT_AND_PRESENT`
- `TRACK_HAND`
- `APPROACH_HANDOVER`
- `RELEASE_CHECK`
- `RELEASE`
- `RETREAT`
- `FAIL_SAFE`

Recommended transition examples:

- `IDLE -> ACQUIRE_OBJECT` when object is valid
- `LIFT_AND_PRESENT -> TRACK_HAND` when grasp is complete
- `TRACK_HAND -> APPROACH_HANDOVER` when hand is valid and close enough
- `APPROACH_HANDOVER -> RELEASE_CHECK` when pose and distance stay stable
- `RELEASE_CHECK -> RELEASE` only after time and confidence checks pass
- any state -> `FAIL_SAFE` on timeout or safety violation

Done criteria:

- all robot motion decisions come from the task manager
- perception modules do not command the robot directly

## Step 9. Implement the RTDE robot controller

The RTDE controller should translate task commands into robot actions.

Implementation tasks:

1. Create `robot/rtde_controller.py`
2. Connect to the UR5 over RTDE
3. Read current robot state continuously
4. Expose a small command interface:
   - move to joint pose
   - move to Cartesian pose
   - servo to pose
   - stop safely
   - gripper open/close
5. Add watchdog behavior if task data becomes stale

Design rules:

- the controller should not parse images
- the controller should not decide task transitions
- the controller should be able to hold last safe pose if perception drops out

Recommended command abstraction:

```python
RobotCommand:
  type: str
  target_pose_base: np.ndarray | None
  target_joints: np.ndarray | None
  gripper_action: str | None
```

Done criteria:

- robot can be commanded from synthetic targets before any perception is connected

## Step 10. Add safety guards

Safety must be explicit in code, not assumed.

Implementation tasks:

1. Create `robot/safety.py`
2. Add checks for:
   - workspace bounds
   - max translation step
   - max orientation step
   - max robot speed
   - stale perception timeout
   - invalid calibration
3. Prevent release if:
   - hand is lost
   - object confidence is poor
   - target pose leaves workspace

Recommended fail-safe actions:

- hold current pose
- retreat to present pose
- retreat to home pose
- full motion stop for severe faults

Done criteria:

- every outgoing motion command passes through safety validation

## Step 11. Build the system runner

Create one entry point that starts all processes cleanly.

Implementation tasks:

1. Create `system/runner.py`
2. Initialize shared state and configuration
3. Start:
   - sensor hub
   - object worker
   - hand worker
   - task manager
   - RTDE controller
4. Add shutdown handling and heartbeat logging

Recommended config sections in `configs/handover.yaml`:

- camera settings
- segmentation thresholds
- point cloud parameters
- hand smoothing parameters
- calibration matrix
- task thresholds
- robot IP and control gains
- safety bounds

Done criteria:

- one command launches the full stack
- one command shuts it down cleanly

## Step 12. Test bottom-up before full handover

Do not jump directly to human handover.

Testing order:

1. sensor only
2. object worker only
3. hand worker only
4. calibration verification
5. RTDE controller with fixed synthetic poses
6. object-driven present pose
7. hand tracking without release
8. full release logic

Recommended test gates:

- Gate A: object centroid is stable in camera frame
- Gate B: hand palm center is stable in camera frame
- Gate C: both transform correctly into base frame
- Gate D: robot moves safely to a static target
- Gate E: robot tracks a slowly moving hand target in simulation-like conditions
- Gate F: release occurs only under strict conditions

Done criteria:

- each gate passes before moving to the next

## Step 13. Add logging and replay support

Debugging integrated perception-control systems is difficult without logs.

Implementation tasks:

1. log timestamps for all module outputs
2. log state machine transitions
3. log robot commands and robot feedback
4. optionally save synchronized RGB/depth snapshots during test runs

Recommended fields to log:

- frame id
- timestamps
- object centroid
- hand palm center
- task state
- target pose
- robot actual pose
- safety flags

Done criteria:

- any failure can be traced after the run

## Step 14. Tune thresholds for handover

Once the full pipeline runs, tune the handover policy.

Tuneable values:

- hand proximity threshold
- hand stability duration
- palm orientation threshold
- object confidence threshold
- release dwell time
- max approach speed
- retreat trigger thresholds

Recommended tuning strategy:

1. tune object stability first
2. tune hand stability second
3. tune approach behavior third
4. tune release last

Done criteria:

- release feels deliberate rather than reactive

## Step 15. Prepare for future upgrades

After the first non-ROS version is stable, consider staged upgrades.

Possible next steps:

- ROS2 migration for process supervision and message transport
- multi-camera fusion for reduced occlusion
- grasp quality scoring from point cloud geometry
- force sensing for safer release confirmation
- predictive hand motion modeling

## Recommended First Coding Order

If implementation starts immediately, use this exact order:

1. create `configs/handover.yaml`
2. create `system/shared_state.py`
3. create `system/sensor_hub.py`
4. create `perception/object_worker.py`
5. create `perception/hand_worker.py`
6. create `calibration/extrinsics.py`
7. create `system/task_manager.py`
8. create `robot/rtde_controller.py`
9. create `robot/safety.py`
10. create `system/runner.py`

## Acceptance Checklist

The first integrated version is acceptable when:

- one RealSense feed is shared by all perception modules
- object and hand outputs are available in the same base frame
- task manager drives robot behavior through explicit states
- RTDE controller runs independently of perception latency
- stale or invalid perception does not cause unsafe motion
- a simple handover demo runs end-to-end on one PC

## Notes

- Keep visualization optional and disabled by default in the integrated system.
- Prefer `multiprocessing` over one giant sequential script.
- Use latest-state sharing, not frame backlogs.
- Treat calibration as a first-class deliverable, not cleanup work.
- Keep the robot controller simple and conservative in the first version.
