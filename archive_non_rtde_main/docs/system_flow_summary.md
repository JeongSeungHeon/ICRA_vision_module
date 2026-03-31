# Vision-to-Robot Control Flow Summary

## Overview

This document summarizes the current end-to-end code flow from dual-camera vision input to UR RTDE robot control.
The runtime entrypoint is [runner.py](/home/ur5/ICRA_vision_module/system/runner.py).

At a high level, one loop does:

1. Read synchronized `cam0` / `cam1` frames from RealSense.
2. Run object perception on both cameras.
3. Run hand perception on both cameras.
4. Select one hand according to the handedness rule.
5. Merge object observations into one base-frame object state.
6. Fuse object and hand signals into hand-approach / lift / activation events.
7. Compute a grasp target in robot base coordinates.
8. Run the task state machine.
9. Validate the command with safety checks.
10. Send the sanitized command to the UR robot through RTDE.

## Main Runtime

The main single-process runtime is [runner.py](/home/ur5/ICRA_vision_module/system/runner.py).

`HandoverSystemRunner.step()` runs the modules in this order:

1. `DualSensorHub.read_next_pair()`
2. `ObjectWorkerCam0.process_frame()`
3. `ObjectWorkerCam1.process_frame()`
4. `HandWorkerCam0.process_frame()`
5. `HandWorkerCam1.process_frame()`
6. `HandSelector.process_states()`
7. `ObjectMerger.process_states()`
8. `PerceptionFusion.process_states()`
9. `GraspTargetPlanner.process_states()`
10. `TaskManager.process_states()`
11. `SafetyValidator.validate()`
12. `RtdeController.step()`

All intermediate outputs are stored in [shared_state.py](/home/ur5/ICRA_vision_module/system/shared_state.py) inside `SharedStateBundle`.

## Stage 1: Sensor Input

The dual-camera input layer is [dual_sensor_hub.py](/home/ur5/ICRA_vision_module/system/dual_sensor_hub.py).

Responsibilities:

- Own both RealSense devices through one interface
- Read a near-synchronized `cam0` / `cam1` pair
- Expose `FrameBundle` for each camera
- Expose per-camera `SensorState`

Key outputs:

- `sensor_cam0`
- `sensor_cam1`
- `snapshot.cam0`
- `snapshot.cam1`

Each `FrameBundle` carries:

- `color_image`
- `depth_image_m`
- `intrinsics`
- `timestamp_ms`
- `serial`

## Stage 2: Per-Camera Object Perception

The object workers are in [object_worker.py](/home/ur5/ICRA_vision_module/perception/object_worker.py).

One worker runs per camera:

- `ObjectWorkerCam0`
- `ObjectWorkerCam1`

Flow inside each object worker:

1. Run YOLO segmentation through [segmentation_engine.py](/home/ur5/ICRA_vision_module/object_pt_extraction/segmentation_engine.py)
2. Select the target instance(s)
3. Lift masked depth pixels into camera-frame 3D points
4. Transform those 3D points into robot base coordinates
5. Downsample and remove outliers
6. Compute centroid and object summary
7. Emit `ObjectState`

Important detail:

- The object worker does not output image-space points only.
- It converts the selected object into `points_base` and `centroid_base`.

Key output per camera:

- `object_cam0`
- `object_cam1`

Each `ObjectState` mainly contains:

- `object_detected`
- `label`
- `confidence`
- `centroid_base`
- `point_count`
- `points_base`
- `valid`

## Stage 3: Per-Camera Hand Perception

The hand workers are in [hand_worker.py](/home/ur5/ICRA_vision_module/perception/hand_worker.py).

One worker runs per camera:

- `HandWorkerCam0`
- `HandWorkerCam1`

Flow inside each hand worker:

1. Detect a 2D hand with MediaPipe
2. Lift 2D landmarks into camera-frame 3D using depth
3. Transform 3D landmarks into robot base coordinates
4. Estimate palm center, palm normal, wrist position, and velocity
5. Emit `HandState`

Key output per camera:

- `hand_cam0`
- `hand_cam1`

Each `HandState` mainly contains:

- `hand_detected`
- `handedness`
- `palm_center_base`
- `palm_normal_base`
- `wrist_base`
- `hand_velocity_base`
- `confidence`
- `valid`

## Stage 4: Hand Selection

The hand selection layer is [hand_selector.py](/home/ur5/ICRA_vision_module/perception/hand_selector.py).

Current rule:

- right hand -> prefer `cam0`
- left hand -> prefer `cam1`

It also applies hysteresis and dropout holding so the selected hand does not switch too aggressively.

Key output:

- `selected_hand`

`SelectedHandState` is the hand signal used by all later stages.

## Stage 5: Object Merge

The merged object layer is [object_merger.py](/home/ur5/ICRA_vision_module/perception/object_merger.py).

Flow:

1. Collect valid `points_base` from `object_cam0` and `object_cam1`
2. Merge them in robot base coordinates
3. Downsample and clean the merged cloud
4. Compute merged centroid
5. Memorize the initial centroid when stable
6. Detect object lift relative to the memorized centroid

Key output:

- `merged_object`

`MergedObjectState` mainly contains:

- `centroid_base`
- `initial_centroid_base`
- `merged_points_base`
- `merged_point_count`
- `object_lifted`
- `lift_height_delta_m`
- `valid`

## Stage 6: Perception Fusion

The event fusion layer is [fusion.py](/home/ur5/ICRA_vision_module/perception/fusion.py).

Flow:

1. Read `merged_object`
2. Read `selected_hand`
3. Smooth object and hand positions
4. Reject stale or implausible jumps
5. Compute hand-object distance
6. Detect hand approach
7. Re-check object lift
8. Decide whether robot activation conditions are satisfied

Key output:

- `fusion`

`FusionState` mainly contains:

- `filtered_object_centroid_base`
- `filtered_hand_center_base`
- `hand_object_distance_m`
- `hand_approach_detected`
- `hand_approach_latched`
- `object_lifted`
- `lift_height_delta_m`
- `robot_activation_ready`

## Stage 7: Grasp Target Planning

The grasp planner is [grasp_target.py](/home/ur5/ICRA_vision_module/perception/grasp_target.py).

Flow:

1. Start from merged object points in robot base coordinates
2. Use the selected hand pose as a clearance constraint
3. Search for a point near the object centroid
4. Filter out points that violate the minimum height-axis clearance from the hand
5. Keep a fixed end-effector orientation
6. Output only a target position in robot base

Key output:

- `grasp_target`

`GraspTargetState` mainly contains:

- `target_position_base`
- `fixed_orientation_base`
- `hand_height_clearance_m`
- `centroid_distance_m`
- `valid`

## Stage 8: Task Logic

The high-level state machine is [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py).

Its job is to convert perception states into robot intents.

Major states:

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

Important behavior:

- The first stable object centroid is memorized as the later placement target.
- The system waits for hand approach and object lift before robot activation.
- When grasp is verified, the robot goes back to the memorized initial object location and releases there.

Key outputs:

- `task`
- `robot_command`

`TaskState` describes the high-level mode.
`RobotCommandState` describes what the robot should do next.

## Stage 9: Safety Validation

The safety layer is [safety.py](/home/ur5/ICRA_vision_module/robot/safety.py).

Flow:

1. Inspect current task mode
2. Inspect requested robot command
3. Inspect perception freshness and validity
4. Check workspace bounds
5. Check target plausibility
6. Check that grasp/release conditions are allowed
7. Either pass the command or replace it with a safe stop / hold command

Key output:

- `SafetyResult`

Main fields:

- `safe`
- `reason`
- `sanitized_command`

The runner always sends the sanitized command to the RTDE controller, not the raw task-manager command.

## Stage 10: RTDE Robot Control

The UR control layer is [rtde_controller.py](/home/ur5/ICRA_vision_module/robot/rtde_controller.py).

Flow:

1. Connect to UR RTDE
2. Read the sanitized `RobotCommandState`
3. Apply frame mapping between project coordinates and RTDE coordinates
4. Execute move / servo / stop command
5. Apply gripper outputs when requested
6. Read back TCP pose, speed, force, and joint currents
7. Evaluate grasp verification from force/current thresholds
8. Emit updated `RobotState`

Key output:

- `robot`

`RobotState` mainly contains:

- `actual_tcp_pose_base`
- `actual_tcp_speed`
- `actual_tcp_force_base`
- `tcp_force_norm_n`
- `joint_currents_a`
- `mean_joint_current_a`
- `gripper_state`
- `grasp_verified_force_current`
- `robot_mode`

## Coordinate Flow

The transform loader is [extrinsics.py](/home/ur5/ICRA_vision_module/calibration/extrinsics.py).

Current transform chain:

1. `cam0` points -> `robot base` with `c0_to_robot.pckl`
2. `cam1` points -> `cam0` with `rot_trans_c1.dat`
3. `cam1` points -> `robot base` through the composed chain

In other words:

- object and hand observations are first reconstructed in each camera frame
- then transformed into robot base coordinates
- then merged / fused / planned in robot base coordinates

The control stack after planning also works in robot base coordinates.

## Shared State Structure

[shared_state.py](/home/ur5/ICRA_vision_module/system/shared_state.py) collects the full runtime snapshot.

Important fields in `SharedStateBundle`:

- `sensor_cam0`
- `sensor_cam1`
- `object_cam0`
- `object_cam1`
- `hand_cam0`
- `hand_cam1`
- `selected_hand`
- `merged_object`
- `fusion`
- `grasp_target`
- `task`
- `robot_command`
- `robot`

This bundle is the easiest place to inspect when debugging stage-by-stage outputs.

## Practical Debug Order

When something goes wrong, this order is usually the fastest way to localize it:

1. Check `object_cam0` and `object_cam1`
2. Check `hand_cam0` and `hand_cam1`
3. Check `selected_hand`
4. Check `merged_object.centroid_base`
5. Check `fusion.hand_object_distance_m`
6. Check `grasp_target.target_position_base`
7. Check `task.mode` and `task.active_reason`
8. Check `robot_command.command_type`
9. Check `safety.safe` and `safety.reason`
10. Check `robot.actual_tcp_pose_base`

## Current Execution Model

Although the architecture is logically modular, the current implementation runs in one Python process through [runner.py](/home/ur5/ICRA_vision_module/system/runner.py).

That means:

- the system is modular by code structure
- but still serialized in one runtime loop
- and all stage outputs are available immediately in `SharedStateBundle`

This makes debugging easier while the pipeline is still being validated.
