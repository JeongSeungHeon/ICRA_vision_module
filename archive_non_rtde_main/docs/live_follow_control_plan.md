# Live Follow Robot Control Migration Plan

## Goal

This document describes how to adapt the current UR5 RTDE robot control stack so that it follows the control style of [robot_control_sebin.py](/home/ur5/ICRA_vision_module/robot_control_sebin.py), while keeping the current vision pipeline unchanged.

The key idea is:

- keep the current dual-camera vision system as it is
- keep object / hand / fusion / grasp-target generation as it is
- replace only the robot-control policy with a real-time follow-servo style controller

## What Will Stay the Same

The following vision modules stay unchanged in principle:

- [dual_sensor_hub.py](/home/ur5/ICRA_vision_module/system/dual_sensor_hub.py)
- [object_worker.py](/home/ur5/ICRA_vision_module/perception/object_worker.py)
- [hand_worker.py](/home/ur5/ICRA_vision_module/perception/hand_worker.py)
- [hand_selector.py](/home/ur5/ICRA_vision_module/perception/hand_selector.py)
- [object_merger.py](/home/ur5/ICRA_vision_module/perception/object_merger.py)
- [fusion.py](/home/ur5/ICRA_vision_module/perception/fusion.py)
- [grasp_target.py](/home/ur5/ICRA_vision_module/perception/grasp_target.py)

This means the robot will still use:

- merged object point cloud
- selected hand pose
- fusion events such as hand approach and object lift
- final grasp target in robot base coordinates

## What Will Change

The control side will be changed from:

- discrete point-to-point task execution

into:

- real-time follow-servo execution
- target tracking with timeout handling
- workspace clamping
- fixed-orientation servo motion
- x-axis approach offset logic

The main files that will be modified are:

- [handover.yaml](/home/ur5/ICRA_vision_module/configs/handover.yaml)
- [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)
- [rtde_controller.py](/home/ur5/ICRA_vision_module/robot/rtde_controller.py)
- [runner.py](/home/ur5/ICRA_vision_module/system/runner.py)

The main new file to add is:

- `robot/live_follow_controller.py`

## Key Differences Between Current System and robot_control_sebin.py

### Current system

- uses state-machine-driven move logic
- uses `grasp_target.target_position_base` as a single target point
- activates robot only after hand approach and object lift
- moves toward grasp target, closes gripper, verifies grasp, returns to initial placement point

### robot_control_sebin.py style

- keeps a continuously refreshed latest target
- follows the target with a servo loop
- applies x-axis approach offset before grasp
- clamps command inside workspace
- uses target timeout to stop motion safely
- uses fixed orientation while updating only position

## Migration Strategy

Instead of rewriting the full system, we will insert a new control policy layer between:

- `grasp_target / fusion / task_manager`
- and `rtde_controller`

So the migration is:

1. preserve vision outputs
2. preserve robot-base target generation
3. replace the control policy that consumes those targets

## Step-by-Step Implementation Plan

## Step 1: Add Follow-Control Config

### Goal

Add a dedicated config block for the new follow-servo mode.

### Files

- [handover.yaml](/home/ur5/ICRA_vision_module/configs/handover.yaml)

### Additions

Add a new section such as:

```yaml
robot:
  control_mode: live_follow_servo
  live_follow:
    enabled: true
    control_hz: 30.0
    min_valid_count: 3
    target_timeout_sec: 0.5
    follow_z: true
    approach_axis: x
    approach_offset_m: -0.08
    max_xy_speed_mps: 0.08
    max_z_speed_mps: 0.08
    max_xy_step_m: 0.0027
    max_z_step_m: 0.0027
    workspace_clamp_enabled: true
```

### Why

This keeps the new behavior configurable without breaking the current classic control mode.

## Step 2: Create Follow Controller Policy Module

### Goal

Create a control-policy module inspired by [robot_control_sebin.py](/home/ur5/ICRA_vision_module/robot_control_sebin.py), but robot-agnostic and UR-compatible.

### New file

- `robot/live_follow_controller.py`

### Responsibilities

- hold the latest valid target
- maintain valid detection streak
- apply timeout logic
- apply x-axis approach offset
- optionally freeze z or allow z following
- limit per-step motion
- clamp target inside workspace
- output servo-compatible target position in robot base frame

### Input

- `grasp_target`
- `fusion`
- `robot_state`
- current task mode

### Output

- follow target in base frame
- follow validity flag
- debug info such as timeout / clamp / streak

## Step 3: Extend Shared State for Follow Debugging

### Goal

Make the new follow-control layer observable during runtime.

### Files

- [shared_state.py](/home/ur5/ICRA_vision_module/system/shared_state.py)

### Additions

Add a new state dataclass, for example:

- `LiveFollowState`

Suggested fields:

- `follow_enabled`
- `target_position_base`
- `approach_target_base`
- `streak_count`
- `target_is_fresh`
- `timed_out`
- `workspace_clamped`
- `follow_z_enabled`
- `valid`

### Why

We will need this to debug why the servo target is or is not moving.

## Step 4: Add Control-Mode Switch

### Goal

Allow the system to run either:

- current classic state-machine control
- new live-follow-servo control

### Files

- [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)
- [runner.py](/home/ur5/ICRA_vision_module/system/runner.py)

### Plan

Introduce a mode switch:

- `classic_state_machine`
- `live_follow_servo`

The runner reads config and routes control accordingly.

### Why

This lets us compare the new controller against the old one without losing the existing implementation.

## Step 5: Simplify Task Logic for Follow Mode

### Goal

In follow mode, reduce the role of [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py) so it becomes a high-level gate rather than a low-level trajectory planner.

### New follow-mode state progression

Recommended states:

1. `IDLE`
2. `OBSERVE_INITIAL_OBJECT`
3. `WAIT_FOR_HAND_APPROACH`
4. `WAIT_FOR_OBJECT_LIFT`
5. `FOLLOW_SERVO`
6. `CLOSE_GRIPPER`
7. `VERIFY_GRASP`
8. `RETURN_TO_INITIAL_PLACE`
9. `RELEASE`
10. `RETREAT`
11. `FAIL_SAFE`

### Behavior

- `FOLLOW_SERVO` continuously updates the robot target from the latest grasp target
- no single fixed pregrasp point is latched at entry
- the servo target moves as the perception target moves
- gripper close happens once target/robot alignment conditions are met

### Why

This matches the feel of `robot_control_sebin.py` much more closely.

## Step 6: Convert xArm Servo Logic into UR RTDE Servo Logic

### Goal

Map the `xarm.set_servo_cartesian()` style to UR RTDE `servoL()`.

### Files

- [rtde_controller.py](/home/ur5/ICRA_vision_module/robot/rtde_controller.py)

### Tasks

- keep using `ROBOT_CMD_SERVO_TO_POSITION`
- ensure the target is refreshed every loop
- keep orientation fixed
- update only `x, y, z`
- keep watchdog behavior for stale commands

### Notes

The low-level RTDE support already exists, so this step is mostly about making the controller consume continuous follow targets correctly.

## Step 7: Add x-Axis Approach Logic

### Goal

Match the `robot_control_sebin.py` behavior where the EEF approaches with an x-direction offset from the object.

### Files

- `robot/live_follow_controller.py`
- possibly [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)

### Logic

For follow mode:

- start from `grasp_target.target_position_base`
- add `approach_offset_m` along robot base x-axis
- optionally later support tool-frame approach, but base-frame x-axis is safer for the first implementation

### Why

This is one of the most important behavioral differences the user wants to preserve from `robot_control_sebin.py`.

## Step 8: Add Step Limiting and Workspace Clamp

### Goal

Keep the real-time servo target smooth and safe.

### Files

- `robot/live_follow_controller.py`
- [safety.py](/home/ur5/ICRA_vision_module/robot/safety.py)

### Logic

- limit XY motion per cycle
- limit Z motion per cycle separately
- clamp final target inside workspace bounds
- if target is stale, stop sending moving targets

### Why

This is necessary for UR safety and also matches the behavior of the reference xArm controller.

## Step 9: Add Target Timeout and Detection Streak Logic

### Goal

Replicate the `min_valid_count` and `target_timeout_s` behavior from [robot_control_sebin.py](/home/ur5/ICRA_vision_module/robot_control_sebin.py).

### Files

- `robot/live_follow_controller.py`

### Logic

- do not start servo follow until the target has been valid for N consecutive updates
- if the target goes stale for too long, hold or stop
- keep a short memory of the last valid target

### Why

This prevents target chatter and unnecessary robot twitching.

## Step 10: Define the Grasp Trigger Condition in Follow Mode

### Goal

Decide when the follow phase ends and the system closes the gripper.

### Files

- [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)

### Candidate rule

Close the gripper when all of the following hold:

- follow target is valid and fresh
- robot TCP is close enough to the follow target
- object is still lifted
- hand/object conditions remain valid for a short dwell

### Why

This replaces the current one-shot approach-to-grasp logic with a follow-based grasp trigger.

## Step 11: Keep the Post-Grasp Logic Mostly Unchanged

### Goal

Avoid changing more than necessary after grasp succeeds.

### Files

- [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)

### Keep as-is where possible

- force/current-based grasp verification
- return to memorized initial object centroid
- release there
- retreat

### Why

The main requested change is the approach/follow behavior, not the entire receive-place sequence.

## Step 12: Update Runner Integration

### Goal

Wire the new follow controller into the main loop.

### Files

- [runner.py](/home/ur5/ICRA_vision_module/system/runner.py)

### Plan

Add these steps to the runner in follow mode:

1. compute perception states as usual
2. compute `grasp_target` as usual
3. feed target into `live_follow_controller`
4. feed follow output into `task_manager`
5. send servo-style command to RTDE

### Why

This preserves the current top-level execution architecture while changing only the control behavior.

## Step 13: Extend Logging and Visualization

### Goal

Make the new servo-follow behavior debuggable.

### Files

- [runner.py](/home/ur5/ICRA_vision_module/system/runner.py)
- optional debug tools in `tools/`

### Add logs for

- raw grasp target
- approach-offset target
- clamped servo target
- follow valid streak
- target timeout state
- whether z-follow is active

### Why

Without these logs, it will be hard to diagnose why the robot is not tracking correctly.

## Step 14: Mock Verification First

### Goal

Test the new control logic without moving the real UR5.

### Method

Run:

```bash
python system/runner.py --config configs/handover.yaml --force-mock
```

Check:

- follow target updates continuously
- command type becomes `servo_to_position`
- timeout produces stop/hold
- workspace clamp behaves as expected

## Step 15: Real Robot Low-Speed Validation

### Goal

Deploy to UR5 safely.

### Safety-first plan

1. reduce speeds and step limits
2. test with `follow_z: false` first
3. verify x-axis approach offset direction
4. verify timeout stop
5. verify gripper close trigger
6. verify return-to-place after grasp verification

### Recommended first real-robot sequence

1. target tracking only, gripper disabled
2. target tracking with fixed z
3. target tracking with z enabled
4. full grasp trigger
5. return and release

## Recommended File-Level Work Order

1. [handover.yaml](/home/ur5/ICRA_vision_module/configs/handover.yaml)
2. `robot/live_follow_controller.py`
3. [shared_state.py](/home/ur5/ICRA_vision_module/system/shared_state.py)
4. [task_manager.py](/home/ur5/ICRA_vision_module/system/task_manager.py)
5. [rtde_controller.py](/home/ur5/ICRA_vision_module/robot/rtde_controller.py)
6. [runner.py](/home/ur5/ICRA_vision_module/system/runner.py)
7. logging / debug tools

## Implementation Philosophy

The safest way to do this migration is not to replace the whole robot stack at once.

Instead:

- keep the current vision outputs stable
- introduce one new follow-control policy layer
- preserve the old classic mode for comparison
- validate in mock first
- validate on UR slowly

This minimizes risk while still moving the system toward the behavior of [robot_control_sebin.py](/home/ur5/ICRA_vision_module/robot_control_sebin.py).
