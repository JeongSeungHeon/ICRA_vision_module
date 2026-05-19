# RobotWorker Thread Structure Proposal

## Goal

`robot_control_rtde_fitting_final.py` currently keeps camera reading and perception
on the main thread, while only live follow servo control runs in a separate follow
thread. Blocking robot actions such as gripper close, return, and place are still
executed from the main thread.

The preferred long-term structure is to introduce one dedicated `RobotWorker`
thread that owns all robot commands.

Main goal:

- Keep camera, perception, rendering, and recording alive during all robot actions.
- Prevent multiple threads from issuing conflicting RTDE commands.
- Make follow, grasp, return, place, reset, and stop behavior explicit through one
  robot state machine.

## Current Structure

```text
Main thread
  camera read
  object / hand perception
  fusion and grasp target update
  rendering
  keyboard handling
  gripper close             <- blocking
  return / place / home     <- blocking
  reset / save handling

Follow thread
  reads FollowSharedState
  sends SERVO_TO_POSITION while follow is active
```

This works, but it has two important limitations:

1. Camera processing stops during blocking main-thread robot actions.
2. Robot command ownership is split between the main thread and the follow thread.

## Proposed Structure

Replace the separate follow-command ownership and main-thread robot actions with a
single `RobotWorker`.

```text
Main thread
  camera read
  object / hand perception
  fusion and grasp target update
  rendering
  keyboard handling
  sends high-level requests to RobotWorker

RobotWorker thread
  owns RtdeController
  reads FollowSharedState when following
  executes follow servo
  executes gripper close
  executes return / place / home
  handles reset / stop requests
  publishes robot status snapshots
```

In this model, the main thread should not directly call motion-producing robot
methods. It should request robot work and continue the perception loop.

## Ownership Rule

The central rule is:

```text
Only RobotWorker sends robot motion or gripper commands.
```

This should include:

- `controller.step(...)`
- `send_robot_command(...)`
- `move_robot_to_home_pose(...)`
- `move_robot_and_wait(...)`
- `execute_gripper_close(...)`
- `execute_gripper_open(...)`
- `execute_return_and_place(...)`
- `safe_stop_rtde(...)`

The main thread may still read passive status if needed, but the cleaner design is
for `RobotWorker` to publish a thread-safe robot status snapshot for overlays and
debug logging.

## RobotWorker State Machine

Recommended states:

```text
IDLE
INITIALIZING
FOLLOWING
GRASPING
RETURNING
PLACING
RESETTING
DONE
ERROR
STOPPING
```

Suggested meaning:

- `IDLE`: robot is connected or ready, no active task.
- `INITIALIZING`: RTDE connection, HOME move, fixed pose capture.
- `FOLLOWING`: live servo follows the current target from `FollowSharedState`.
- `GRASPING`: follow has stopped, gripper close is running.
- `RETURNING`: robot is moving toward the delivery/place region.
- `PLACING`: release, backoff, and HOME sequence are running.
- `RESETTING`: reset/home sequence is running.
- `DONE`: current handover task is complete.
- `ERROR`: worker hit an unrecoverable action error.
- `STOPPING`: safe stop or shutdown is being processed.

## Request Interface

The main thread should communicate with `RobotWorker` using high-level requests.

Example request types:

```text
INIT_ROBOT
START_FOLLOW
STOP_FOLLOW
START_GRASP_PLACE
RESET_HOME
SAVE_AND_STOP
EMERGENCY_STOP
SHUTDOWN
```

The request can be implemented with `queue.Queue`.

Example shape:

```python
@dataclass
class RobotRequest:
    type: str
    payload: dict
    created_at: float = field(default_factory=time.time)
```

Main thread example:

```python
robot_worker.submit(RobotRequest("START_GRASP_PLACE", {
    "metadata_recorder": metadata_recorder,
    "tactile_manager": tactile_manager,
}))
```

The worker should reject or defer requests that are invalid for the current state.

## Follow Behavior

The existing `robot_control_loop()` behavior should move into `RobotWorker`.

During `FOLLOWING`, the worker repeatedly:

1. Reads `shared_state.get_snapshot()`.
2. Checks whether follow is active.
3. Computes the bounded servo target.
4. Sends `ROBOT_CMD_SERVO_TO_POSITION`.
5. Sleeps to maintain `args.control_hz`.

The main thread continues to update target measurements through
`shared_state.update_target(...)`.

```text
Main thread:
  perception -> shared_state.update_target()

RobotWorker:
  FOLLOWING -> shared_state.get_snapshot() -> servo command
```

## Grasp / Return / Place Behavior

When `START_GRASP_PLACE` is accepted:

```text
FOLLOWING
  -> stop live servo
  -> GRASPING
  -> execute gripper close
  -> save grasp offset
  -> RETURNING
  -> execute return move
  -> PLACING
  -> execute release / backoff / HOME
  -> DONE
```

Important detail:

- The worker should snapshot action-critical state at the start of the action.
- Perception may keep updating shared state, but the active grasp/place sequence
  should use stable values captured for that action.

Values to snapshot or freeze carefully:

- fixed orientation,
- initial robot pose,
- grasp offset,
- place target,
- frozen place z samples,
- home object position,
- tactile release parameters.

## Reset Behavior

The main thread can keep handling the r key, but it should not directly run robot
reset motion. Instead:

```text
r key
  -> main thread submits RESET_HOME request
  -> RobotWorker cancels or stops current robot activity safely
  -> RobotWorker performs HOME/reset robot motion
  -> main thread resets perception/task bookkeeping when worker reports reset done
```

This is different from the current structure, where the main thread directly
performs reset motion after stopping the follow thread.

With `RobotWorker`, reset should be part of the same state machine so it cannot
collide with grasp/place or follow commands.

## Cancel Policy

RobotWorker needs an explicit cancel policy because some current robot functions are
blocking.

Recommended behavior for `RESET_HOME` during an active action:

```text
1. Mark cancel requested.
2. Stop current motion safely.
3. Exit the active action at the next safe checkpoint.
4. Enter RESETTING.
5. Move robot to HOME.
6. Report reset complete.
```

To support this, blocking functions should be made cancel-aware.

Examples:

```python
def wait_until_target_reached(..., cancel_event=None):
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_rtde(controller)
            return False
        ...
```

```python
def execute_gripper_close(..., cancel_event=None):
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            stop_gripper_motion_safely(controller, "cancel")
            return False
        ...
```

This gives r key behavior similar to the current follow reset behavior, but makes it
safe for long grasp/place actions too.

## Main Thread Responsibilities

After the change, the main thread should be responsible for:

- sensor read,
- object and hand perception,
- fusion and target estimation,
- `FollowSharedState` updates,
- rendering and keyboard input,
- video/debug/runtime logging,
- sending high-level robot requests,
- reacting to worker status changes.

The main thread should not block on long robot moves.

## RobotWorker Responsibilities

`RobotWorker` should be responsible for:

- RTDE initialization and connection lifecycle,
- HOME movement,
- live follow servo,
- gripper close/open,
- return/place/backoff sequence,
- reset/home sequence,
- safe stop,
- cancel handling,
- robot status snapshots,
- action result reporting.

## Status Reporting

RobotWorker should publish a small thread-safe status object.

Suggested fields:

```text
state
last_error
is_connected
using_mock
active_request
last_robot_pose
last_command_type
grasp_ok
task_done
reset_done
```

The main thread can use this for:

- overlay text,
- logging,
- metadata,
- deciding whether to submit new requests.

## Benefits

The `RobotWorker` structure gives the system a cleaner concurrency boundary:

- Camera and perception continue during grasp/place/reset.
- All robot commands are serialized through one owner.
- Follow and grasp/place cannot fight for RTDE control.
- r key reset can be made safe through cancel-aware worker logic.
- Robot behavior becomes easier to reason about as a state machine.
- Future features such as emergency stop, retry, or task replay can be added more cleanly.

## Migration Plan

Recommended implementation order:

1. Add `RobotWorker` class with request queue, status snapshot, and lifecycle methods.
2. Move RTDE initialization and HOME startup into `RobotWorker`.
3. Move the existing follow loop logic into `RobotWorker.FOLLOWING`.
4. Replace direct main-thread follow thread creation with `robot_worker.submit(START_FOLLOW)`.
5. Move gripper close and return/place calls into worker states.
6. Convert r key reset into a `RESET_HOME` request.
7. Add cancel-aware checks to blocking wait loops.
8. Move passive robot pose reads for overlays to worker status snapshots where practical.
9. Add focused tests for request/state transitions and duplicate request rejection.

## Summary

The preferred final architecture is not "main thread plus follow thread plus action
thread." The preferred architecture is:

```text
Main thread:
  perception, UI, logging, and high-level decisions

RobotWorker thread:
  the single owner of all robot commands and robot task state
```

This requires more refactoring than a short-term action-thread patch, but it is the
more robust structure for continued development.