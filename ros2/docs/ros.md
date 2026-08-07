# ROS2 전환 구현 개요

이 ROS2 전환 구현은 기존 Python 시스템 코드를 수정하지 않고, 모든 ROS2 관련 파일을 `./ros2` 아래에 격리한다.

## 원칙

- 기존 root 코드와 `configs/handover.yaml`은 읽기 전용으로 사용한다.
- ROS2 노드는 런타임에 repository root를 `PYTHONPATH`에 추가한 뒤 기존 모듈을 import한다.
- 새 message, service, action, launch, node 코드는 모두 `ros2/` 아래에 둔다.
- 로봇 명령은 기존 `RobotWorker`를 통해서만 실행해 RTDE/gripper 단일 소유권을 유지한다.

## 패키지

- `icra_handover_interfaces`
  - `ObjectState`, `HandState`, `FusionState`, `ShapeFitState`, `GraspTarget`, `RobotStatus`, `HandoverTaskState`
  - `SetFollow`, `ResetHome`, `EmergencyStop`
  - `ExecuteHandover`, `ExecuteMotionPlan`
- `icra_handover_ros2`
  - `dual_camera_node`
  - `perception_node`
  - `robot_node`
  - `handover_coordinator_node`
  - `visualization_node`
- `icra_handover_moveit`
  - `moveit_handover_node`
  - `handover_moveit.launch.py`

## 노드 역할

`dual_camera_node`는 기존 `DualSensorHub`를 감싸서 cam0/cam1 RGB, depth, camera info, sync 상태를 publish한다.

`perception_node`는 기존 object/hand/fusion/shape fitting/grasp planner 모듈을 사용한다. 기본은 ROS image topic 입력이며, `use_local_sensor_hub:=true`로 기존 카메라 허브를 직접 열 수 있다.

`robot_node`는 기존 `RobotWorker`를 service/action 인터페이스로 감싼다. launch 기본값은 hardware safety를 위해 `use_robot_node:=false`, `auto_init:=false`다.

`moveit_handover_node`는 `/perception/grasp_target`을 MoveIt2 pose goal로 변환해 UR5 `ur_manipulator` trajectory를 계획한다. 이 모드에서는 기존 RTDE `robot_node`를 끄고 `/motion/execute_plan`, `/motion/reset_home`, `/motion/emergency_stop`만 사용한다.

`handover_coordinator_node`는 perception target과 robot status를 모아 task state를 publish한다.

`visualization_node`는 RViz2용 marker를 publish한다.

## 기본 실행

```bash
cd ros2
colcon build
source install/setup.bash
ros2 launch icra_handover_ros2 handover.launch.py
```

conda 환경이 활성화되어 있으면 ROS Humble의 시스템 Python 3.10을 우선해서 빌드한다:

```bash
cd ros2
env PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/humble/bin \
  colcon build --cmake-clean-cache --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so
source install/setup.bash
```

`ModuleNotFoundError: No module named 'em'`은 보통 conda의 `python3`가
ROS message/action generator를 실행할 때 발생한다.

다른 위치에서 실행할 경우:

```bash
ros2 launch icra_handover_ros2 handover.launch.py repo_root:=/home/ur5/test3/ICRA_vision_module
```

`perception_node`는 기본적으로 `/home/ur5/miniforge3/envs/handover_ros2/bin/python`이
존재하면 이 Python으로 실행한다. 다른 런타임을 쓰려면
`perception_python:=/path/to/python`을 넘기고, 설치된 script shebang을 그대로 쓰려면
`perception_python:=`을 넘긴다.

## MoveIt2 실행

MoveIt2 motion planning은 `/home/ur5/workspace/install/setup.bash`의 official UR/MoveIt 패키지를 전제로 한다. `handover_moveit.launch.py`는 perception/camera/visualization stack을 include하되 `robot_node`는 항상 끈다.

```bash
cd ros2
source /opt/ros/humble/setup.bash
source /home/ur5/workspace/install/setup.bash
source install/setup.bash
ros2 launch icra_handover_moveit handover_moveit.launch.py
```

기본값은 `plan_only_default:=true`라 action goal이 실행 요청을 보내도 실제 robot/gripper execute는 막힌다. RViz에서 `/perception/grasp_target`, collision object, trajectory를 확인한 뒤에만 `plan_only_default:=false`로 실제 동작을 허용한다.

## 검증 순서

1. `colcon build`
2. `ros2 launch ... use_robot_node:=false`
3. camera topic 확인
4. perception output topic 확인
5. `use_robot_node:=true robot_auto_init:=false`로 robot node service/action smoke test
6. MoveIt2 모드는 `ros2 launch icra_handover_moveit handover_moveit.launch.py` 후 `/motion/execute_plan`을 `plan_only=true`로 검증
7. 실제 hardware 준비 후 `robot_auto_init:=true` 또는 MoveIt2 모드의 `plan_only_default:=false`
