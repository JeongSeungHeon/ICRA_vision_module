# ICRA Handover ROS2 Workspace

This directory is an isolated ROS2 colcon workspace for the existing Python
handover runtime. It does not modify or move the root Python system.

## Layout

```text
ros2/
  src/icra_handover_interfaces/  # custom msg/srv/action definitions
  src/icra_handover_ros2/        # ROS2 adapter nodes and launch files
  src/icra_handover_moveit/      # MoveIt2 motion planning node and launch
  config/handover_ros2.yaml      # ROS2-specific parameter defaults
  docs/ros.md                    # migration overview
```

## Build

```bash
cd ros2
colcon build
source install/setup.bash
```

If a conda environment is active, build with ROS Humble's system Python:

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

This avoids ROS interface generation accidentally using a conda Python that is
missing ROS build modules such as `em`.

When building the MoveIt2 package from an active conda shell, also clear conda
library paths so the linker uses the system OpenSSL/libcurl used by ROS:

```bash
cd ros2
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH PKG_CONFIG_PATH
source /opt/ros/humble/setup.bash
source /home/ur5/workspace/install/setup.bash
env PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/humble/bin \
  colcon build --packages-select icra_handover_interfaces icra_handover_moveit icra_handover_ros2 \
  --cmake-clean-cache --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so
source install/setup.bash
```

## Launch

From the repository's `ros2` directory:

```bash
ros2 launch icra_handover_ros2 handover.launch.py
```

The launch file infers the repository root from the current directory. If you
run from another location, pass it explicitly:

```bash
ros2 launch icra_handover_ros2 handover.launch.py repo_root:=/path/to/ICRA_vision_module
```

By default, `perception_node` uses `/home/ur5/miniforge3/envs/handover_ros2/bin/python`
when that environment exists. Override it with `perception_python:=/path/to/python`,
or pass `perception_python:=` to use the installed script's shebang.

Robot hardware is disabled by default:

```bash
ros2 launch icra_handover_ros2 handover.launch.py use_robot_node:=true robot_auto_init:=false
```

Set `robot_auto_init:=true` only when the UR/RTDE and gripper environment is
ready.

## MoveIt2 Motion Planning

Start the official UR driver and MoveIt externally, then launch the perception
stack plus `moveit_handover_node`:

```bash
cd ros2
source /opt/ros/humble/setup.bash
source /home/ur5/workspace/install/setup.bash
source install/setup.bash
ros2 launch icra_handover_moveit handover_moveit.launch.py
```

The MoveIt launch keeps the old RTDE `robot_node` disabled and uses
`plan_only_default:=true` by default. To ask it to plan from the latest
perception target:

```bash
ros2 action send_goal /motion/execute_plan icra_handover_interfaces/action/ExecuteMotionPlan \
  "{use_latest_target: true, plan_only: true, reset_before_start: true, execute_grasp_place: true, timeout_s: 5.0}"
```

Only set `plan_only_default:=false` after validating the plan in RViz and
confirming that the UR driver, MoveIt controller, gripper, and emergency stop
path are ready.
