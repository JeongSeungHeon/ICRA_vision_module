"""Launch the ROS2 adapter stack for the ICRA handover runtime."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _default_repo_root() -> str:
    env_value = os.environ.get("ICRA_VISION_REPO_ROOT")
    if env_value:
        return env_value
    cwd = Path.cwd().resolve()
    if cwd.name == "ros2" and (cwd.parent / "robot_control_rtde_fitting_final.py").exists():
        return str(cwd.parent)
    if (cwd / "robot_control_rtde_fitting_final.py").exists():
        return str(cwd)
    return str(cwd)


def _default_perception_python() -> str:
    env_value = os.environ.get("ICRA_VISION_PERCEPTION_PYTHON")
    if env_value:
        return env_value
    handover_env_python = Path("/home/ur5/miniforge3/envs/handover_ros2/bin/python")
    if handover_env_python.exists():
        return str(handover_env_python)
    return ""


def _default_perception_ld_library_path() -> str:
    handover_env_lib = Path("/home/ur5/miniforge3/envs/handover_ros2/lib")
    current_value = os.environ.get("LD_LIBRARY_PATH", "")
    if not handover_env_lib.exists():
        return current_value
    paths = [str(handover_env_lib)]
    paths.extend(path for path in current_value.split(":") if path and path not in paths)
    return ":".join(paths)


def generate_launch_description() -> LaunchDescription:
    repo_root = LaunchConfiguration("repo_root")
    config_path = LaunchConfiguration("config_path")

    declared_arguments = [
        DeclareLaunchArgument("repo_root", default_value=_default_repo_root()),
        DeclareLaunchArgument("config_path", default_value="configs/handover.yaml"),
        DeclareLaunchArgument("use_camera_node", default_value="true"),
        DeclareLaunchArgument("use_perception_node", default_value="true"),
        DeclareLaunchArgument("use_robot_node", default_value="false"),
        DeclareLaunchArgument("use_coordinator_node", default_value="true"),
        DeclareLaunchArgument("use_visualization_node", default_value="true"),
        DeclareLaunchArgument("camera_start_on_init", default_value="true"),
        DeclareLaunchArgument("perception_use_local_sensor_hub", default_value="false"),
        DeclareLaunchArgument("robot_auto_init", default_value="false"),
        DeclareLaunchArgument("robot_auto_start_follow", default_value="false"),
        DeclareLaunchArgument("robot_ip", default_value=""),
        DeclareLaunchArgument("model", default_value="yoloe-26l-seg.pt"),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("perception_python", default_value=_default_perception_python()),
    ]

    common_params = {
        "repo_root": repo_root,
        "config_path": config_path,
    }

    return LaunchDescription(
        declared_arguments
        + [
            SetEnvironmentVariable("ICRA_VISION_REPO_ROOT", repo_root),
            SetEnvironmentVariable("MPLCONFIGDIR", "/tmp/icra_handover_mpl"),
            Node(
                package="icra_handover_ros2",
                executable="dual_camera_node",
                name="dual_camera_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_camera_node")),
                cwd=repo_root,
                parameters=[
                    common_params,
                    {
                        "start_on_init": ParameterValue(LaunchConfiguration("camera_start_on_init"), value_type=bool),
                    },
                ],
            ),
            Node(
                package="icra_handover_ros2",
                executable="perception_node",
                name="perception_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_perception_node")),
                prefix=LaunchConfiguration("perception_python"),
                cwd=repo_root,
                additional_env={
                    "LD_LIBRARY_PATH": _default_perception_ld_library_path(),
                },
                parameters=[
                    common_params,
                    {
                        "use_local_sensor_hub": ParameterValue(
                            LaunchConfiguration("perception_use_local_sensor_hub"),
                            value_type=bool,
                        ),
                        "model": LaunchConfiguration("model"),
                        "device": LaunchConfiguration("device"),
                    },
                ],
            ),
            Node(
                package="icra_handover_ros2",
                executable="robot_node",
                name="robot_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_robot_node")),
                cwd=repo_root,
                parameters=[
                    common_params,
                    {
                        "auto_init": ParameterValue(LaunchConfiguration("robot_auto_init"), value_type=bool),
                        "auto_start_follow": ParameterValue(
                            LaunchConfiguration("robot_auto_start_follow"),
                            value_type=bool,
                        ),
                        "robot_ip": LaunchConfiguration("robot_ip"),
                    },
                ],
            ),
            Node(
                package="icra_handover_ros2",
                executable="handover_coordinator_node",
                name="handover_coordinator_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_coordinator_node")),
            ),
            Node(
                package="icra_handover_ros2",
                executable="visualization_node",
                name="visualization_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_visualization_node")),
            ),
        ]
    )
