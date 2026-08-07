"""Launch the ICRA perception stack with the MoveIt2 handover motion node."""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


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


def _robot_description(ur_type: LaunchConfiguration, robot_ip: LaunchConfiguration) -> dict:
    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "joint_limits.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "default_kinematics.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "visual_parameters.yaml"]
    )
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "joint_limit_params:=",
            joint_limit_params,
            " ",
            "kinematics_params:=",
            kinematics_params,
            " ",
            "physical_params:=",
            physical_params,
            " ",
            "visual_params:=",
            visual_params,
            " ",
            "safety_limits:=true",
            " ",
            "safety_pos_margin:=0.15",
            " ",
            "safety_k_position:=20",
            " ",
            "name:=ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "prefix:=",
            '""',
            " ",
        ]
    )
    return {"robot_description": ParameterValue(robot_description_content, value_type=str)}


def _robot_description_semantic() -> dict:
    semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"]),
            " ",
            "name:=ur",
            " ",
            "prefix:=",
            '""',
            " ",
        ]
    )
    return {"robot_description_semantic": ParameterValue(semantic_content, value_type=str)}


def generate_launch_description() -> LaunchDescription:
    repo_root = LaunchConfiguration("repo_root")
    config_path = LaunchConfiguration("config_path")
    robot_ip = LaunchConfiguration("robot_ip")
    ur_type = LaunchConfiguration("ur_type")

    declared_arguments = [
        DeclareLaunchArgument("repo_root", default_value=_default_repo_root()),
        DeclareLaunchArgument("config_path", default_value="configs/handover.yaml"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.56.101"),
        DeclareLaunchArgument("ur_type", default_value="ur5"),
        DeclareLaunchArgument("reverse_ip", default_value=""),
        DeclareLaunchArgument("launch_ur_driver", default_value="false"),
        DeclareLaunchArgument("launch_move_group", default_value="false"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("use_camera_node", default_value="true"),
        DeclareLaunchArgument("use_perception_node", default_value="true"),
        DeclareLaunchArgument("use_coordinator_node", default_value="true"),
        DeclareLaunchArgument("use_visualization_node", default_value="true"),
        DeclareLaunchArgument("use_moveit_node", default_value="true"),
        DeclareLaunchArgument("camera_start_on_init", default_value="true"),
        DeclareLaunchArgument("perception_use_local_sensor_hub", default_value="false"),
        DeclareLaunchArgument("model", default_value="yoloe-26l-seg.pt"),
        DeclareLaunchArgument("device", default_value=""),
        DeclareLaunchArgument("perception_python", default_value=_default_perception_python()),
        DeclareLaunchArgument("planning_group", default_value="ur_manipulator"),
        DeclareLaunchArgument("end_effector_link", default_value="tool0"),
        DeclareLaunchArgument("target_frame", default_value="robot_base"),
        DeclareLaunchArgument("assume_identity_base_frame", default_value="true"),
        DeclareLaunchArgument("plan_only_default", default_value="true"),
        DeclareLaunchArgument("enable_gripper", default_value="true"),
        DeclareLaunchArgument("max_velocity_scaling", default_value="0.05"),
        DeclareLaunchArgument("max_acceleration_scaling", default_value="0.05"),
        DeclareLaunchArgument("min_cartesian_fraction", default_value="0.95"),
        DeclareLaunchArgument("max_target_age_s", default_value="0.5"),
    ]

    handover_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("icra_handover_ros2"), "launch", "handover.launch.py"]
            )
        ),
        launch_arguments={
            "repo_root": repo_root,
            "config_path": config_path,
            "use_camera_node": LaunchConfiguration("use_camera_node"),
            "use_perception_node": LaunchConfiguration("use_perception_node"),
            "use_robot_node": "false",
            "use_coordinator_node": LaunchConfiguration("use_coordinator_node"),
            "use_visualization_node": LaunchConfiguration("use_visualization_node"),
            "camera_start_on_init": LaunchConfiguration("camera_start_on_init"),
            "perception_use_local_sensor_hub": LaunchConfiguration(
                "perception_use_local_sensor_hub"
            ),
            "robot_auto_init": "false",
            "robot_auto_start_follow": "false",
            "robot_ip": "",
            "model": LaunchConfiguration("model"),
            "device": LaunchConfiguration("device"),
            "perception_python": LaunchConfiguration("perception_python"),
        }.items(),
    )

    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_robot_driver"), "launch", "ur_control.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_ur_driver")),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "launch_rviz": "false",
        }.items(),
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_move_group")),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "reverse_ip": LaunchConfiguration("reverse_ip"),
            "launch_rviz": LaunchConfiguration("launch_rviz"),
        }.items(),
    )

    moveit_node = Node(
        package="icra_handover_moveit",
        executable="moveit_handover_node",
        name="moveit_handover_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_moveit_node")),
        parameters=[
            _robot_description(ur_type, robot_ip),
            _robot_description_semantic(),
            {
                "robot_ip": robot_ip,
                "planning_group": LaunchConfiguration("planning_group"),
                "end_effector_link": LaunchConfiguration("end_effector_link"),
                "target_frame": LaunchConfiguration("target_frame"),
                "assume_identity_base_frame": ParameterValue(
                    LaunchConfiguration("assume_identity_base_frame"), value_type=bool
                ),
                "plan_only_default": ParameterValue(
                    LaunchConfiguration("plan_only_default"), value_type=bool
                ),
                "enable_gripper": ParameterValue(
                    LaunchConfiguration("enable_gripper"), value_type=bool
                ),
                "max_velocity_scaling": ParameterValue(
                    LaunchConfiguration("max_velocity_scaling"), value_type=float
                ),
                "max_acceleration_scaling": ParameterValue(
                    LaunchConfiguration("max_acceleration_scaling"), value_type=float
                ),
                "min_cartesian_fraction": ParameterValue(
                    LaunchConfiguration("min_cartesian_fraction"), value_type=float
                ),
                "max_target_age_s": ParameterValue(
                    LaunchConfiguration("max_target_age_s"), value_type=float
                ),
            },
        ],
    )

    return LaunchDescription(
        declared_arguments + [ur_driver, move_group, handover_stack, moveit_node]
    )
