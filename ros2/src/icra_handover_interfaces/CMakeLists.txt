cmake_minimum_required(VERSION 3.8)
project(icra_handover_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FusionState.msg"
  "msg/GraspTarget.msg"
  "msg/HandState.msg"
  "msg/HandoverTaskState.msg"
  "msg/ObjectState.msg"
  "msg/RobotStatus.msg"
  "msg/SensorSyncStatus.msg"
  "msg/ShapeFitState.msg"
  "srv/EmergencyStop.srv"
  "srv/ResetHome.srv"
  "srv/SetFollow.srv"
  "action/ExecuteHandover.action"
  "action/ExecuteMotionPlan.action"
  DEPENDENCIES builtin_interfaces geometry_msgs sensor_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
