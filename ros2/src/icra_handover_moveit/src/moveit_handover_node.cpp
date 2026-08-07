#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <icra_handover_interfaces/action/execute_motion_plan.hpp>
#include <icra_handover_interfaces/msg/grasp_target.hpp>
#include <icra_handover_interfaces/msg/object_state.hpp>
#include <icra_handover_interfaces/msg/robot_status.hpp>
#include <icra_handover_interfaces/srv/emergency_stop.hpp>
#include <icra_handover_interfaces/srv/reset_home.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <shape_msgs/msg/plane.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <ur_rtde/robotiq_gripper.h>

namespace
{
using namespace std::chrono_literals;

using ExecuteMotionPlan = icra_handover_interfaces::action::ExecuteMotionPlan;
using ExecuteMotionPlanGoalHandle = rclcpp_action::ServerGoalHandle<ExecuteMotionPlan>;

constexpr const char * kObjectCollisionId = "handover_object";
constexpr const char * kGroundCollisionId = "handover_ground";

bool is_finite(double value)
{
  return std::isfinite(value);
}

bool finite_point(const geometry_msgs::msg::Point & point)
{
  return is_finite(point.x) && is_finite(point.y) && is_finite(point.z);
}

bool finite_pose(const geometry_msgs::msg::Pose & pose)
{
  return finite_point(pose.position) && is_finite(pose.orientation.x) && is_finite(
    pose.orientation.y) &&
         is_finite(pose.orientation.z) && is_finite(pose.orientation.w);
}

std::vector<double> declare_vector(
  const rclcpp::Node::SharedPtr & node,
  const std::string & name,
  const std::vector<double> & default_value)
{
  node->declare_parameter<std::vector<double>>(name, default_value);
  return node->get_parameter(name).as_double_array();
}

double norm3(const std::vector<double> & values)
{
  if (values.size() < 3) {
    return 0.0;
  }
  return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

geometry_msgs::msg::Pose offset_pose(
  geometry_msgs::msg::Pose pose,
  const std::vector<double> & axis,
  double distance_m)
{
  if (axis.size() >= 3) {
    pose.position.x += axis[0] * distance_m;
    pose.position.y += axis[1] * distance_m;
    pose.position.z += axis[2] * distance_m;
  }
  return pose;
}

double stamp_age_s(const rclcpp::Node::SharedPtr & node, const std_msgs::msg::Header & header)
{
  const rclcpp::Time stamp(header.stamp);
  if (stamp.nanoseconds() == 0) {
    return 0.0;
  }
  return (node->now() - stamp).seconds();
}

template<typename T>
bool read_point_field(
  const sensor_msgs::msg::PointCloud2 & msg, size_t offset, size_t index,
  double & out)
{
  T value{};
  const size_t base = index * msg.point_step + offset;
  if (base + sizeof(T) > msg.data.size()) {
    return false;
  }
  std::memcpy(&value, msg.data.data() + base, sizeof(T));
  out = static_cast<double>(value);
  return std::isfinite(out);
}

struct CloudBounds
{
  bool valid{false};
  rclcpp::Time stamp;
  std::string frame_id;
  geometry_msgs::msg::Point center;
  double size_x{0.0};
  double size_y{0.0};
  double size_z{0.0};
};

std::optional<CloudBounds> compute_cloud_bounds(const sensor_msgs::msg::PointCloud2 & msg)
{
  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  int datatype = -1;

  for (const auto & field : msg.fields) {
    if (field.name == "x") {
      x_offset = static_cast<int>(field.offset);
      datatype = static_cast<int>(field.datatype);
    } else if (field.name == "y") {
      y_offset = static_cast<int>(field.offset);
    } else if (field.name == "z") {
      z_offset = static_cast<int>(field.offset);
    }
  }
  if (x_offset < 0 || y_offset < 0 || z_offset < 0 || msg.point_step == 0) {
    return std::nullopt;
  }

  const size_t count = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  size_t valid_count = 0;

  for (size_t i = 0; i < count; ++i) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool ok = false;
    if (datatype == sensor_msgs::msg::PointField::FLOAT64) {
      ok = read_point_field<double>(msg, x_offset, i, x) && read_point_field<double>(
        msg, y_offset,
        i, y) &&
        read_point_field<double>(msg, z_offset, i, z);
    } else {
      ok = read_point_field<float>(msg, x_offset, i, x) && read_point_field<float>(
        msg, y_offset, i,
        y) &&
        read_point_field<float>(msg, z_offset, i, z);
    }
    if (!ok) {
      continue;
    }
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    min_z = std::min(min_z, z);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    max_z = std::max(max_z, z);
    ++valid_count;
  }

  if (valid_count == 0) {
    return std::nullopt;
  }

  CloudBounds bounds;
  bounds.valid = true;
  bounds.stamp = rclcpp::Time(msg.header.stamp);
  bounds.frame_id = msg.header.frame_id;
  bounds.center.x = 0.5 * (min_x + max_x);
  bounds.center.y = 0.5 * (min_y + max_y);
  bounds.center.z = 0.5 * (min_z + max_z);
  bounds.size_x = std::max(max_x - min_x, 0.02);
  bounds.size_y = std::max(max_y - min_y, 0.02);
  bounds.size_z = std::max(max_z - min_z, 0.02);
  return bounds;
}

}  // namespace

class MoveItHandoverNode
{
public:
  explicit MoveItHandoverNode(const rclcpp::Node::SharedPtr & node)
  : node_(node), tf_buffer_(node_->get_clock()), tf_listener_(tf_buffer_)
  {
    planning_group_ = node_->declare_parameter<std::string>("planning_group", "ur_manipulator");
    eef_link_ = node_->declare_parameter<std::string>("end_effector_link", "tool0");
    target_frame_ = node_->declare_parameter<std::string>("target_frame", "robot_base");
    assume_identity_base_frame_ =
      node_->declare_parameter<bool>("assume_identity_base_frame", true);
    robot_ip_ = node_->declare_parameter<std::string>("robot_ip", "192.168.56.101");
    gripper_port_ = node_->declare_parameter<int>("gripper_port", 63352);
    enable_gripper_ = node_->declare_parameter<bool>("enable_gripper", true);
    activate_gripper_on_connect_ = node_->declare_parameter<bool>(
      "activate_gripper_on_connect",
      true);
    plan_only_default_ = node_->declare_parameter<bool>("plan_only_default", true);
    max_velocity_scaling_ = node_->declare_parameter<double>("max_velocity_scaling", 0.05);
    max_acceleration_scaling_ = node_->declare_parameter<double>("max_acceleration_scaling", 0.05);
    min_cartesian_fraction_ = node_->declare_parameter<double>("min_cartesian_fraction", 0.95);
    planning_time_s_ = node_->declare_parameter<double>("planning_time_s", 5.0);
    max_target_age_s_ = node_->declare_parameter<double>("max_target_age_s", 0.5);
    object_cloud_max_age_s_ = node_->declare_parameter<double>("object_cloud_max_age_s", 0.5);
    object_state_max_age_s_ = node_->declare_parameter<double>("object_state_max_age_s", 0.5);
    eef_step_ = node_->declare_parameter<double>("eef_step", 0.005);
    jump_threshold_ = node_->declare_parameter<double>("jump_threshold", 0.0);
    pregrasp_offset_m_ = node_->declare_parameter<double>("pregrasp_offset_m", 0.10);
    retreat_z_offset_m_ = node_->declare_parameter<double>("retreat_z_offset_m", 0.08);
    ground_z_m_ = node_->declare_parameter<double>("ground_z_m", -0.01);
    ground_size_m_ = node_->declare_parameter<double>("ground_size_m", 3.0);
    status_rate_hz_ = node_->declare_parameter<double>("status_rate_hz", 5.0);
    gripper_open_speed_ = node_->declare_parameter<double>("gripper_open_speed", 1.0);
    gripper_open_force_ = node_->declare_parameter<double>("gripper_open_force", 1.0);
    gripper_close_speed_ = node_->declare_parameter<double>("gripper_close_speed", 1.0);
    gripper_close_force_ = node_->declare_parameter<double>("gripper_close_force", 1.0);

    approach_axis_ = declare_vector(node_, "approach_axis_base", {-1.0, 0.0, 0.0});
    const double approach_norm = norm3(approach_axis_);
    if (approach_norm > 1e-9 && approach_axis_.size() >= 3) {
      approach_axis_[0] /= approach_norm;
      approach_axis_[1] /= approach_norm;
      approach_axis_[2] /= approach_norm;
    }
    home_joints_deg_ =
      declare_vector(node_, "home_joints_deg", {0.0, -135.0, 135.0, 0.0, 90.0, 0.0});
    place_xyz_m_ = declare_vector(node_, "place_xyz_m", {0.330, 0.097, 0.171});
    default_object_size_m_ = declare_vector(node_, "default_object_size_m", {0.10, 0.10, 0.12});

    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      node_,
      planning_group_);
    if (!eef_link_.empty()) {
      move_group_->setEndEffectorLink(eef_link_);
    }
    move_group_->setMaxVelocityScalingFactor(max_velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(max_acceleration_scaling_);
    move_group_->setPlanningTime(planning_time_s_);
    planning_frame_ = move_group_->getPlanningFrame();

    planning_scene_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
    add_ground_plane();

    const rclcpp::QoS qos(rclcpp::KeepLast(10));
    grasp_sub_ = node_->create_subscription<icra_handover_interfaces::msg::GraspTarget>(
      "/perception/grasp_target", qos,
      [this](icra_handover_interfaces::msg::GraspTarget::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_grasp_target_ = *msg;
      });
    object_sub_ = node_->create_subscription<icra_handover_interfaces::msg::ObjectState>(
      "/perception/merged_object_state", qos,
      [this](icra_handover_interfaces::msg::ObjectState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_object_state_ = *msg;
      });
    fitted_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/perception/fitted_template_cloud", qos, [this](
        sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto bounds = compute_cloud_bounds(*msg);
        if (!bounds) {
          return;
        }
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_cloud_bounds_ = *bounds;
      });

    status_pub_ = node_->create_publisher<icra_handover_interfaces::msg::RobotStatus>(
      "/robot/status", qos);
    reset_srv_ = node_->create_service<icra_handover_interfaces::srv::ResetHome>(
      "/motion/reset_home",
      [this](const std::shared_ptr<icra_handover_interfaces::srv::ResetHome::Request> request,
      std::shared_ptr<icra_handover_interfaces::srv::ResetHome::Response> response) {
        handle_reset_home(request, response);
      });
    emergency_stop_srv_ = node_->create_service<icra_handover_interfaces::srv::EmergencyStop>(
      "/motion/emergency_stop",
      [this](const std::shared_ptr<icra_handover_interfaces::srv::EmergencyStop::Request> request,
      std::shared_ptr<icra_handover_interfaces::srv::EmergencyStop::Response> response) {
        handle_emergency_stop(request, response);
      });

    action_server_ = rclcpp_action::create_server<ExecuteMotionPlan>(
      node_,
      "/motion/execute_plan",
      std::bind(
        &MoveItHandoverNode::handle_goal, this, std::placeholders::_1,
        std::placeholders::_2),
      std::bind(&MoveItHandoverNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&MoveItHandoverNode::handle_accepted, this, std::placeholders::_1));

    const double safe_status_rate_hz = std::max(status_rate_hz_, 0.5);
    status_timer_ = node_->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(
          1.0 /
          safe_status_rate_hz)),
      [this]() {publish_status();});

    set_state("IDLE", "");
    RCLCPP_INFO(
      node_->get_logger(),
      "moveit_handover_node ready: group=%s eef=%s planning_frame=%s plan_only_default=%s",
      planning_group_.c_str(),
      eef_link_.c_str(),
      planning_frame_.c_str(),
      plan_only_default_ ? "true" : "false");
  }

private:
  struct Target
  {
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Point object_point;
    std::string frame_id;
    double age_s{0.0};
  };

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ExecuteMotionPlan::Goal>)
  {
    if (active_.load()) {
      RCLCPP_WARN(node_->get_logger(), "rejecting motion goal because another goal is active");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<ExecuteMotionPlanGoalHandle>)
  {
    cancel_requested_.store(true);
    if (move_group_) {
      move_group_->stop();
    }
    set_state("STOPPING", "motion goal cancelled");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<ExecuteMotionPlanGoalHandle> goal_handle)
  {
    active_.store(true);
    cancel_requested_.store(false);
    std::thread([this, goal_handle]() {execute_goal(goal_handle);}).detach();
  }

  void execute_goal(const std::shared_ptr<ExecuteMotionPlanGoalHandle> goal_handle)
  {
    auto result = std::make_shared<ExecuteMotionPlan::Result>();
    result->success = false;
    result->executed = false;
    result->final_state = "ERROR";
    result->min_cartesian_fraction = 1.0F;

    std::unique_lock<std::mutex> lock(move_group_mutex_);
    try {
      const auto goal = goal_handle->get_goal();
      const bool effective_plan_only = plan_only_default_ || goal->plan_only;
      Target target = resolve_target(*goal);
      target.pose = transform_pose_to_planning_frame(target.pose, target.frame_id);
      geometry_msgs::msg::Point object_point_planning = transform_point_to_planning_frame(
        target.object_point, target.frame_id);

      const double planning_time = goal->timeout_s >
        0.0F ? static_cast<double>(goal->timeout_s) : planning_time_s_;
      move_group_->setPlanningTime(planning_time);
      move_group_->setMaxVelocityScalingFactor(max_velocity_scaling_);
      move_group_->setMaxAccelerationScalingFactor(max_acceleration_scaling_);
      move_group_->clearPoseTargets();
      add_ground_plane();
      add_or_update_object_collision(object_point_planning);

      publish_feedback(goal_handle, "TARGET_READY", target.age_s, 1.0);
      set_state(effective_plan_only ? "PLANNING" : "MOVING", "");

      if (goal->reset_before_start) {
        publish_feedback(goal_handle, "PLAN_HOME", target.age_s, 1.0);
        plan_and_maybe_execute_joints(home_joints_rad(), effective_plan_only, "HOME", result);
        check_cancel(goal_handle);
      }

      const geometry_msgs::msg::Pose pregrasp_pose =
        offset_pose(target.pose, approach_axis_, std::abs(pregrasp_offset_m_));
      publish_feedback(goal_handle, "PLAN_PREGRASP", target.age_s, 1.0);
      plan_and_maybe_execute_pose(pregrasp_pose, effective_plan_only, "PREGRASP", result);
      check_cancel(goal_handle);

      if (!goal->execute_grasp_place) {
        result->success = true;
        result->final_state = effective_plan_only ? "PLANNED_PREGRASP" : "PREGRASP_READY";
        goal_handle->succeed(result);
        set_state(result->final_state, "");
        active_.store(false);
        return;
      }

      remove_object_collision();

      publish_feedback(goal_handle, "PLAN_APPROACH", target.age_s, 1.0);
      const double approach_fraction = plan_and_maybe_execute_cartesian(
        {pregrasp_pose, target.pose}, effective_plan_only, "APPROACH", result);
      result->min_cartesian_fraction =
        std::min(result->min_cartesian_fraction, static_cast<float>(approach_fraction));
      check_cancel(goal_handle);

      if (!effective_plan_only) {
        publish_feedback(goal_handle, "GRIPPER_CLOSE", target.age_s, approach_fraction);
        close_gripper();
        check_cancel(goal_handle);
      }

      geometry_msgs::msg::Pose retreat_pose = target.pose;
      retreat_pose.position.z += std::abs(retreat_z_offset_m_);
      publish_feedback(goal_handle, "PLAN_RETREAT", target.age_s, 1.0);
      const double retreat_fraction = plan_and_maybe_execute_cartesian(
        {target.pose, retreat_pose}, effective_plan_only, "RETREAT", result);
      result->min_cartesian_fraction =
        std::min(result->min_cartesian_fraction, static_cast<float>(retreat_fraction));
      check_cancel(goal_handle);

      geometry_msgs::msg::Pose place_pose = target.pose;
      if (place_xyz_m_.size() >= 3) {
        place_pose.position.x = place_xyz_m_[0];
        place_pose.position.y = place_xyz_m_[1];
        place_pose.position.z = place_xyz_m_[2];
      }
      publish_feedback(goal_handle, "PLAN_PLACE", target.age_s, 1.0);
      plan_and_maybe_execute_pose(place_pose, effective_plan_only, "PLACE", result);
      check_cancel(goal_handle);

      if (!effective_plan_only) {
        publish_feedback(goal_handle, "GRIPPER_OPEN", target.age_s, 1.0);
        open_gripper();
        check_cancel(goal_handle);
      }

      publish_feedback(goal_handle, "PLAN_RETURN_HOME", target.age_s, 1.0);
      plan_and_maybe_execute_joints(home_joints_rad(), effective_plan_only, "RETURN_HOME", result);

      result->success = true;
      result->final_state = effective_plan_only ? "PLANNED" : "DONE";
      goal_handle->succeed(result);
      set_state(result->final_state, "");
    } catch (const Cancelled &) {
      result->success = false;
      result->final_state = "CANCELLED";
      result->error = "cancelled";
      goal_handle->canceled(result);
      set_state("IDLE", "cancelled");
    } catch (const std::exception & exc) {
      result->success = false;
      result->final_state = "ERROR";
      result->error = exc.what();
      goal_handle->abort(result);
      set_state("ERROR", exc.what());
      RCLCPP_ERROR(node_->get_logger(), "motion goal failed: %s", exc.what());
    }
    active_.store(false);
  }

  struct Cancelled : public std::exception
  {
    const char * what() const noexcept override {return "cancelled";}
  };

  void check_cancel(const std::shared_ptr<ExecuteMotionPlanGoalHandle> & goal_handle)
  {
    if (cancel_requested_.load() || goal_handle->is_canceling()) {
      if (move_group_) {
        move_group_->stop();
      }
      throw Cancelled();
    }
  }

  void publish_feedback(
    const std::shared_ptr<ExecuteMotionPlanGoalHandle> & goal_handle,
    const std::string & phase,
    double target_age_s,
    double cartesian_fraction)
  {
    auto feedback = std::make_shared<ExecuteMotionPlan::Feedback>();
    feedback->phase = phase;
    feedback->target_age_s = static_cast<float>(target_age_s);
    feedback->cartesian_fraction = static_cast<float>(cartesian_fraction);
    goal_handle->publish_feedback(feedback);
    set_state(phase, "");
  }

  Target resolve_target(const ExecuteMotionPlan::Goal & goal)
  {
    if (!goal.use_latest_target) {
      if (!finite_pose(goal.target_pose)) {
        throw std::runtime_error("goal target_pose contains non-finite values");
      }
      Target target;
      target.pose = goal.target_pose;
      target.object_point = goal.object_point_base;
      target.frame_id = target_frame_;
      target.age_s = 0.0;
      return target;
    }

    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (!latest_grasp_target_.has_value()) {
      throw std::runtime_error("no /perception/grasp_target has been received");
    }
    const auto & msg = latest_grasp_target_.value();
    const double age_s = stamp_age_s(node_, msg.header);
    if (!msg.valid) {
      throw std::runtime_error("latest /perception/grasp_target is invalid");
    }
    if (age_s > max_target_age_s_) {
      throw std::runtime_error("latest /perception/grasp_target is stale");
    }
    if (!finite_pose(msg.target_pose_base)) {
      throw std::runtime_error("latest /perception/grasp_target pose contains non-finite values");
    }
    Target target;
    target.pose = msg.target_pose_base;
    target.object_point = msg.object_point_base;
    target.frame_id = msg.header.frame_id.empty() ? target_frame_ : msg.header.frame_id;
    target.age_s = age_s;
    return target;
  }

  geometry_msgs::msg::Pose transform_pose_to_planning_frame(
    const geometry_msgs::msg::Pose & pose,
    const std::string & source_frame)
  {
    if (source_frame.empty() || source_frame == planning_frame_ ||
      (assume_identity_base_frame_ && source_frame == target_frame_))
    {
      return pose;
    }

    geometry_msgs::msg::PoseStamped stamped;
    stamped.header.frame_id = source_frame;
    stamped.header.stamp = node_->now();
    stamped.pose = pose;
    geometry_msgs::msg::PoseStamped out;
    out = tf_buffer_.transform(stamped, planning_frame_, tf2::durationFromSec(0.2));
    return out.pose;
  }

  geometry_msgs::msg::Point transform_point_to_planning_frame(
    const geometry_msgs::msg::Point & point,
    const std::string & source_frame)
  {
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position = point;
    return transform_pose_to_planning_frame(pose, source_frame).position;
  }

  std::vector<double> home_joints_rad() const
  {
    std::vector<double> joints;
    joints.reserve(home_joints_deg_.size());
    for (const double value_deg : home_joints_deg_) {
      joints.push_back(value_deg * M_PI / 180.0);
    }
    return joints;
  }

  void plan_and_maybe_execute_joints(
    const std::vector<double> & joints,
    bool plan_only,
    const std::string & phase,
    std::shared_ptr<ExecuteMotionPlan::Result> & result)
  {
    move_group_->clearPoseTargets();
    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(joints)) {
      throw std::runtime_error(phase + " joint target rejected by MoveIt");
    }
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto code = move_group_->plan(plan);
    if (!code) {
      throw std::runtime_error(phase + " planning failed");
    }
    if (!plan_only) {
      const auto exec_code = move_group_->execute(plan);
      if (!exec_code) {
        throw std::runtime_error(phase + " execution failed");
      }
      result->executed = true;
    }
  }

  void plan_and_maybe_execute_pose(
    const geometry_msgs::msg::Pose & pose,
    bool plan_only,
    const std::string & phase,
    std::shared_ptr<ExecuteMotionPlan::Result> & result)
  {
    if (!finite_pose(pose)) {
      throw std::runtime_error(phase + " target pose contains non-finite values");
    }
    move_group_->clearPoseTargets();
    move_group_->setStartStateToCurrentState();
    if (!move_group_->setPoseTarget(pose, eef_link_)) {
      throw std::runtime_error(phase + " pose target rejected by MoveIt");
    }
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const auto code = move_group_->plan(plan);
    if (!code) {
      throw std::runtime_error(phase + " planning failed");
    }
    if (!plan_only) {
      const auto exec_code = move_group_->execute(plan);
      if (!exec_code) {
        throw std::runtime_error(phase + " execution failed");
      }
      result->executed = true;
    }
  }

  double plan_and_maybe_execute_cartesian(
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    bool plan_only,
    const std::string & phase,
    std::shared_ptr<ExecuteMotionPlan::Result> & result)
  {
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step_, jump_threshold_,
      trajectory);
    if (fraction < min_cartesian_fraction_) {
      throw std::runtime_error(
              phase + " Cartesian fraction below threshold: " +
              std::to_string(fraction));
    }
    if (!retime_cartesian_trajectory(trajectory)) {
      throw std::runtime_error(phase + " trajectory retiming failed");
    }
    if (!plan_only) {
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_ = trajectory;
      const auto exec_code = move_group_->execute(plan);
      if (!exec_code) {
        throw std::runtime_error(phase + " execution failed");
      }
      result->executed = true;
    }
    return fraction;
  }

  bool retime_cartesian_trajectory(moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    const auto current_state = move_group_->getCurrentState(1.0);
    if (!current_state) {
      return false;
    }
    const auto robot_model = move_group_->getRobotModel();
    robot_trajectory::RobotTrajectory robot_trajectory(robot_model, planning_group_);
    robot_trajectory.setRobotTrajectoryMsg(*current_state, trajectory);

    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(
        robot_trajectory, max_velocity_scaling_,
        max_acceleration_scaling_))
    {
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      if (!iptp.computeTimeStamps(
          robot_trajectory, max_velocity_scaling_,
          max_acceleration_scaling_))
      {
        return false;
      }
    }
    robot_trajectory.getRobotTrajectoryMsg(trajectory);
    return true;
  }

  void add_ground_plane()
  {
    moveit_msgs::msg::CollisionObject ground;
    ground.header.frame_id = planning_frame_;
    ground.id = kGroundCollisionId;

    shape_msgs::msg::Plane plane;
    plane.coef = {0.0, 0.0, 1.0, -ground_z_m_};

    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;

    ground.planes.push_back(plane);
    ground.plane_poses.push_back(pose);
    ground.operation = moveit_msgs::msg::CollisionObject::ADD;
    planning_scene_->applyCollisionObject(ground);
  }

  void add_or_update_object_collision(const geometry_msgs::msg::Point & fallback_center)
  {
    moveit_msgs::msg::CollisionObject object;
    object.header.frame_id = planning_frame_;
    object.id = kObjectCollisionId;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions.resize(3);

    geometry_msgs::msg::Pose object_pose;
    object_pose.orientation.w = 1.0;
    bool have_object = false;

    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      if (latest_cloud_bounds_.has_value()) {
        const auto & bounds = latest_cloud_bounds_.value();
        const double age = (node_->now() - bounds.stamp).seconds();
        if (bounds.valid && age <= object_cloud_max_age_s_) {
          object_pose.position = transform_point_to_planning_frame(bounds.center, bounds.frame_id);
          primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = bounds.size_x;
          primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = bounds.size_y;
          primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = bounds.size_z;
          have_object = true;
        }
      }
      if (!have_object && latest_object_state_.has_value()) {
        const auto & state = latest_object_state_.value();
        const double age = stamp_age_s(node_, state.header);
        if (state.valid && state.object_detected && age <= object_state_max_age_s_ &&
          finite_point(state.centroid_base))
        {
          const std::string frame =
            state.header.frame_id.empty() ? target_frame_ : state.header.frame_id;
          object_pose.position = transform_point_to_planning_frame(state.centroid_base, frame);
          apply_default_object_size(primitive);
          have_object = true;
        }
      }
    }

    if (!have_object && finite_point(fallback_center)) {
      object_pose.position = fallback_center;
      apply_default_object_size(primitive);
      have_object = true;
    }

    if (!have_object) {
      return;
    }

    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(object_pose);
    object.operation = moveit_msgs::msg::CollisionObject::ADD;
    planning_scene_->applyCollisionObject(object);
  }

  void apply_default_object_size(shape_msgs::msg::SolidPrimitive & primitive) const
  {
    primitive.dimensions.resize(3);
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] =
      default_object_size_m_.size() > 0 ? default_object_size_m_[0] : 0.10;
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] =
      default_object_size_m_.size() > 1 ? default_object_size_m_[1] : 0.10;
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] =
      default_object_size_m_.size() > 2 ? default_object_size_m_[2] : 0.12;
  }

  void remove_object_collision()
  {
    planning_scene_->removeCollisionObjects({kObjectCollisionId});
  }

  void ensure_gripper_connected()
  {
    if (!enable_gripper_) {
      return;
    }
    if (!gripper_) {
      gripper_ = std::make_unique<ur_rtde::RobotiqGripper>(robot_ip_, gripper_port_, false);
    }
    if (!gripper_->isConnected()) {
      gripper_->connect();
    }
    if (activate_gripper_on_connect_ && !gripper_->isActive()) {
      gripper_->activate();
    }
  }

  void open_gripper()
  {
    if (!enable_gripper_) {
      RCLCPP_INFO(node_->get_logger(), "gripper open skipped because enable_gripper=false");
      return;
    }
    ensure_gripper_connected();
    gripper_->open(
      gripper_open_speed_, gripper_open_force_,
      ur_rtde::RobotiqGripper::WAIT_FINISHED);
  }

  void close_gripper()
  {
    if (!enable_gripper_) {
      RCLCPP_INFO(node_->get_logger(), "gripper close skipped because enable_gripper=false");
      return;
    }
    ensure_gripper_connected();
    gripper_->close(
      gripper_close_speed_, gripper_close_force_,
      ur_rtde::RobotiqGripper::WAIT_FINISHED);
  }

  void handle_reset_home(
    const std::shared_ptr<icra_handover_interfaces::srv::ResetHome::Request>,
    std::shared_ptr<icra_handover_interfaces::srv::ResetHome::Response> response)
  {
    if (active_.load()) {
      response->accepted = false;
      response->state = current_state();
      response->error = "cannot reset while a motion action is active";
      return;
    }
    std::lock_guard<std::mutex> lock(move_group_mutex_);
    try {
      auto result = std::make_shared<ExecuteMotionPlan::Result>();
      plan_and_maybe_execute_joints(home_joints_rad(), plan_only_default_, "RESET_HOME", result);
      response->accepted = true;
      response->state = plan_only_default_ ? "PLANNED_HOME" : "IDLE";
      response->error = "";
      set_state(response->state, "");
    } catch (const std::exception & exc) {
      response->accepted = false;
      response->state = "ERROR";
      response->error = exc.what();
      set_state("ERROR", exc.what());
    }
  }

  void handle_emergency_stop(
    const std::shared_ptr<icra_handover_interfaces::srv::EmergencyStop::Request> request,
    std::shared_ptr<icra_handover_interfaces::srv::EmergencyStop::Response> response)
  {
    cancel_requested_.store(true);
    if (move_group_) {
      move_group_->stop();
    }
    set_state("STOPPING", "emergency stop requested: " + request->reason);
    response->accepted = true;
    response->state = current_state();
    response->error = "";
  }

  void publish_status()
  {
    icra_handover_interfaces::msg::RobotStatus msg;
    msg.header.stamp = node_->now();
    msg.header.frame_id = target_frame_;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      msg.state = state_;
      msg.last_error = last_error_;
      msg.last_command_type = last_command_type_;
    }
    msg.is_connected = true;
    msg.using_mock = false;
    msg.active_request = active_.load() ? "EXECUTE_MOTION_PLAN" : "";
    msg.active_request_id = active_.load() ? 1 : 0;
    msg.tcp_pose_base = last_tcp_pose_;
    msg.grasp_ok = false;
    msg.task_ready_epoch = 0;
    msg.reset_done_epoch = 0;
    msg.task_done_epoch = 0;

    std::unique_lock<std::mutex> lock(move_group_mutex_, std::try_to_lock);
    if (lock.owns_lock() && move_group_) {
      try {
        const auto current_pose = move_group_->getCurrentPose(eef_link_).pose;
        msg.tcp_pose_base = current_pose;
        last_tcp_pose_ = current_pose;
      } catch (const std::exception &) {
        // Keep the previous pose if MoveIt state is temporarily unavailable.
      }
    }
    status_pub_->publish(msg);
  }

  void set_state(const std::string & state, const std::string & error)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = state;
    last_error_ = error;
    last_command_type_ = state;
  }

  std::string current_state() const
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
  }

  rclcpp::Node::SharedPtr node_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string planning_group_;
  std::string eef_link_;
  std::string target_frame_;
  std::string planning_frame_;
  std::string robot_ip_;
  int gripper_port_{63352};
  bool enable_gripper_{true};
  bool activate_gripper_on_connect_{true};
  bool assume_identity_base_frame_{true};
  bool plan_only_default_{true};
  double max_velocity_scaling_{0.05};
  double max_acceleration_scaling_{0.05};
  double min_cartesian_fraction_{0.95};
  double planning_time_s_{5.0};
  double max_target_age_s_{0.5};
  double object_cloud_max_age_s_{0.5};
  double object_state_max_age_s_{0.5};
  double eef_step_{0.005};
  double jump_threshold_{0.0};
  double pregrasp_offset_m_{0.10};
  double retreat_z_offset_m_{0.08};
  double ground_z_m_{-0.01};
  double ground_size_m_{3.0};
  double status_rate_hz_{5.0};
  double gripper_open_speed_{1.0};
  double gripper_open_force_{1.0};
  double gripper_close_speed_{1.0};
  double gripper_close_force_{1.0};
  std::vector<double> approach_axis_;
  std::vector<double> home_joints_deg_;
  std::vector<double> place_xyz_m_;
  std::vector<double> default_object_size_m_;

  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_;
  std::unique_ptr<ur_rtde::RobotiqGripper> gripper_;

  rclcpp::Subscription<icra_handover_interfaces::msg::GraspTarget>::SharedPtr grasp_sub_;
  rclcpp::Subscription<icra_handover_interfaces::msg::ObjectState>::SharedPtr object_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr fitted_cloud_sub_;
  rclcpp::Publisher<icra_handover_interfaces::msg::RobotStatus>::SharedPtr status_pub_;
  rclcpp::Service<icra_handover_interfaces::srv::ResetHome>::SharedPtr reset_srv_;
  rclcpp::Service<icra_handover_interfaces::srv::EmergencyStop>::SharedPtr emergency_stop_srv_;
  rclcpp_action::Server<ExecuteMotionPlan>::SharedPtr action_server_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  std::mutex move_group_mutex_;
  std::mutex cache_mutex_;
  mutable std::mutex state_mutex_;
  std::optional<icra_handover_interfaces::msg::GraspTarget> latest_grasp_target_;
  std::optional<icra_handover_interfaces::msg::ObjectState> latest_object_state_;
  std::optional<CloudBounds> latest_cloud_bounds_;

  std::atomic_bool active_{false};
  std::atomic_bool cancel_requested_{false};
  std::string state_{"IDLE"};
  std::string last_error_;
  std::string last_command_type_;
  geometry_msgs::msg::Pose last_tcp_pose_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("moveit_handover_node", options);
  auto app = std::make_shared<MoveItHandoverNode>(node);
  (void)app;

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
