/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cmath>

#include <ros/ros.h>

#include "lee_position_controller_node.h"

#include "Eigen/src/Core/Matrix.h"
#include "rotors_control/parameters_ros.h"
#include "rotors_control/common.h"

#include "airsim_ros/RotorPWM.h"

namespace rotors_control {

namespace {
Eigen::Matrix4d TransformFromPose(const geometry_msgs::Pose& pose) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  Eigen::Quaterniond q(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z);
  transform.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  transform.block<3, 1>(0, 3) << pose.position.x, pose.position.y, pose.position.z;
  return transform;
}
}

LeePositionControllerNode::LeePositionControllerNode(
  const ros::NodeHandle& nh, const ros::NodeHandle& private_nh)
  :nh_(nh),
   private_nh_(private_nh),
   use_airsim_frame_adapter_(false),
   got_initial_pose_(false),
   command_pose_topic_("command/pose"),
   command_trajectory_topic_("command/trajectory"),
   odometry_topic_("/eskf_odom"),
   initial_pose_topic_("/airsim_node/initial_pose"),
   motor_topic_("/airsim_node/drone_1/rotor_pwm_cmd"),
   disturbance_estimate_topic_("eso_disturbance"),
   torque_disturbance_estimate_topic_("eso_torque_disturbance"),
   last_odometry_stamp_(0.0),
   t_world_flu_initial_(Eigen::Matrix4d::Identity()),
   t_world_flu_initial_inv_(Eigen::Matrix4d::Identity()),
   t_flu_ned_(Eigen::Matrix4d::Identity()){
  t_flu_ned_(1, 1) = -1.0;
  t_flu_ned_(2, 2) = -1.0;

  InitializeParams();

  cmd_pose_sub_ = nh_.subscribe(
      command_pose_topic_, 1,
      &LeePositionControllerNode::CommandPoseCallback, this);

  cmd_multi_dof_joint_trajectory_sub_ = nh_.subscribe(
      command_trajectory_topic_, 1,
      &LeePositionControllerNode::MultiDofJointTrajectoryCallback, this);

  //TODO: (pxx)odometry estimate, SLAM
  odometry_sub_ = nh_.subscribe(odometry_topic_, 1,
  // odometry_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 1,
                               &LeePositionControllerNode::OdometryCallback, this);

  if (use_airsim_frame_adapter_) {
    initial_pose_sub_ = nh_.subscribe(
        initial_pose_topic_, 1,
        &LeePositionControllerNode::InitialPoseCallback, this);
  }

  //TODO: (pxx)msg type = airsim_ros::RotorPWM
  motor_velocity_reference_pub_ = nh_.advertise<airsim_ros::RotorPWM>(
      motor_topic_, 1);
  disturbance_estimate_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>(
      disturbance_estimate_topic_, 1);
  torque_disturbance_estimate_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>(
      torque_disturbance_estimate_topic_, 1);

  command_timer_ = nh_.createTimer(ros::Duration(0), &LeePositionControllerNode::TimedCommandCallback, this,
                                  true, false);
}

LeePositionControllerNode::~LeePositionControllerNode() { }

void LeePositionControllerNode::InitializeParams() {

  GetRosParameter(private_nh_, "command_pose_topic",
                  command_pose_topic_,
                  &command_pose_topic_);
  GetRosParameter(private_nh_, "command_trajectory_topic",
                  command_trajectory_topic_,
                  &command_trajectory_topic_);
  GetRosParameter(private_nh_, "odometry_topic",
                  odometry_topic_,
                  &odometry_topic_);
  GetRosParameter(private_nh_, "initial_pose_topic",
                  initial_pose_topic_,
                  &initial_pose_topic_);
  GetRosParameter(private_nh_, "motor_topic",
                  motor_topic_,
                  &motor_topic_);
  GetRosParameter(private_nh_, "disturbance_observer_topic",
                  disturbance_estimate_topic_,
                  &disturbance_estimate_topic_);
  GetRosParameter(private_nh_, "disturbance_observer_torque_topic",
                  torque_disturbance_estimate_topic_,
                  &torque_disturbance_estimate_topic_);

  // Read parameters from rosparam.
  GetRosParameter(private_nh_, "position_gain/x",
                  lee_position_controller_.controller_parameters_.position_gain_.x(),
                  &lee_position_controller_.controller_parameters_.position_gain_.x());
  GetRosParameter(private_nh_, "position_gain/y",
                  lee_position_controller_.controller_parameters_.position_gain_.y(),
                  &lee_position_controller_.controller_parameters_.position_gain_.y());
  GetRosParameter(private_nh_, "position_gain/z",
                  lee_position_controller_.controller_parameters_.position_gain_.z(),
                  &lee_position_controller_.controller_parameters_.position_gain_.z());
  GetRosParameter(private_nh_, "velocity_gain/x",
                  lee_position_controller_.controller_parameters_.velocity_gain_.x(),
                  &lee_position_controller_.controller_parameters_.velocity_gain_.x());
  GetRosParameter(private_nh_, "velocity_gain/y",
                  lee_position_controller_.controller_parameters_.velocity_gain_.y(),
                  &lee_position_controller_.controller_parameters_.velocity_gain_.y());
  GetRosParameter(private_nh_, "velocity_gain/z",
                  lee_position_controller_.controller_parameters_.velocity_gain_.z(),
                  &lee_position_controller_.controller_parameters_.velocity_gain_.z());
  GetRosParameter(private_nh_, "attitude_gain/x",
                  lee_position_controller_.controller_parameters_.attitude_gain_.x(),
                  &lee_position_controller_.controller_parameters_.attitude_gain_.x());
  GetRosParameter(private_nh_, "attitude_gain/y",
                  lee_position_controller_.controller_parameters_.attitude_gain_.y(),
                  &lee_position_controller_.controller_parameters_.attitude_gain_.y());
  GetRosParameter(private_nh_, "attitude_gain/z",
                  lee_position_controller_.controller_parameters_.attitude_gain_.z(),
                  &lee_position_controller_.controller_parameters_.attitude_gain_.z());
  GetRosParameter(private_nh_, "angular_rate_gain/x",
                  lee_position_controller_.controller_parameters_.angular_rate_gain_.x(),
                  &lee_position_controller_.controller_parameters_.angular_rate_gain_.x());
  GetRosParameter(private_nh_, "angular_rate_gain/y",
                  lee_position_controller_.controller_parameters_.angular_rate_gain_.y(),
                  &lee_position_controller_.controller_parameters_.angular_rate_gain_.y());
  GetRosParameter(private_nh_, "angular_rate_gain/z",
                  lee_position_controller_.controller_parameters_.angular_rate_gain_.z(),
                  &lee_position_controller_.controller_parameters_.angular_rate_gain_.z());
  GetRosParameter(private_nh_, "odometry_velocity_in_body_frame",
                  lee_position_controller_.odometry_velocity_in_body_frame_,
                  &lee_position_controller_.odometry_velocity_in_body_frame_);
  GetRosParameter(private_nh_, "use_airsim_frame_adapter",
                  use_airsim_frame_adapter_,
                  &use_airsim_frame_adapter_);
  GetRosParameter(private_nh_, "use_airsim_pwm_mixer",
                  lee_position_controller_.use_airsim_pwm_mixer_,
                  &lee_position_controller_.use_airsim_pwm_mixer_);
  GetRosParameter(private_nh_, "enable_disturbance_observer",
                  lee_position_controller_.enable_disturbance_observer_,
                  &lee_position_controller_.enable_disturbance_observer_);
  GetRosParameter(private_nh_, "enable_attitude_disturbance_observer",
                  lee_position_controller_.enable_attitude_disturbance_observer_,
                  &lee_position_controller_.enable_attitude_disturbance_observer_);
  GetRosParameter(private_nh_, "disturbance_observer_position_gain/l1",
                  lee_position_controller_.disturbance_observer_position_gain_.x(),
                  &lee_position_controller_.disturbance_observer_position_gain_.x());
  GetRosParameter(private_nh_, "disturbance_observer_position_gain/l2",
                  lee_position_controller_.disturbance_observer_position_gain_.y(),
                  &lee_position_controller_.disturbance_observer_position_gain_.y());
  GetRosParameter(private_nh_, "disturbance_observer_position_gain/l3",
                  lee_position_controller_.disturbance_observer_position_gain_.z(),
                  &lee_position_controller_.disturbance_observer_position_gain_.z());
  GetRosParameter(private_nh_, "attitude_disturbance_observer_gain/l1",
                  lee_position_controller_.attitude_disturbance_observer_gain_.x(),
                  &lee_position_controller_.attitude_disturbance_observer_gain_.x());
  GetRosParameter(private_nh_, "attitude_disturbance_observer_gain/l2",
                  lee_position_controller_.attitude_disturbance_observer_gain_.y(),
                  &lee_position_controller_.attitude_disturbance_observer_gain_.y());
  GetRosParameter(private_nh_, "attitude_disturbance_observer_gain/l3",
                  lee_position_controller_.attitude_disturbance_observer_gain_.z(),
                  &lee_position_controller_.attitude_disturbance_observer_gain_.z());
  GetRosParameter(private_nh_, "disturbance_observer_max_acc",
                  lee_position_controller_.disturbance_observer_max_acc_,
                  &lee_position_controller_.disturbance_observer_max_acc_);
  GetRosParameter(private_nh_, "disturbance_observer_compensation_gain",
                  lee_position_controller_.disturbance_observer_compensation_gain_,
                  &lee_position_controller_.disturbance_observer_compensation_gain_);
  double disturbance_observer_freeze_tilt_deg =
      lee_position_controller_.disturbance_observer_freeze_tilt_rad_ * 180.0 / M_PI;
  GetRosParameter(private_nh_, "disturbance_observer_freeze_tilt_deg",
                  disturbance_observer_freeze_tilt_deg,
                  &disturbance_observer_freeze_tilt_deg);
  lee_position_controller_.disturbance_observer_freeze_tilt_rad_ =
      std::max(0.0, disturbance_observer_freeze_tilt_deg) * M_PI / 180.0;
  GetRosParameter(private_nh_, "disturbance_observer_decay_rate",
                  lee_position_controller_.disturbance_observer_decay_rate_,
                  &lee_position_controller_.disturbance_observer_decay_rate_);
  GetRosParameter(private_nh_, "attitude_disturbance_observer_max_acc",
                  lee_position_controller_.attitude_disturbance_observer_max_acc_,
                  &lee_position_controller_.attitude_disturbance_observer_max_acc_);
  GetRosParameter(private_nh_, "attitude_disturbance_observer_compensation_gain",
                  lee_position_controller_.attitude_disturbance_observer_compensation_gain_,
                  &lee_position_controller_.attitude_disturbance_observer_compensation_gain_);
  double attitude_disturbance_observer_freeze_tilt_deg =
      lee_position_controller_.attitude_disturbance_observer_freeze_tilt_rad_ * 180.0 / M_PI;
  GetRosParameter(private_nh_, "attitude_disturbance_observer_freeze_tilt_deg",
                  attitude_disturbance_observer_freeze_tilt_deg,
                  &attitude_disturbance_observer_freeze_tilt_deg);
  lee_position_controller_.attitude_disturbance_observer_freeze_tilt_rad_ =
      std::max(0.0, attitude_disturbance_observer_freeze_tilt_deg) * M_PI / 180.0;
  GetRosParameter(private_nh_, "attitude_disturbance_observer_decay_rate",
                  lee_position_controller_.attitude_disturbance_observer_decay_rate_,
                  &lee_position_controller_.attitude_disturbance_observer_decay_rate_);
  GetRosParameter(private_nh_, "max_rotor_velocity",
                  lee_position_controller_.max_rotor_velocity_,
                  &lee_position_controller_.max_rotor_velocity_);
  GetRosParameter(private_nh_, "min_rotor_pwm",
                  lee_position_controller_.min_rotor_pwm_,
                  &lee_position_controller_.min_rotor_pwm_);
  GetRosParameter(private_nh_, "max_rotor_pwm",
                  lee_position_controller_.max_rotor_pwm_,
                  &lee_position_controller_.max_rotor_pwm_);
  GetRosParameter(private_nh_, "enable_pwm_desaturation",
                  lee_position_controller_.enable_pwm_desaturation_,
                  &lee_position_controller_.enable_pwm_desaturation_);
  GetRosParameter(private_nh_, "enable_attitude_recovery_pwm_floor",
                  lee_position_controller_.enable_attitude_recovery_pwm_floor_,
                  &lee_position_controller_.enable_attitude_recovery_pwm_floor_);
  double attitude_recovery_pwm_min_tilt_deg =
      lee_position_controller_.attitude_recovery_pwm_min_tilt_rad_ * 180.0 / M_PI;
  GetRosParameter(private_nh_, "attitude_recovery_pwm_min_tilt_deg",
                  attitude_recovery_pwm_min_tilt_deg,
                  &attitude_recovery_pwm_min_tilt_deg);
  lee_position_controller_.attitude_recovery_pwm_min_tilt_rad_ =
      std::max(0.0, attitude_recovery_pwm_min_tilt_deg) * M_PI / 180.0;
  double attitude_recovery_pwm_full_tilt_deg =
      lee_position_controller_.attitude_recovery_pwm_full_tilt_rad_ * 180.0 / M_PI;
  GetRosParameter(private_nh_, "attitude_recovery_pwm_full_tilt_deg",
                  attitude_recovery_pwm_full_tilt_deg,
                  &attitude_recovery_pwm_full_tilt_deg);
  lee_position_controller_.attitude_recovery_pwm_full_tilt_rad_ =
      std::max(0.0, attitude_recovery_pwm_full_tilt_deg) * M_PI / 180.0;
  GetRosParameter(private_nh_, "attitude_recovery_min_pwm",
                  lee_position_controller_.attitude_recovery_min_pwm_,
                  &lee_position_controller_.attitude_recovery_min_pwm_);
  double max_tilt_angle_deg = lee_position_controller_.max_tilt_angle_rad_ * 180.0 / M_PI;
  GetRosParameter(private_nh_, "max_tilt_angle_deg",
                  max_tilt_angle_deg,
                  &max_tilt_angle_deg);
  lee_position_controller_.max_tilt_angle_rad_ =
      std::max(0.0, max_tilt_angle_deg) * M_PI / 180.0;
  GetRosParameter(private_nh_, "torque_scale/x",
                  lee_position_controller_.torque_scale_.x(),
                  &lee_position_controller_.torque_scale_.x());
  GetRosParameter(private_nh_, "torque_scale/y",
                  lee_position_controller_.torque_scale_.y(),
                  &lee_position_controller_.torque_scale_.y());
  GetRosParameter(private_nh_, "torque_scale/z",
                  lee_position_controller_.torque_scale_.z(),
                  &lee_position_controller_.torque_scale_.z());
  GetVehicleParameters(private_nh_, &lee_position_controller_.vehicle_parameters_);
  lee_position_controller_.InitializeParameters();
  ROS_INFO(
      "Lee control params: pos_eso enabled=%s max_acc=%.2f comp_gain=%.2f freeze_tilt=%.1f "
      "att_eso enabled=%s max_acc=%.2f comp_gain=%.2f freeze_tilt=%.1f "
      "pwm_desaturation=%s attitude_pwm_floor=%s",
      lee_position_controller_.enable_disturbance_observer_ ? "true" : "false",
      lee_position_controller_.disturbance_observer_max_acc_,
      lee_position_controller_.disturbance_observer_compensation_gain_,
      lee_position_controller_.disturbance_observer_freeze_tilt_rad_ * 180.0 / M_PI,
      lee_position_controller_.enable_attitude_disturbance_observer_ ? "true" : "false",
      lee_position_controller_.attitude_disturbance_observer_max_acc_,
      lee_position_controller_.attitude_disturbance_observer_compensation_gain_,
      lee_position_controller_.attitude_disturbance_observer_freeze_tilt_rad_ * 180.0 / M_PI,
      lee_position_controller_.enable_pwm_desaturation_ ? "true" : "false",
      lee_position_controller_.enable_attitude_recovery_pwm_floor_ ? "true" : "false");
}

void LeePositionControllerNode::Publish() {
}

void LeePositionControllerNode::CommandPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {
  // Clear all pending commands.
  command_timer_.stop();
  commands_.clear();
  command_waiting_times_.clear();

  EigenTrajectoryPoint eigen_reference;
  eigenTrajectoryPointFromPoseMsg(*pose_msg, &eigen_reference);
  if (use_airsim_frame_adapter_ &&
      !TransformTrajectoryPointFromAirsim(&eigen_reference)) {
    ROS_WARN_THROTTLE(1.0, "Waiting for /airsim_node/initial_pose before accepting pose commands.");
    return;
  }
  commands_.push_front(eigen_reference);

  lee_position_controller_.SetTrajectoryPoint(commands_.front());
  commands_.pop_front();
}

void LeePositionControllerNode::MultiDofJointTrajectoryCallback(
    const trajectory_msgs::MultiDOFJointTrajectoryConstPtr& msg) {
  // Clear all pending commands.
  command_timer_.stop();
  commands_.clear();
  command_waiting_times_.clear();

  const size_t n_commands = msg->points.size();

  if(n_commands < 1){
    ROS_WARN_STREAM("Got MultiDOFJointTrajectory message, but message has no points.");
    return;
  }

  EigenTrajectoryPoint eigen_reference;
  eigenTrajectoryPointFromMsg(msg->points.front(), &eigen_reference);
  if (use_airsim_frame_adapter_ &&
      !TransformTrajectoryPointFromAirsim(&eigen_reference)) {
    ROS_WARN_THROTTLE(1.0, "Waiting for /airsim_node/initial_pose before accepting trajectory commands.");
    return;
  }
  commands_.push_front(eigen_reference);

  for (size_t i = 1; i < n_commands; ++i) {
    const trajectory_msgs::MultiDOFJointTrajectoryPoint& reference_before = msg->points[i-1];
    const trajectory_msgs::MultiDOFJointTrajectoryPoint& current_reference = msg->points[i];

    eigenTrajectoryPointFromMsg(current_reference, &eigen_reference);
    if (use_airsim_frame_adapter_ &&
        !TransformTrajectoryPointFromAirsim(&eigen_reference)) {
      return;
    }

    commands_.push_back(eigen_reference);
    command_waiting_times_.push_back(current_reference.time_from_start - reference_before.time_from_start);
  }

  // We can trigger the first command immediately.
  lee_position_controller_.SetTrajectoryPoint(commands_.front());
  commands_.pop_front();

  if (n_commands > 1) {
    command_timer_.setPeriod(command_waiting_times_.front());
    command_waiting_times_.pop_front();
    command_timer_.start();
  }
}

void LeePositionControllerNode::TimedCommandCallback(const ros::TimerEvent& e) {

  if(commands_.empty()){
    ROS_WARN("Commands empty, this should not happen here");
    return;
  }

  const EigenTrajectoryPoint eigen_reference = commands_.front();
  lee_position_controller_.SetTrajectoryPoint(commands_.front());
  commands_.pop_front();
  command_timer_.stop();
  if(!command_waiting_times_.empty()){
    command_timer_.setPeriod(command_waiting_times_.front());
    command_waiting_times_.pop_front();
    command_timer_.start();
  }
}

void LeePositionControllerNode::OdometryCallback(const nav_msgs::OdometryConstPtr& odometry_msg) {

  ROS_INFO_ONCE("LeePositionController got first odometry message.");

  EigenOdometry odometry;
  eigenOdometryFromMsg(odometry_msg, &odometry);
  if (use_airsim_frame_adapter_ &&
      !TransformOdometryFromAirsim(&odometry)) {
    ROS_WARN_THROTTLE(1.0, "Waiting for /airsim_node/initial_pose before publishing rotor PWM.");
    return;
  }
  double dt = 0.0;
  if (!last_odometry_stamp_.isZero()) {
    dt = (odometry_msg->header.stamp - last_odometry_stamp_).toSec();
  }
  last_odometry_stamp_ = odometry_msg->header.stamp;
  lee_position_controller_.SetOdometry(odometry, dt);

  Eigen::VectorXd ref_rotor_velocities;
  lee_position_controller_.CalculateRotorVelocities(&ref_rotor_velocities);

  //TODO: (pxx)use airsim::RotorPWM and transform angular_velocities to pwm value
  // 0: Right Front, 1: Left Back, 2: Left Front, 3: Right Back
  airsim_ros::RotorPWM actuator_msg;
  actuator_msg.header.stamp = odometry_msg->header.stamp;
  actuator_msg.header.frame_id = odometry_msg->child_frame_id.empty()
      ? "drone_1"
      : odometry_msg->child_frame_id;
  actuator_msg.rotorPWM0 = ref_rotor_velocities[0];
  actuator_msg.rotorPWM1 = ref_rotor_velocities[1];
  actuator_msg.rotorPWM2 = ref_rotor_velocities[2];
  actuator_msg.rotorPWM3 = ref_rotor_velocities[3];

  // mav_msgs::ActuatorsPtr actuator_msg(new mav_msgs::Actuators);
  // actuator_msg->angular_velocities.clear();
  // for (int i = 0; i < ref_rotor_velocities.size(); i++)
  //   actuator_msg->angular_velocities.push_back(ref_rotor_velocities[i]);
  // actuator_msg->header.stamp = odometry_msg->header.stamp;

  motor_velocity_reference_pub_.publish(actuator_msg);

  const Eigen::Vector3d disturbance =
      lee_position_controller_.GetDisturbanceEstimate();
  geometry_msgs::Vector3Stamped disturbance_msg;
  disturbance_msg.header.stamp = odometry_msg->header.stamp;
  disturbance_msg.header.frame_id = odometry_msg->child_frame_id.empty()
      ? "drone_1"
      : odometry_msg->child_frame_id;
  disturbance_msg.vector.x = disturbance.x();
  disturbance_msg.vector.y = disturbance.y();
  disturbance_msg.vector.z = disturbance.z();
  disturbance_estimate_pub_.publish(disturbance_msg);

  const Eigen::Vector3d torque_disturbance =
      lee_position_controller_.GetTorqueDisturbanceEstimate();
  geometry_msgs::Vector3Stamped torque_msg;
  torque_msg.header = disturbance_msg.header;
  torque_msg.vector.x = torque_disturbance.x();
  torque_msg.vector.y = torque_disturbance.y();
  torque_msg.vector.z = torque_disturbance.z();
  torque_disturbance_estimate_pub_.publish(torque_msg);

  const double pwm_min = ref_rotor_velocities.minCoeff();
  const double pwm_max = ref_rotor_velocities.maxCoeff();
  const double recovery_pwm_floor =
      lee_position_controller_.GetAttitudeRecoveryPwmFloor();
  const double tilt_deg =
      lee_position_controller_.GetCurrentTiltAngleRadEstimate() * 180.0 / M_PI;
  if (recovery_pwm_floor > lee_position_controller_.min_rotor_pwm_ + 1.0e-4) {
    ROS_WARN_THROTTLE(
        0.2,
        "Lee attitude recovery PWM floor active: tilt=%.1f floor=%.3f",
        tilt_deg,
        recovery_pwm_floor);
  }
  if (pwm_max >= lee_position_controller_.max_rotor_pwm_ - 0.02 ||
      pwm_min <= lee_position_controller_.min_rotor_pwm_ + 0.02) {
    ROS_WARN_THROTTLE(
        0.5,
        "Lee rotor PWM near saturation: min=%.3f max=%.3f limits=[%.3f, %.3f]",
        pwm_min,
        pwm_max,
        lee_position_controller_.min_rotor_pwm_,
        lee_position_controller_.max_rotor_pwm_);
  }
  ROS_INFO_THROTTLE(
      1.0,
      "Lee ESO disturbance=(%.2f, %.2f, %.2f) torque=(%.2f, %.2f, %.2f) enabled=%s attitude_enabled=%s",
      disturbance.x(),
      disturbance.y(),
      disturbance.z(),
      torque_disturbance.x(),
      torque_disturbance.y(),
      torque_disturbance.z(),
      lee_position_controller_.enable_disturbance_observer_ ? "true" : "false",
      lee_position_controller_.enable_attitude_disturbance_observer_ ? "true" : "false");

  const bool hard_saturation =
      pwm_max >= lee_position_controller_.max_rotor_pwm_ - 0.02 ||
      (pwm_min <= lee_position_controller_.min_rotor_pwm_ + 0.02 && pwm_max > 0.05);
  const bool pwm_desaturated =
      lee_position_controller_.GetLastPwmDesaturated();
  if (pwm_desaturated) {
    ROS_WARN_THROTTLE(
        0.2,
        "Lee PWM desaturation active: preserving motor differential under saturation");
  }
  if (tilt_deg > 25.0 || hard_saturation || recovery_pwm_floor > 1.0e-4 ||
      pwm_desaturated) {
    const Eigen::Vector3d pos_err =
        lee_position_controller_.GetLastPositionError();
    const Eigen::Vector3d vel_err =
        lee_position_controller_.GetLastVelocityError();
    const Eigen::Vector3d fb_nom =
        lee_position_controller_.GetLastNominalFeedbackAcceleration();
    const Eigen::Vector3d fb =
        lee_position_controller_.GetLastFeedbackAcceleration();
    const Eigen::Vector3d acc =
        lee_position_controller_.GetLastDesiredAcceleration();
    const Eigen::Vector3d ang_acc =
        lee_position_controller_.GetLastAngularAcceleration();
    const Eigen::Vector3d torque_cmd =
        lee_position_controller_.GetLastTorqueCommand();
    const Eigen::Vector3d att_err =
        lee_position_controller_.GetLastAttitudeError();
    const Eigen::Vector3d rate_err =
        lee_position_controller_.GetLastAngularRateError();
    const Eigen::Vector3d omega =
        lee_position_controller_.GetLastAngularVelocity();
    const Eigen::Vector4d raw_pwm =
        lee_position_controller_.GetLastRawRotorPwm();
    const Eigen::Vector4d final_pwm =
        lee_position_controller_.GetLastFinalRotorPwm();
    ROS_WARN_THROTTLE(
        0.2,
        "Lee diag: tilt=%.1f thrust=%.3f floor=%.3f desat=%s pos_err=(%.2f,%.2f,%.2f) vel_err=(%.2f,%.2f,%.2f) fb_nom=(%.2f,%.2f,%.2f) fb=(%.2f,%.2f,%.2f) acc=(%.2f,%.2f,%.2f) omega=(%.2f,%.2f,%.2f) att_err=(%.2f,%.2f,%.2f) rate_err=(%.2f,%.2f,%.2f) ang_acc=(%.2f,%.2f,%.2f) torque=(%.3f,%.3f,%.3f) raw_pwm=(%.3f,%.3f,%.3f,%.3f) pwm=(%.3f,%.3f,%.3f,%.3f)",
        tilt_deg,
        lee_position_controller_.GetLastThrustCommand(),
        recovery_pwm_floor,
        pwm_desaturated ? "true" : "false",
        pos_err.x(), pos_err.y(), pos_err.z(),
        vel_err.x(), vel_err.y(), vel_err.z(),
        fb_nom.x(), fb_nom.y(), fb_nom.z(),
        fb.x(), fb.y(), fb.z(),
        acc.x(), acc.y(), acc.z(),
        omega.x(), omega.y(), omega.z(),
        att_err.x(), att_err.y(), att_err.z(),
        rate_err.x(), rate_err.y(), rate_err.z(),
        ang_acc.x(), ang_acc.y(), ang_acc.z(),
        torque_cmd.x(), torque_cmd.y(), torque_cmd.z(),
        raw_pwm[0], raw_pwm[1], raw_pwm[2], raw_pwm[3],
        final_pwm[0], final_pwm[1], final_pwm[2], final_pwm[3]);
  }
}

void LeePositionControllerNode::InitialPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {
  const Eigen::Matrix4d t_world_ned_initial = TransformFromPose(pose_msg->pose);
  t_world_flu_initial_ = t_flu_ned_ * t_world_ned_initial * t_flu_ned_;
  t_world_flu_initial_inv_ = t_world_flu_initial_.inverse();
  got_initial_pose_ = true;
  ROS_INFO_ONCE("LeePositionController got AirSim initial pose for NED->FLU frame adapter.");
}

bool LeePositionControllerNode::TransformTrajectoryPointFromAirsim(
    EigenTrajectoryPoint* trajectory_point) const {
  if (!got_initial_pose_) {
    return false;
  }

  Eigen::Matrix4d t_world_ned_command = Eigen::Matrix4d::Identity();
  t_world_ned_command.block<3, 3>(0, 0) =
      trajectory_point->orientation_W_B.normalized().toRotationMatrix();
  t_world_ned_command.block<3, 1>(0, 3) = trajectory_point->position_W;

  const Eigen::Matrix4d t_world_flu_command =
      t_flu_ned_ * t_world_ned_command * t_flu_ned_;
  const Eigen::Matrix4d t_initial_flu_command =
      t_world_flu_initial_inv_ * t_world_flu_command;

  const Eigen::Matrix3d r_initial_world_flu =
      t_world_flu_initial_inv_.block<3, 3>(0, 0);
  const Eigen::Matrix3d r_flu_ned = t_flu_ned_.block<3, 3>(0, 0);

  trajectory_point->position_W = t_initial_flu_command.block<3, 1>(0, 3);
  trajectory_point->orientation_W_B =
      Eigen::Quaterniond(t_initial_flu_command.block<3, 3>(0, 0));
  trajectory_point->velocity_W =
      r_initial_world_flu * r_flu_ned * trajectory_point->velocity_W;
  trajectory_point->acceleration_W =
      r_initial_world_flu * r_flu_ned * trajectory_point->acceleration_W;
  trajectory_point->angular_velocity_W =
      r_initial_world_flu * r_flu_ned * trajectory_point->angular_velocity_W;
  return true;
}

bool LeePositionControllerNode::TransformOdometryFromAirsim(
    EigenOdometry* odometry) const {
  if (!got_initial_pose_) {
    return false;
  }

  Eigen::Matrix4d t_world_ned_body = Eigen::Matrix4d::Identity();
  t_world_ned_body.block<3, 3>(0, 0) =
      odometry->orientation.normalized().toRotationMatrix();
  t_world_ned_body.block<3, 1>(0, 3) = odometry->position;

  const Eigen::Matrix4d t_world_flu_body =
      t_flu_ned_ * t_world_ned_body * t_flu_ned_;
  const Eigen::Matrix4d t_initial_flu_body =
      t_world_flu_initial_inv_ * t_world_flu_body;

  const Eigen::Matrix3d r_initial_world_flu =
      t_world_flu_initial_inv_.block<3, 3>(0, 0);
  const Eigen::Matrix3d r_flu_ned = t_flu_ned_.block<3, 3>(0, 0);

  odometry->position = t_initial_flu_body.block<3, 1>(0, 3);
  odometry->orientation =
      Eigen::Quaterniond(t_initial_flu_body.block<3, 3>(0, 0));
  odometry->velocity = r_initial_world_flu * r_flu_ned * odometry->velocity;
  odometry->angular_velocity = r_flu_ned * odometry->angular_velocity;
  return true;
}

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "lee_position_controller_node");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  rotors_control::LeePositionControllerNode lee_position_controller_node(nh, private_nh);

  ros::spin();

  return 0;
}
