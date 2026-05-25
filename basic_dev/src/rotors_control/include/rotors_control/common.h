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

#ifndef INCLUDE_ROTORS_CONTROL_COMMON_H_
#define INCLUDE_ROTORS_CONTROL_COMMON_H_

#include <assert.h>
#include <deque>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>

#include "rotors_control/parameters.h"

namespace rotors_control {

// Default values.
static const std::string kDefaultNamespace = "";
static const std::string kDefaultCommandMotorSpeedTopic = "command/motor_speed";
static const std::string kDefaultCommandMultiDofJointTrajectoryTopic = "command/trajectory";
static const std::string kDefaultCommandRollPitchYawrateThrustTopic = "command/roll_pitch_yawrate_thrust";
static const std::string kDefaultImuTopic = "imu";
static const std::string kDefaultOdometryTopic = "odometry";

struct EigenOdometry {
  EigenOdometry()
      : position(0.0, 0.0, 0.0),
        orientation(Eigen::Quaterniond::Identity()),
        velocity(0.0, 0.0, 0.0),
        angular_velocity(0.0, 0.0, 0.0) {};

  EigenOdometry(const Eigen::Vector3d& _position,
                const Eigen::Quaterniond& _orientation,
                const Eigen::Vector3d& _velocity,
                const Eigen::Vector3d& _angular_velocity) {
    position = _position;
    orientation = _orientation;
    velocity = _velocity;
    angular_velocity = _angular_velocity;
  };

  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  Eigen::Vector3d velocity;
  Eigen::Vector3d angular_velocity;
};

inline void eigenOdometryFromMsg(const nav_msgs::OdometryConstPtr& msg,
                                 EigenOdometry* odometry) {
  odometry->position = Eigen::Vector3d(msg->pose.pose.position.x,
                                       msg->pose.pose.position.y,
                                       msg->pose.pose.position.z);
  odometry->orientation = Eigen::Quaterniond(msg->pose.pose.orientation.w,
                                             msg->pose.pose.orientation.x,
                                             msg->pose.pose.orientation.y,
                                             msg->pose.pose.orientation.z);
  odometry->velocity = Eigen::Vector3d(msg->twist.twist.linear.x,
                                       msg->twist.twist.linear.y,
                                       msg->twist.twist.linear.z);
  odometry->angular_velocity = Eigen::Vector3d(msg->twist.twist.angular.x,
                                               msg->twist.twist.angular.y,
                                               msg->twist.twist.angular.z);
}

struct EigenTrajectoryPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EigenTrajectoryPoint()
      : position_W(Eigen::Vector3d::Zero()),
        velocity_W(Eigen::Vector3d::Zero()),
        acceleration_W(Eigen::Vector3d::Zero()),
        orientation_W_B(Eigen::Quaterniond::Identity()),
        angular_velocity_W(Eigen::Vector3d::Zero()) {}

  double getYaw() const {
    return Eigen::Quaterniond(orientation_W_B).toRotationMatrix().eulerAngles(2, 1, 0)[0];
  }

  double getYawRate() const {
    return angular_velocity_W.z();
  }

  Eigen::Vector3d position_W;
  Eigen::Vector3d velocity_W;
  Eigen::Vector3d acceleration_W;
  Eigen::Quaterniond orientation_W_B;
  Eigen::Vector3d angular_velocity_W;
};

typedef std::deque<EigenTrajectoryPoint, Eigen::aligned_allocator<EigenTrajectoryPoint>>
    EigenTrajectoryPointDeque;

inline void eigenTrajectoryPointFromPoseMsg(const geometry_msgs::PoseStamped& msg,
                                            EigenTrajectoryPoint* trajectory_point) {
  assert(trajectory_point != nullptr);
  trajectory_point->position_W = Eigen::Vector3d(msg.pose.position.x,
                                                 msg.pose.position.y,
                                                 msg.pose.position.z);
  trajectory_point->orientation_W_B = Eigen::Quaterniond(msg.pose.orientation.w,
                                                         msg.pose.orientation.x,
                                                         msg.pose.orientation.y,
                                                         msg.pose.orientation.z);
  trajectory_point->velocity_W.setZero();
  trajectory_point->acceleration_W.setZero();
  trajectory_point->angular_velocity_W.setZero();
}

inline void eigenTrajectoryPointFromMsg(
    const trajectory_msgs::MultiDOFJointTrajectoryPoint& msg,
    EigenTrajectoryPoint* trajectory_point) {
  assert(trajectory_point != nullptr);

  if (!msg.transforms.empty()) {
    const geometry_msgs::Transform& transform = msg.transforms.front();
    trajectory_point->position_W = Eigen::Vector3d(transform.translation.x,
                                                   transform.translation.y,
                                                   transform.translation.z);
    trajectory_point->orientation_W_B = Eigen::Quaterniond(transform.rotation.w,
                                                           transform.rotation.x,
                                                           transform.rotation.y,
                                                           transform.rotation.z);
  } else {
    trajectory_point->position_W.setZero();
    trajectory_point->orientation_W_B = Eigen::Quaterniond::Identity();
  }

  if (!msg.velocities.empty()) {
    const geometry_msgs::Twist& velocity = msg.velocities.front();
    trajectory_point->velocity_W = Eigen::Vector3d(velocity.linear.x,
                                                   velocity.linear.y,
                                                   velocity.linear.z);
    trajectory_point->angular_velocity_W = Eigen::Vector3d(velocity.angular.x,
                                                           velocity.angular.y,
                                                           velocity.angular.z);
  } else {
    trajectory_point->velocity_W.setZero();
    trajectory_point->angular_velocity_W.setZero();
  }

  if (!msg.accelerations.empty()) {
    const geometry_msgs::Twist& acceleration = msg.accelerations.front();
    trajectory_point->acceleration_W = Eigen::Vector3d(acceleration.linear.x,
                                                       acceleration.linear.y,
                                                       acceleration.linear.z);
  } else {
    trajectory_point->acceleration_W.setZero();
  }
}

inline void calculateAllocationMatrix(const RotorConfiguration& rotor_configuration,
                                      Eigen::Matrix4Xd* allocation_matrix) {
  assert(allocation_matrix != nullptr);
  allocation_matrix->resize(4, rotor_configuration.rotors.size());
  unsigned int i = 0;
  for (const Rotor& rotor : rotor_configuration.rotors) {
    // Set first row of allocation matrix.
    (*allocation_matrix)(0, i) = -sin(rotor.angle) * rotor.arm_length
                                 * rotor.rotor_force_constant;
    // Set second row of allocation matrix.
    (*allocation_matrix)(1, i) = cos(rotor.angle) * rotor.arm_length
                                 * rotor.rotor_force_constant;
    // Set third row of allocation matrix.
    // (*allocation_matrix)(2, i) = -rotor.direction * rotor.rotor_force_constant
    //                              * rotor.rotor_moment_constant;
    (*allocation_matrix)(2, i) = rotor.direction * rotor.rotor_moment_constant;
    // Set forth row of allocation matrix.
    (*allocation_matrix)(3, i) = rotor.rotor_force_constant;
    ++i;
  }

  Eigen::FullPivLU<Eigen::Matrix4Xd> lu(*allocation_matrix);
  // Setting the threshold for when pivots of the rank calculation should be considered nonzero.
  lu.setThreshold(1e-9);
  int rank = lu.rank();
  if (rank < 4) {
    std::cout << "The rank of the allocation matrix is " << lu.rank()
              << ", it should have rank 4, to have a fully controllable system,"
              << " check your configuration." << std::endl;
  }

}

inline void skewMatrixFromVector(Eigen::Vector3d& vector, Eigen::Matrix3d* skew_matrix) {
  *skew_matrix << 0, -vector.z(), vector.y(),
                  vector.z(), 0, -vector.x(),
                  -vector.y(), vector.x(), 0;
}

inline void vectorFromSkewMatrix(Eigen::Matrix3d& skew_matrix, Eigen::Vector3d* vector) {
  *vector << skew_matrix(2, 1), skew_matrix(0,2), skew_matrix(1, 0);
}
}

#endif /* INCLUDE_ROTORS_CONTROL_COMMON_H_ */
