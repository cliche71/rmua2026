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

#include "rotors_control/lee_position_controller.h"

#include <algorithm>
#include <cmath>

namespace rotors_control {

namespace {
Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d mat;
  mat << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
  return mat;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
  const double angle = w.norm();
  if (angle < 1.0e-9) {
    return Eigen::Matrix3d::Identity() + Skew(w);
  }
  return Eigen::AngleAxisd(angle, w / angle).toRotationMatrix();
}
}  // namespace

LeePositionController::LeePositionController()
    : initialized_params_(false),
      controller_active_(false),
      odometry_velocity_in_body_frame_(false),
      use_airsim_pwm_mixer_(true),
      enable_disturbance_observer_(false),
      enable_attitude_disturbance_observer_(false),
      disturbance_observer_position_gain_(Eigen::Vector3d(30.0, 500.0, 1000.0)),
      attitude_disturbance_observer_gain_(Eigen::Vector3d(20.0, 50.0, 120.0)),
      disturbance_observer_max_acc_(2.0),
      disturbance_observer_compensation_gain_(1.0),
      disturbance_observer_freeze_tilt_rad_(0.0),
      disturbance_observer_decay_rate_(1.5),
      attitude_disturbance_observer_max_acc_(8.0),
      attitude_disturbance_observer_compensation_gain_(0.7),
      attitude_disturbance_observer_freeze_tilt_rad_(1.40),
      attitude_disturbance_observer_decay_rate_(3.0),
      max_rotor_velocity_(184.6505),
      min_rotor_pwm_(0.0),
      max_rotor_pwm_(1.0),
      enable_pwm_desaturation_(false),
      max_tilt_angle_rad_(0.61),
      enable_attitude_recovery_pwm_floor_(false),
      attitude_recovery_pwm_min_tilt_rad_(0.79),
      attitude_recovery_pwm_full_tilt_rad_(1.57),
      attitude_recovery_min_pwm_(0.12),
      torque_scale_(Eigen::Vector3d::Ones()),
      odometry_dt_(1.0 / 100.0),
      disturbance_observer_initialized_(false),
      attitude_disturbance_observer_initialized_(false),
      eso_position_(Eigen::Vector3d::Zero()),
      eso_velocity_(Eigen::Vector3d::Zero()),
      eso_disturbance_(Eigen::Vector3d::Zero()),
      eso_attitude_(Eigen::Quaterniond::Identity()),
      eso_angular_velocity_(Eigen::Vector3d::Zero()),
      eso_torque_disturbance_(Eigen::Vector3d::Zero()),
      last_attitude_recovery_pwm_floor_(0.0),
      last_tilt_angle_rad_(0.0),
      last_thrust_command_(0.0),
      last_position_error_(Eigen::Vector3d::Zero()),
      last_velocity_error_(Eigen::Vector3d::Zero()),
      last_nominal_feedback_acceleration_(Eigen::Vector3d::Zero()),
      last_feedback_acceleration_(Eigen::Vector3d::Zero()),
      last_desired_acceleration_(Eigen::Vector3d::Zero()),
      last_angular_acceleration_(Eigen::Vector3d::Zero()),
      last_torque_command_(Eigen::Vector3d::Zero()),
      last_attitude_error_(Eigen::Vector3d::Zero()),
      last_angular_rate_error_(Eigen::Vector3d::Zero()),
      last_angular_velocity_(Eigen::Vector3d::Zero()),
      last_raw_rotor_pwm_(Eigen::Vector4d::Zero()),
      last_final_rotor_pwm_(Eigen::Vector4d::Zero()),
      last_pwm_desaturated_(false) {
  InitializeParameters();
}

LeePositionController::~LeePositionController() {}

void LeePositionController::InitializeParameters() {
  calculateAllocationMatrix(vehicle_parameters_.rotor_configuration_, &(controller_parameters_.allocation_matrix_));
  // To make the tuning independent of the inertia matrix we divide here.
  normalized_attitude_gain_ = controller_parameters_.attitude_gain_.transpose()
      * vehicle_parameters_.inertia_.inverse();
  // To make the tuning independent of the inertia matrix we divide here.
  normalized_angular_rate_gain_ = controller_parameters_.angular_rate_gain_.transpose()
      * vehicle_parameters_.inertia_.inverse();

  // normalized_attitude_gain_ = controller_parameters_.attitude_gain_;
  // normalized_angular_rate_gain_ = controller_parameters_.angular_rate_gain_;

  Eigen::Matrix4d I;
  I.setZero();
  I.block<3, 3>(0, 0) = vehicle_parameters_.inertia_;
  I(3, 3) = 1;
  angular_acc_to_rotor_velocities_.resize(vehicle_parameters_.rotor_configuration_.rotors.size(), 4);
  // Calculate the pseude-inverse A^{ \dagger} and then multiply by the inertia matrix I.
  // A^{ \dagger} = A^T*(A*A^T)^{-1}
  angular_acc_to_rotor_velocities_ = controller_parameters_.allocation_matrix_.transpose()
      * (controller_parameters_.allocation_matrix_
      * controller_parameters_.allocation_matrix_.transpose()).inverse() * I;
  initialized_params_ = true;
}

void LeePositionController::CalculateRotorVelocities(Eigen::VectorXd* rotor_velocities) {
  assert(rotor_velocities);
  assert(initialized_params_);

  rotor_velocities->resize(vehicle_parameters_.rotor_configuration_.rotors.size());
  // Return 0 velocities on all rotors, until the first command is received.
  if (!controller_active_) {
    *rotor_velocities = Eigen::VectorXd::Zero(rotor_velocities->rows());
    last_raw_rotor_pwm_.setZero();
    last_final_rotor_pwm_.setZero();
    last_pwm_desaturated_ = false;
    last_thrust_command_ = 0.0;
    return;
  }

  Eigen::Vector3d acceleration;
  ComputeDesiredAcceleration(&acceleration);

  Eigen::Vector3d angular_acceleration;
  ComputeDesiredAngularAcc(acceleration, &angular_acceleration);

  // Project thrust onto body z axis.
  double thrust = -vehicle_parameters_.mass_ * acceleration.dot(odometry_.orientation.toRotationMatrix().col(2));
  last_thrust_command_ = thrust;

  if (use_airsim_pwm_mixer_ &&
      vehicle_parameters_.rotor_configuration_.rotors.size() == 4) {
    const Rotor& rotor = vehicle_parameters_.rotor_configuration_.rotors.front();
    const double ct = rotor.rotor_force_constant;
    const double cq = rotor.rotor_moment_constant;
    const double arm = rotor.arm_length;
    const double f_max = ct * max_rotor_velocity_ * max_rotor_velocity_;

    Eigen::Vector3d torque =
        (vehicle_parameters_.inertia_ * angular_acceleration).cwiseProduct(torque_scale_);
    last_torque_command_ = torque;
    Eigen::Matrix4d mixer;
    mixer << 1, 1, 1, 1,
            -1, 1, 1, -1,
            -1, 1, -1, 1,
            -1, -1, 1, 1;
    Eigen::Vector4d control;
    control << thrust / ct,
               torque.x() / (arm * ct * std::sqrt(0.5)),
               torque.y() / (arm * ct * std::sqrt(0.5)),
               torque.z() / cq;
    Eigen::Vector4d rotor_forces = ct * mixer.inverse() * control;
    *rotor_velocities = rotor_forces / f_max;
  } else {
    last_torque_command_ =
        (vehicle_parameters_.inertia_ * angular_acceleration).cwiseProduct(torque_scale_);
    Eigen::Vector4d angular_acceleration_thrust;
    angular_acceleration_thrust.block<3, 1>(0, 0) = angular_acceleration;
    angular_acceleration_thrust(3) = thrust;

    *rotor_velocities = angular_acc_to_rotor_velocities_ * angular_acceleration_thrust;
    // *rotor_velocities = rotor_velocities->cwiseSqrt();
    *rotor_velocities = *rotor_velocities / (max_rotor_velocity_ * max_rotor_velocity_);
  }

  last_raw_rotor_pwm_.setZero();
  for (int i = 0; i < std::min<int>(4, rotor_velocities->rows()); ++i) {
    last_raw_rotor_pwm_[i] = (*rotor_velocities)[i];
  }

  last_tilt_angle_rad_ = CurrentTiltAngleRad();
  last_attitude_recovery_pwm_floor_ = 0.0;
  if (enable_attitude_recovery_pwm_floor_ &&
      use_airsim_pwm_mixer_ &&
      attitude_recovery_min_pwm_ > min_rotor_pwm_ &&
      last_tilt_angle_rad_ > attitude_recovery_pwm_min_tilt_rad_) {
    const double full_tilt = std::max(
        attitude_recovery_pwm_full_tilt_rad_,
        attitude_recovery_pwm_min_tilt_rad_ + 1.0e-3);
    double scale =
        (last_tilt_angle_rad_ - attitude_recovery_pwm_min_tilt_rad_) /
        (full_tilt - attitude_recovery_pwm_min_tilt_rad_);
    scale = std::max(0.0, std::min(1.0, scale));
    last_attitude_recovery_pwm_floor_ =
        std::min(max_rotor_pwm_, attitude_recovery_min_pwm_ * scale);
  }
  const double effective_min_pwm =
      std::max(min_rotor_pwm_, last_attitude_recovery_pwm_floor_);
  Eigen::VectorXd final_rotor_pwm = *rotor_velocities;
  last_pwm_desaturated_ = false;
  if (enable_pwm_desaturation_ && final_rotor_pwm.rows() > 0) {
    const double raw_min = final_rotor_pwm.minCoeff();
    const double raw_max = final_rotor_pwm.maxCoeff();
    const double allowed_span = max_rotor_pwm_ - effective_min_pwm;
    if (allowed_span > 1.0e-6) {
      const double raw_span = raw_max - raw_min;
      if (raw_span > allowed_span) {
        final_rotor_pwm =
            ((final_rotor_pwm.array() - raw_min) * (allowed_span / raw_span) +
             effective_min_pwm)
                .matrix();
      } else {
        if (raw_max > max_rotor_pwm_) {
          final_rotor_pwm.array() -= raw_max - max_rotor_pwm_;
        }
        const double shifted_min = final_rotor_pwm.minCoeff();
        if (shifted_min < effective_min_pwm) {
          final_rotor_pwm.array() += effective_min_pwm - shifted_min;
        }
      }
    } else {
      final_rotor_pwm.setConstant(std::max(effective_min_pwm, max_rotor_pwm_));
    }
    last_pwm_desaturated_ =
        (final_rotor_pwm - *rotor_velocities).cwiseAbs().maxCoeff() > 1.0e-6;
  }
  final_rotor_pwm = final_rotor_pwm.cwiseMax(
      Eigen::VectorXd::Constant(final_rotor_pwm.rows(), effective_min_pwm));
  final_rotor_pwm = final_rotor_pwm.cwiseMin(
      Eigen::VectorXd::Constant(final_rotor_pwm.rows(), max_rotor_pwm_));
  *rotor_velocities = final_rotor_pwm;
  last_final_rotor_pwm_.setZero();
  for (int i = 0; i < std::min<int>(4, rotor_velocities->rows()); ++i) {
    last_final_rotor_pwm_[i] = (*rotor_velocities)[i];
  }

}

void LeePositionController::SetOdometry(const EigenOdometry& odometry) {
  odometry_ = odometry;
}

void LeePositionController::SetOdometry(const EigenOdometry& odometry, double dt) {
  odometry_ = odometry;
  if (std::isfinite(dt) && dt > 1.0e-4) {
    odometry_dt_ = std::min(0.05, std::max(0.001, dt));
  }
}

void LeePositionController::SetTrajectoryPoint(
    const EigenTrajectoryPoint& command_trajectory) {
  command_trajectory_ = command_trajectory;
  controller_active_ = true;
}

Eigen::Vector3d LeePositionController::GetDisturbanceEstimate() const {
  return eso_disturbance_;
}

Eigen::Vector3d LeePositionController::GetTorqueDisturbanceEstimate() const {
  return eso_torque_disturbance_;
}

double LeePositionController::GetAttitudeRecoveryPwmFloor() const {
  return last_attitude_recovery_pwm_floor_;
}

double LeePositionController::GetCurrentTiltAngleRadEstimate() const {
  return last_tilt_angle_rad_;
}

double LeePositionController::GetLastThrustCommand() const {
  return last_thrust_command_;
}

Eigen::Vector3d LeePositionController::GetLastPositionError() const {
  return last_position_error_;
}

Eigen::Vector3d LeePositionController::GetLastVelocityError() const {
  return last_velocity_error_;
}

Eigen::Vector3d LeePositionController::GetLastNominalFeedbackAcceleration() const {
  return last_nominal_feedback_acceleration_;
}

Eigen::Vector3d LeePositionController::GetLastFeedbackAcceleration() const {
  return last_feedback_acceleration_;
}

Eigen::Vector3d LeePositionController::GetLastDesiredAcceleration() const {
  return last_desired_acceleration_;
}

Eigen::Vector3d LeePositionController::GetLastAngularAcceleration() const {
  return last_angular_acceleration_;
}

Eigen::Vector3d LeePositionController::GetLastTorqueCommand() const {
  return last_torque_command_;
}

Eigen::Vector3d LeePositionController::GetLastAttitudeError() const {
  return last_attitude_error_;
}

Eigen::Vector3d LeePositionController::GetLastAngularRateError() const {
  return last_angular_rate_error_;
}

Eigen::Vector3d LeePositionController::GetLastAngularVelocity() const {
  return last_angular_velocity_;
}

Eigen::Vector4d LeePositionController::GetLastRawRotorPwm() const {
  return last_raw_rotor_pwm_;
}

Eigen::Vector4d LeePositionController::GetLastFinalRotorPwm() const {
  return last_final_rotor_pwm_;
}

bool LeePositionController::GetLastPwmDesaturated() const {
  return last_pwm_desaturated_;
}

Eigen::Vector3d LeePositionController::CurrentVelocityWorld() const {
  const Eigen::Matrix3d R_W_I = odometry_.orientation.toRotationMatrix();
  return odometry_velocity_in_body_frame_
      ? R_W_I * odometry_.velocity
      : odometry_.velocity;
}

Eigen::Vector3d LeePositionController::ComputeNominalFeedbackAcceleration() {
  Eigen::Vector3d position_error;
  position_error = odometry_.position - command_trajectory_.position_W;

  Eigen::Vector3d velocity_error;
  velocity_error = CurrentVelocityWorld() - command_trajectory_.velocity_W;

  last_position_error_ = position_error;
  last_velocity_error_ = velocity_error;
  last_angular_velocity_ = odometry_.angular_velocity;
  last_nominal_feedback_acceleration_ =
      (-position_error.cwiseProduct(controller_parameters_.position_gain_)
       - velocity_error.cwiseProduct(controller_parameters_.velocity_gain_)) / vehicle_parameters_.mass_
      + command_trajectory_.acceleration_W;
  return last_nominal_feedback_acceleration_;
}

double LeePositionController::CurrentTiltAngleRad() const {
  const Eigen::Matrix3d R = odometry_.orientation.normalized().toRotationMatrix();
  const double cos_tilt = std::max(-1.0, std::min(1.0, R.col(2).dot(Eigen::Vector3d::UnitZ())));
  return std::acos(cos_tilt);
}

void LeePositionController::UpdateDisturbanceObserver(
    const Eigen::Vector3d& nominal_acceleration) {
  if (!enable_disturbance_observer_) {
    disturbance_observer_initialized_ = false;
    eso_disturbance_.setZero();
    return;
  }

  if (!disturbance_observer_initialized_) {
    eso_position_ = odometry_.position;
    eso_velocity_ = CurrentVelocityWorld();
    eso_disturbance_.setZero();
    disturbance_observer_initialized_ = true;
    return;
  }

  const double dt = std::min(0.05, std::max(0.001, odometry_dt_));
  if (disturbance_observer_freeze_tilt_rad_ > 0.0 &&
      CurrentTiltAngleRad() > disturbance_observer_freeze_tilt_rad_) {
    eso_position_ = odometry_.position;
    eso_velocity_ = CurrentVelocityWorld();
    if (disturbance_observer_decay_rate_ > 0.0) {
      const double decay = std::exp(-disturbance_observer_decay_rate_ * dt);
      eso_disturbance_ *= decay;
    }
    return;
  }

  const Eigen::Vector3d error = odometry_.position - eso_position_;
  const Eigen::Vector3d z1_dot =
      eso_velocity_ + disturbance_observer_position_gain_[0] * error;
  const Eigen::Vector3d z2_dot =
      nominal_acceleration + eso_disturbance_ +
      disturbance_observer_position_gain_[1] * error;
  const Eigen::Vector3d z3_dot =
      disturbance_observer_position_gain_[2] * error;

  eso_position_ += dt * z1_dot;
  eso_velocity_ += dt * z2_dot;
  eso_disturbance_ += dt * z3_dot;

  if (!eso_position_.allFinite() || !eso_velocity_.allFinite() ||
      !eso_disturbance_.allFinite()) {
    eso_position_ = odometry_.position;
    eso_velocity_ = CurrentVelocityWorld();
    eso_disturbance_.setZero();
    return;
  }

  const double max_acc = disturbance_observer_max_acc_;
  if (max_acc > 0.0 && eso_disturbance_.norm() > max_acc) {
    eso_disturbance_ = eso_disturbance_.normalized() * max_acc;
  }
}

void LeePositionController::UpdateAttitudeDisturbanceObserver(
    const Eigen::Vector3d& nominal_angular_acceleration) {
  if (!enable_attitude_disturbance_observer_) {
    attitude_disturbance_observer_initialized_ = false;
    eso_torque_disturbance_.setZero();
    return;
  }

  if (!attitude_disturbance_observer_initialized_) {
    eso_attitude_ = odometry_.orientation.normalized();
    eso_angular_velocity_ = odometry_.angular_velocity;
    eso_torque_disturbance_.setZero();
    attitude_disturbance_observer_initialized_ = true;
    return;
  }

  const double dt = std::min(0.05, std::max(0.001, odometry_dt_));
  if (attitude_disturbance_observer_freeze_tilt_rad_ > 0.0 &&
      CurrentTiltAngleRad() > attitude_disturbance_observer_freeze_tilt_rad_) {
    eso_attitude_ = odometry_.orientation.normalized();
    eso_angular_velocity_ = odometry_.angular_velocity;
    if (attitude_disturbance_observer_decay_rate_ > 0.0) {
      const double decay = std::exp(-attitude_disturbance_observer_decay_rate_ * dt);
      eso_torque_disturbance_ *= decay;
    } else {
      eso_torque_disturbance_.setZero();
    }
    return;
  }

  const Eigen::Quaterniond q_meas = odometry_.orientation.normalized();
  const Eigen::Matrix3d R_meas = q_meas.toRotationMatrix();
  const Eigen::Matrix3d R_hat = eso_attitude_.toRotationMatrix();

  Eigen::Matrix3d attitude_error_matrix =
      0.5 * (R_hat.transpose() * R_meas - R_meas.transpose() * R_hat);
  Eigen::Vector3d attitude_error;
  vectorFromSkewMatrix(attitude_error_matrix, &attitude_error);

  const Eigen::Vector3d omega_correction =
      eso_angular_velocity_ + attitude_disturbance_observer_gain_.x() * attitude_error;
  const Eigen::Vector3d omega_dot =
      R_hat.transpose() * R_meas * nominal_angular_acceleration +
      eso_torque_disturbance_ +
      attitude_disturbance_observer_gain_.y() * attitude_error;
  const Eigen::Vector3d torque_disturbance_dot =
      attitude_disturbance_observer_gain_.z() * attitude_error;

  eso_attitude_ =
      Eigen::Quaterniond(R_hat * ExpSO3(omega_correction * dt)).normalized();
  eso_angular_velocity_ += dt * omega_dot;
  eso_torque_disturbance_ += dt * torque_disturbance_dot;

  if (!eso_attitude_.coeffs().allFinite() ||
      !eso_angular_velocity_.allFinite() ||
      !eso_torque_disturbance_.allFinite()) {
    eso_attitude_ = odometry_.orientation.normalized();
    eso_angular_velocity_ = odometry_.angular_velocity;
    eso_torque_disturbance_.setZero();
    return;
  }

  const double max_acc = attitude_disturbance_observer_max_acc_;
  if (max_acc > 0.0 && eso_torque_disturbance_.norm() > max_acc) {
    eso_torque_disturbance_ = eso_torque_disturbance_.normalized() * max_acc;
  }
}

void LeePositionController::ComputeDesiredAcceleration(Eigen::Vector3d* acceleration) {
  assert(acceleration);

  Eigen::Vector3d e_3(Eigen::Vector3d::UnitZ());
  Eigen::Vector3d feedback_acceleration = ComputeNominalFeedbackAcceleration();
  UpdateDisturbanceObserver(feedback_acceleration);
  if (enable_disturbance_observer_) {
    feedback_acceleration -=
        disturbance_observer_compensation_gain_ * eso_disturbance_;
  }
  last_feedback_acceleration_ = feedback_acceleration;

  if (use_airsim_pwm_mixer_) {
    // AirSim RotorPWM uses positive normalized thrust. In the local FLU frame,
    // positive z position error must increase collective thrust, so we pass the
    // negative force direction into the Lee attitude construction.
    *acceleration = -feedback_acceleration - vehicle_parameters_.gravity_ * e_3;
  } else {
    *acceleration = feedback_acceleration - vehicle_parameters_.gravity_ * e_3;
  }
  LimitDesiredTilt(acceleration);
  last_desired_acceleration_ = *acceleration;
}

void LeePositionController::LimitDesiredTilt(Eigen::Vector3d* acceleration) const {
  if (acceleration == nullptr || max_tilt_angle_rad_ <= 0.0) {
    return;
  }
  const double vertical = std::abs((*acceleration).z());
  if (vertical < 1.0e-3) {
    return;
  }
  const double max_horizontal = vertical * std::tan(max_tilt_angle_rad_);
  if (max_horizontal <= 0.0) {
    return;
  }

  Eigen::Vector2d horizontal((*acceleration).x(), (*acceleration).y());
  const double horizontal_norm = horizontal.norm();
  if (horizontal_norm <= max_horizontal || horizontal_norm < 1.0e-9) {
    return;
  }

  horizontal *= max_horizontal / horizontal_norm;
  (*acceleration).x() = horizontal.x();
  (*acceleration).y() = horizontal.y();
}

// Implementation from the T. Lee et al. paper
// Control of complex maneuvers for a quadrotor UAV using geometric methods on SE(3)
void LeePositionController::ComputeDesiredAngularAcc(const Eigen::Vector3d& acceleration,
                                                     Eigen::Vector3d* angular_acceleration) {
  assert(angular_acceleration);

  Eigen::Matrix3d R = odometry_.orientation.toRotationMatrix();

  // Get the desired rotation matrix.
  Eigen::Vector3d b1_des;
  double yaw = command_trajectory_.getYaw();
  b1_des << cos(yaw), sin(yaw), 0;

  Eigen::Vector3d b3_des;
  b3_des = -acceleration / acceleration.norm();

  Eigen::Vector3d b2_des;
  b2_des = b3_des.cross(b1_des);
  b2_des.normalize();

  Eigen::Matrix3d R_des;
  R_des.col(0) = b2_des.cross(b3_des);
  R_des.col(1) = b2_des;
  R_des.col(2) = b3_des;

  // Angle error according to lee et al.
  Eigen::Matrix3d angle_error_matrix = 0.5 * (R_des.transpose() * R - R.transpose() * R_des);
  Eigen::Vector3d angle_error;
  vectorFromSkewMatrix(angle_error_matrix, &angle_error);
  last_attitude_error_ = angle_error;

  // TODO:(burrimi) include angular rate references at some point.
  Eigen::Vector3d angular_rate_des(Eigen::Vector3d::Zero());
  angular_rate_des[2] = command_trajectory_.getYawRate();

  Eigen::Vector3d angular_rate_error = odometry_.angular_velocity - R_des.transpose() * R * angular_rate_des;
  last_angular_rate_error_ = angular_rate_error;
  // Eigen::Vector3d angular_rate_error = odometry_.angular_velocity - R.transpose() * R_des * angular_rate_des;

  Eigen::Vector3d nominal_angular_acceleration =
      - angle_error.cwiseProduct(normalized_attitude_gain_)
      - angular_rate_error.cwiseProduct(normalized_angular_rate_gain_)
      + odometry_.angular_velocity.cross(odometry_.angular_velocity); // we don't need the inertia matrix here

  UpdateAttitudeDisturbanceObserver(nominal_angular_acceleration);
  if (enable_attitude_disturbance_observer_) {
    nominal_angular_acceleration -=
        attitude_disturbance_observer_compensation_gain_ * eso_torque_disturbance_;
  }

  last_angular_acceleration_ = nominal_angular_acceleration;
  *angular_acceleration = nominal_angular_acceleration;
}
}
