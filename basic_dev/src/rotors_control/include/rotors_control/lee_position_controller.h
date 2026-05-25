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

#ifndef ROTORS_CONTROL_LEE_POSITION_CONTROLLER_H
#define ROTORS_CONTROL_LEE_POSITION_CONTROLLER_H

#include "rotors_control/common.h"
#include "rotors_control/parameters.h"

namespace rotors_control {

// Default values for the lee position controller and the Asctec Firefly.
static const Eigen::Vector3d kDefaultPositionGain = Eigen::Vector3d(6, 6, 6);
static const Eigen::Vector3d kDefaultVelocityGain = Eigen::Vector3d(4.7, 4.7, 4.7);
static const Eigen::Vector3d kDefaultAttitudeGain = Eigen::Vector3d(3, 3, 0.035);
static const Eigen::Vector3d kDefaultAngularRateGain = Eigen::Vector3d(0.52, 0.52, 0.025);

class LeePositionControllerParameters {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LeePositionControllerParameters()
      : position_gain_(kDefaultPositionGain),
        velocity_gain_(kDefaultVelocityGain),
        attitude_gain_(kDefaultAttitudeGain),
        angular_rate_gain_(kDefaultAngularRateGain) {
    // calculateAllocationMatrix(rotor_configuration_, &allocation_matrix_);
  }

  Eigen::Matrix4Xd allocation_matrix_;
  Eigen::Vector3d position_gain_;
  Eigen::Vector3d velocity_gain_;
  Eigen::Vector3d attitude_gain_;
  Eigen::Vector3d angular_rate_gain_;
  // RotorConfiguration rotor_configuration_;
};

class LeePositionController {
 public:
  LeePositionController();
  ~LeePositionController();
  void InitializeParameters();
  void CalculateRotorVelocities(Eigen::VectorXd* rotor_velocities);

  void SetOdometry(const EigenOdometry& odometry);
  void SetOdometry(const EigenOdometry& odometry, double dt);
  void SetTrajectoryPoint(
    const EigenTrajectoryPoint& command_trajectory);
  Eigen::Vector3d GetDisturbanceEstimate() const;
  Eigen::Vector3d GetTorqueDisturbanceEstimate() const;
  double GetAttitudeRecoveryPwmFloor() const;
  double GetCurrentTiltAngleRadEstimate() const;
  double GetLastThrustCommand() const;
  Eigen::Vector3d GetLastPositionError() const;
  Eigen::Vector3d GetLastVelocityError() const;
  Eigen::Vector3d GetLastNominalFeedbackAcceleration() const;
  Eigen::Vector3d GetLastFeedbackAcceleration() const;
  Eigen::Vector3d GetLastDesiredAcceleration() const;
  Eigen::Vector3d GetLastAngularAcceleration() const;
  Eigen::Vector3d GetLastTorqueCommand() const;
  Eigen::Vector3d GetLastAttitudeError() const;
  Eigen::Vector3d GetLastAngularRateError() const;
  Eigen::Vector3d GetLastAngularVelocity() const;
  Eigen::Vector4d GetLastRawRotorPwm() const;
  Eigen::Vector4d GetLastFinalRotorPwm() const;
  bool GetLastPwmDesaturated() const;

  LeePositionControllerParameters controller_parameters_;
  VehicleParameters vehicle_parameters_;
  bool odometry_velocity_in_body_frame_;
  bool use_airsim_pwm_mixer_;
  bool enable_disturbance_observer_;
  bool enable_attitude_disturbance_observer_;
  Eigen::Vector3d disturbance_observer_position_gain_;
  Eigen::Vector3d attitude_disturbance_observer_gain_;
  double disturbance_observer_max_acc_;
  double disturbance_observer_compensation_gain_;
  double disturbance_observer_freeze_tilt_rad_;
  double disturbance_observer_decay_rate_;
  double attitude_disturbance_observer_max_acc_;
  double attitude_disturbance_observer_compensation_gain_;
  double attitude_disturbance_observer_freeze_tilt_rad_;
  double attitude_disturbance_observer_decay_rate_;
  double max_rotor_velocity_;
  double min_rotor_pwm_;
  double max_rotor_pwm_;
  bool enable_pwm_desaturation_;
  double max_tilt_angle_rad_;
  bool enable_attitude_recovery_pwm_floor_;
  double attitude_recovery_pwm_min_tilt_rad_;
  double attitude_recovery_pwm_full_tilt_rad_;
  double attitude_recovery_min_pwm_;
  Eigen::Vector3d torque_scale_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  bool initialized_params_;
  bool controller_active_;

  Eigen::Vector3d normalized_attitude_gain_;
  Eigen::Vector3d normalized_angular_rate_gain_;
  Eigen::MatrixX4d angular_acc_to_rotor_velocities_;

  EigenTrajectoryPoint command_trajectory_;
  EigenOdometry odometry_;
  double odometry_dt_;
  bool disturbance_observer_initialized_;
  bool attitude_disturbance_observer_initialized_;
  Eigen::Vector3d eso_position_;
  Eigen::Vector3d eso_velocity_;
  Eigen::Vector3d eso_disturbance_;
  Eigen::Quaterniond eso_attitude_;
  Eigen::Vector3d eso_angular_velocity_;
  Eigen::Vector3d eso_torque_disturbance_;
  double last_attitude_recovery_pwm_floor_;
  double last_tilt_angle_rad_;
  double last_thrust_command_;
  Eigen::Vector3d last_position_error_;
  Eigen::Vector3d last_velocity_error_;
  Eigen::Vector3d last_nominal_feedback_acceleration_;
  Eigen::Vector3d last_feedback_acceleration_;
  Eigen::Vector3d last_desired_acceleration_;
  Eigen::Vector3d last_angular_acceleration_;
  Eigen::Vector3d last_torque_command_;
  Eigen::Vector3d last_attitude_error_;
  Eigen::Vector3d last_angular_rate_error_;
  Eigen::Vector3d last_angular_velocity_;
  Eigen::Vector4d last_raw_rotor_pwm_;
  Eigen::Vector4d last_final_rotor_pwm_;
  bool last_pwm_desaturated_;

  void ComputeDesiredAngularAcc(const Eigen::Vector3d& acceleration,
                                Eigen::Vector3d* angular_acceleration);
  void ComputeDesiredAcceleration(Eigen::Vector3d* acceleration);
  Eigen::Vector3d ComputeNominalFeedbackAcceleration();
  Eigen::Vector3d CurrentVelocityWorld() const;
  void UpdateDisturbanceObserver(const Eigen::Vector3d& nominal_acceleration);
  void UpdateAttitudeDisturbanceObserver(
      const Eigen::Vector3d& nominal_angular_acceleration);
  void LimitDesiredTilt(Eigen::Vector3d* acceleration) const;
  double CurrentTiltAngleRad() const;
};
}

#endif // ROTORS_CONTROL_LEE_POSITION_CONTROLLER_H
