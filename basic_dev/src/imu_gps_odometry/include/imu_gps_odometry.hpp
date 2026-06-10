#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "airsim_ros/VelCmd.h"
#include "airsim_ros/PoseCmd.h"
#include "airsim_ros/Takeoff.h"
#include "airsim_ros/Land.h"
#include "airsim_ros/GPSYaw.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/Imu.h"
#include <geometry_msgs/PoseStamped.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>
#include <mutex>
#include "eskf.hpp"

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<double> gauss_dist{0.0, 1.0};
ErrorStateKalmanFilter* g_eskf_ptr;
ros::Publisher g_eskf_odom_puber;
bool g_use_gps_orientation = false;
bool g_use_gps_z_anchor = false;
bool g_use_receive_time_for_imu_dt = true;
bool g_initialize_position_from_first_gps = true;
bool g_enable_accel_model_diagnostics = true;
bool g_enable_correction_diagnostics = true;
bool g_log_every_gps_correction = false;
bool g_have_initial_pose = false;
bool g_have_gps_z_anchor = false;
double g_initial_pose_z = 0.0;
double g_gps_z_anchor = 0.0;
Eigen::Vector3d g_gravity_world = Eigen::Vector3d::Zero();
Eigen::Vector3d g_initial_pose_position = Eigen::Vector3d::Zero();
Eigen::Quaterniond g_initial_pose_orientation = Eigen::Quaterniond::Identity();
double g_imu_stamp_age_sum = 0.0;
double g_imu_stamp_age_max = 0.0;
uint64_t g_imu_stamp_age_count = 0;
double g_predict_dt_window_sum = 0.0;
double g_predict_dt_window_min = 0.0;
double g_predict_dt_window_max = 0.0;
uint64_t g_predict_dt_window_count = 0;
long long g_last_predict_tc = 0;
bool g_have_last_predict_tc = false;
uint64_t g_gps_correction_count = 0;
uint64_t g_odom_publish_count = 0;
uint64_t g_imu_predict_skip_count = 0;
double g_acc_diag_racc_sum = 0.0;
double g_acc_diag_racc_minus_g_sum = 0.0;
double g_acc_diag_racc_plus_g_sum = 0.0;
double g_acc_diag_rtacc_minus_g_sum = 0.0;
double g_acc_diag_rtacc_plus_g_sum = 0.0;
uint64_t g_acc_diag_count = 0;
void odom_local_ned_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
void init_pose_ned_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
void imu_cb(const sensor_msgs::Imu::ConstPtr& msg);
void log_stats_throttled(const ros::Time& now);
void log_correction_debug(const ros::Time& now);
