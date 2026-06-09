#include "imu_gps_odometry.hpp"

#include <algorithm>
#include <cmath>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odometry"); // 初始化ros 节点，命名为 basic
    ros::NodeHandle n; // 创建node控制句柄
    ros::NodeHandle private_nh("~");

    double gravity = -9.81083;
    double pos_noise = 0.1;
    double vel_noise = 0.1;
    double ori_noise = 0.1;
    double gyr_bias_noise = 0.0003158085227;
    double acc_bias_noise = 0.001117221;
    double gps_position_std = 0.3;
    double gps_orientation_std = 1.0;
    double imu_gyro_noise_density = 0.00143;
    double imu_acc_noise_density = 0.0386;

    private_nh.param("gravity", gravity, gravity);
    private_nh.param("pos_noise", pos_noise, pos_noise);
    private_nh.param("vel_noise", vel_noise, vel_noise);
    private_nh.param("ori_noise", ori_noise, ori_noise);
    private_nh.param("gyr_bias_noise", gyr_bias_noise, gyr_bias_noise);
    private_nh.param("acc_bias_noise", acc_bias_noise, acc_bias_noise);
    private_nh.param("gps_position_std", gps_position_std, gps_position_std);
    private_nh.param("gps_orientation_std", gps_orientation_std, gps_orientation_std);
    if (private_nh.hasParam("imu_gyro_noise"))
    {
        private_nh.param("imu_gyro_noise", imu_gyro_noise_density, imu_gyro_noise_density);
        ROS_WARN("imu_gps_odometry: parameter '~imu_gyro_noise' is deprecated; use '~imu_gyro_noise_density' with unit rad/s/sqrt(Hz)");
    }
    if (private_nh.hasParam("imu_acc_noise"))
    {
        private_nh.param("imu_acc_noise", imu_acc_noise_density, imu_acc_noise_density);
        ROS_WARN("imu_gps_odometry: parameter '~imu_acc_noise' is deprecated; use '~imu_acc_noise_density' with unit m/s^2/sqrt(Hz)");
    }
    private_nh.param("imu_gyro_noise_density", imu_gyro_noise_density, imu_gyro_noise_density);
    private_nh.param("imu_acc_noise_density", imu_acc_noise_density, imu_acc_noise_density);
    g_gravity_world = Eigen::Vector3d(0.0, 0.0, gravity);

    //(重力， P_位置不确定度_std, P_速度不确定度_std, P_角度不确定度_std, P_角速度bias不确定度_std, P_加速度bias不确定度_std,
    //gps位置测量噪声_std gpsz姿态测量噪声_std, imu角速度连续噪声密度, imu加速度连续噪声密度)
    g_eskf_ptr = new ErrorStateKalmanFilter(
        gravity, pos_noise, vel_noise, ori_noise, gyr_bias_noise, acc_bias_noise,
        gps_position_std, gps_orientation_std,
        imu_gyro_noise_density, imu_acc_noise_density);

    private_nh.param("use_gps_orientation", g_use_gps_orientation, false);
    private_nh.param("use_gps_z_anchor", g_use_gps_z_anchor, false);
    private_nh.param("use_receive_time_for_imu_dt", g_use_receive_time_for_imu_dt, true);
    private_nh.param(
        "initialize_position_from_first_gps",
        g_initialize_position_from_first_gps, true);
    private_nh.param(
        "enable_accel_model_diagnostics",
        g_enable_accel_model_diagnostics, true);

    std::string odom_topic = "/eskf_odom";
    std::string gps_topic = "/airsim_node/drone_1/gps";
    std::string imu_topic = "/airsim_node/drone_1/imu/imu";
    std::string initial_pose_topic = "/airsim_node/initial_pose";
    private_nh.param("odom_topic", odom_topic, odom_topic);
    private_nh.param("gps_topic", gps_topic, gps_topic);
    private_nh.param("imu_topic", imu_topic, imu_topic);
    private_nh.param("initial_pose_topic", initial_pose_topic, initial_pose_topic);

    ROS_INFO(
        "imu_gps_odometry: use_gps_orientation=%s use_gps_z_anchor=%s use_receive_time_for_imu_dt=%s initialize_position_from_first_gps=%s accel_diag=%s gravity=(%.3f, %.3f, %.3f) pos_noise=%.3f vel_noise=%.3f gps_position_std=%.3f imu_gyro_noise_density=%.6f(rad/s/sqrt(Hz)) imu_acc_noise_density=%.6f(m/s^2/sqrt(Hz)) odom_topic=%s gps_topic=%s imu_topic=%s initial_pose_topic=%s",
        g_use_gps_orientation ? "true" : "false",
        g_use_gps_z_anchor ? "true" : "false",
        g_use_receive_time_for_imu_dt ? "true" : "false",
        g_initialize_position_from_first_gps ? "true" : "false",
        g_enable_accel_model_diagnostics ? "true" : "false",
        g_gravity_world.x(), g_gravity_world.y(), g_gravity_world.z(),
        pos_noise, vel_noise, gps_position_std,
        imu_gyro_noise_density, imu_acc_noise_density,
        odom_topic.c_str(), gps_topic.c_str(), imu_topic.c_str(),
        initial_pose_topic.c_str());

    const int queue_size = 1;
    g_eskf_odom_puber = n.advertise<nav_msgs::Odometry>(odom_topic, queue_size);
    ros::Subscriber odom_suber = n.subscribe<geometry_msgs::PoseStamped>(gps_topic, queue_size, odom_local_ned_cb);
    ros::Subscriber imu_suber = n.subscribe<sensor_msgs::Imu>(imu_topic, queue_size, imu_cb);
    ros::Subscriber init_pose_suber = n.subscribe<geometry_msgs::PoseStamped>(initial_pose_topic, queue_size, init_pose_ned_cb);

    ros::spin();
    delete g_eskf_ptr;
    return 0;
}

void init_pose_ned_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    if (!g_have_initial_pose)
    {
        g_initial_pose_position << msg->pose.position.x, msg->pose.position.y,
            msg->pose.position.z;
        g_initial_pose_orientation = Eigen::Quaterniond(
            msg->pose.orientation.w, msg->pose.orientation.x,
            msg->pose.orientation.y, msg->pose.orientation.z);
        g_initial_pose_orientation.normalize();
        g_initial_pose_z = msg->pose.position.z;
        g_have_initial_pose = true;
        ROS_INFO(
            "imu_gps_odometry: initial_pose position=(%.3f, %.3f, %.3f)",
            g_initial_pose_position.x(), g_initial_pose_position.y(),
            g_initial_pose_position.z());
    }

    if(!g_eskf_ptr->m_isInitailed && !g_initialize_position_from_first_gps)
    {
        Eigen::Matrix4d r0 = Eigen::Matrix4d::Identity();
        r0.block<3,3>(0, 0) = g_initial_pose_orientation.toRotationMatrix();
        r0.block<3,1>(0, 3) = g_initial_pose_position;
        const ros::Time init_time = g_use_receive_time_for_imu_dt
            ? ros::Time::now()
            : msg->header.stamp;
        g_eskf_ptr->Init(r0, Eigen::Vector3d::Zero(), init_time.toNSec());
        g_last_predict_tc = init_time.toNSec();
        g_have_last_predict_tc = true;
        g_eskf_ptr->m_isInitailed = true;
    }
}

void odom_local_ned_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    // ROS_INFO("Get odom_local_ned_cd\n  orientation: %f-%f-%f-%f\n  position: %f-%f-%f\n", 
    // msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, //姿态四元数
    // msg->pose.position.x, msg->pose.position.y,msg->pose.position.z);
    if (!g_have_initial_pose)
    {
        ROS_WARN_THROTTLE(1.0, "imu_gps_odometry: skip gps correction before initial_pose is ready");
        return;
    }

    double raw_gps_z = msg->pose.position.z;
    if (g_use_gps_z_anchor && !g_have_gps_z_anchor)
    {
        g_gps_z_anchor = raw_gps_z;
        g_have_gps_z_anchor = true;
        ROS_INFO("imu_gps_odometry: gps.z anchor=%.3f initial_pose.z=%.3f",
            g_gps_z_anchor, g_initial_pose_z);
    }

    double corrected_gps_z = raw_gps_z;
    if (g_use_gps_z_anchor)
    {
        corrected_gps_z = g_initial_pose_z + (raw_gps_z - g_gps_z_anchor);
    }
    Eigen::Vector3d gps_pos(msg->pose.position.x, msg->pose.position.y, corrected_gps_z);

    if (!g_eskf_ptr->m_isInitailed)
    {
        Eigen::Matrix4d r0 = Eigen::Matrix4d::Identity();
        r0.block<3,3>(0, 0) = g_initial_pose_orientation.toRotationMatrix();
        r0.block<3,1>(0, 3) = gps_pos;
        const ros::Time init_time = g_use_receive_time_for_imu_dt
            ? ros::Time::now()
            : msg->header.stamp;
        g_eskf_ptr->Init(r0, Eigen::Vector3d::Zero(), init_time.toNSec());
        g_last_predict_tc = init_time.toNSec();
        g_have_last_predict_tc = true;
        g_eskf_ptr->m_isInitailed = true;
        ROS_WARN(
            "imu_gps_odometry: initialized position from first GPS=(%.3f, %.3f, %.3f), initial_pose_position=(%.3f, %.3f, %.3f)",
            gps_pos.x(), gps_pos.y(), gps_pos.z(),
            g_initial_pose_position.x(), g_initial_pose_position.y(),
            g_initial_pose_position.z());
    }

    if (g_use_gps_orientation)
    {
        g_eskf_ptr->correct(gps_pos,
            Eigen::Quaterniond(msg->pose.orientation.w,msg->pose.orientation.x, msg->pose.orientation.y,msg->pose.orientation.z));
    }
    else
    {
        g_eskf_ptr->correctPosition(gps_pos);
    }

    ++g_gps_correction_count;
    log_stats_throttled(ros::Time::now());
}

void imu_cb(const sensor_msgs::Imu::ConstPtr& msg)
{
    // ROS_INFO("Get imu data.\n %f %f %f \n %f %f %f", msg->angular_velocity.x, msg->angular_velocity.y,
    // msg->angular_velocity.z, msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    if(g_eskf_ptr->m_isInitailed)
    {
        const ros::Time now = ros::Time::now();
        const double imu_stamp_age = (now - msg->header.stamp).toSec();
        if (std::isfinite(imu_stamp_age) && imu_stamp_age >= 0.0)
        {
            g_imu_stamp_age_sum += imu_stamp_age;
            g_imu_stamp_age_max = std::max(g_imu_stamp_age_max, imu_stamp_age);
            ++g_imu_stamp_age_count;
        }

        Eigen::Vector3d pos, vel, angle_vel;
        Eigen::Quaterniond q;
        const ros::Time predict_time = g_use_receive_time_for_imu_dt
            ? now
            : msg->header.stamp;
        const long long predict_tc = predict_time.toNSec();
        if (g_have_last_predict_tc)
        {
            const double predict_dt =
                static_cast<double>(predict_tc - g_last_predict_tc) / 1000000000.0;
            if (std::isfinite(predict_dt) && predict_dt >= 0.0)
            {
                if (g_predict_dt_window_count == 0)
                {
                    g_predict_dt_window_min = predict_dt;
                    g_predict_dt_window_max = predict_dt;
                }
                else
                {
                    g_predict_dt_window_min =
                        std::min(g_predict_dt_window_min, predict_dt);
                    g_predict_dt_window_max =
                        std::max(g_predict_dt_window_max, predict_dt);
                }
                g_predict_dt_window_sum += predict_dt;
                ++g_predict_dt_window_count;
            }
        }
        g_last_predict_tc = predict_tc;
        g_have_last_predict_tc = true;
        if (!g_eskf_ptr->Predict(Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
            Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z), 
            pos, vel, angle_vel, q, predict_tc))
        {
            ++g_imu_predict_skip_count;
            log_stats_throttled(now);
            return;
        }
        nav_msgs::Odometry msg2;
        msg2.header.stamp = now;
        msg2.header.frame_id = "world";
        msg2.child_frame_id = "drone_1";
        msg2.pose.pose.position.x = pos.x();
        msg2.pose.pose.position.y = pos.y();
        msg2.pose.pose.position.z = pos.z();
        msg2.pose.pose.orientation.w = q.w();
        msg2.pose.pose.orientation.x = q.x();
        msg2.pose.pose.orientation.y = q.y();
        msg2.pose.pose.orientation.z = q.z();
        msg2.twist.twist.linear.x = vel.x();
        msg2.twist.twist.linear.y = vel.y();
        msg2.twist.twist.linear.z = vel.z();
        msg2.twist.twist.angular.x = angle_vel.x();
        msg2.twist.twist.angular.y = angle_vel.y();
        msg2.twist.twist.angular.z = angle_vel.z();
        g_eskf_odom_puber.publish(msg2);
        ++g_odom_publish_count;

        if (g_enable_accel_model_diagnostics)
        {
            Eigen::Quaterniond q_norm = q;
            q_norm.normalize();
            const Eigen::Matrix3d r_body_to_world = q_norm.toRotationMatrix();
            const Eigen::Vector3d imu_acc(
                msg->linear_acceleration.x,
                msg->linear_acceleration.y,
                msg->linear_acceleration.z);
            const Eigen::Vector3d r_acc = r_body_to_world * imu_acc;
            const Eigen::Vector3d rt_acc = r_body_to_world.transpose() * imu_acc;
            g_acc_diag_racc_sum += r_acc.norm();
            g_acc_diag_racc_minus_g_sum += (r_acc - g_gravity_world).norm();
            g_acc_diag_racc_plus_g_sum += (r_acc + g_gravity_world).norm();
            g_acc_diag_rtacc_minus_g_sum += (rt_acc - g_gravity_world).norm();
            g_acc_diag_rtacc_plus_g_sum += (rt_acc + g_gravity_world).norm();
            ++g_acc_diag_count;
        }

        log_stats_throttled(msg2.header.stamp);
    }
}

void log_stats_throttled(const ros::Time& now)
{
    static ros::Time last_log_time;
    static uint64_t last_gps_correction_count = 0;
    static uint64_t last_odom_publish_count = 0;
    static double last_imu_stamp_age_sum = 0.0;
    static uint64_t last_imu_stamp_age_count = 0;

    if (last_log_time.isZero())
    {
        last_log_time = now;
        last_gps_correction_count = g_gps_correction_count;
        last_odom_publish_count = g_odom_publish_count;
        last_imu_stamp_age_sum = g_imu_stamp_age_sum;
        last_imu_stamp_age_count = g_imu_stamp_age_count;
        return;
    }

    const double dt = (now - last_log_time).toSec();
    if (dt < 1.0)
    {
        return;
    }

    const uint64_t gps_delta = g_gps_correction_count - last_gps_correction_count;
    const uint64_t odom_delta = g_odom_publish_count - last_odom_publish_count;
    const double imu_age_sum_delta = g_imu_stamp_age_sum - last_imu_stamp_age_sum;
    const uint64_t imu_age_count_delta =
        g_imu_stamp_age_count - last_imu_stamp_age_count;
    const double imu_age_avg = imu_age_count_delta > 0
        ? imu_age_sum_delta / static_cast<double>(imu_age_count_delta)
        : 0.0;
    const double predict_dt_avg = g_predict_dt_window_count > 0
        ? g_predict_dt_window_sum / static_cast<double>(g_predict_dt_window_count)
        : 0.0;
    const double acc_diag_inv_count = g_acc_diag_count > 0
        ? 1.0 / static_cast<double>(g_acc_diag_count)
        : 0.0;
    ROS_INFO(
        "imu_gps_odometry stats: corrections=%llu odom_published=%llu correction_hz=%.1f odom_hz=%.1f predict_skipped=%llu predict_dt_avg=%.4f predict_dt_min=%.4f predict_dt_max=%.4f imu_stamp_age_avg=%.3f imu_stamp_age_max=%.3f imu_time_source=%s acc_diag_avg_norms[Racc=%.3f Racc-g=%.3f Racc+g=%.3f RTacc-g=%.3f RTacc+g=%.3f]",
        static_cast<unsigned long long>(g_gps_correction_count),
        static_cast<unsigned long long>(g_odom_publish_count),
        gps_delta / dt,
        odom_delta / dt,
        static_cast<unsigned long long>(g_imu_predict_skip_count),
        predict_dt_avg,
        g_predict_dt_window_min,
        g_predict_dt_window_max,
        imu_age_avg,
        g_imu_stamp_age_max,
        g_use_receive_time_for_imu_dt ? "receive" : "header",
        g_acc_diag_racc_sum * acc_diag_inv_count,
        g_acc_diag_racc_minus_g_sum * acc_diag_inv_count,
        g_acc_diag_racc_plus_g_sum * acc_diag_inv_count,
        g_acc_diag_rtacc_minus_g_sum * acc_diag_inv_count,
        g_acc_diag_rtacc_plus_g_sum * acc_diag_inv_count);

    last_log_time = now;
    last_gps_correction_count = g_gps_correction_count;
    last_odom_publish_count = g_odom_publish_count;
    last_imu_stamp_age_sum = g_imu_stamp_age_sum;
    last_imu_stamp_age_count = g_imu_stamp_age_count;
    g_predict_dt_window_sum = 0.0;
    g_predict_dt_window_min = 0.0;
    g_predict_dt_window_max = 0.0;
    g_predict_dt_window_count = 0;
    g_acc_diag_racc_sum = 0.0;
    g_acc_diag_racc_minus_g_sum = 0.0;
    g_acc_diag_racc_plus_g_sum = 0.0;
    g_acc_diag_rtacc_minus_g_sum = 0.0;
    g_acc_diag_rtacc_plus_g_sum = 0.0;
    g_acc_diag_count = 0;
}
