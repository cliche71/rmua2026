#include "eskf.hpp"

namespace
{
Eigen::Matrix3d Skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}
}

ErrorStateKalmanFilter::ErrorStateKalmanFilter(double gravity, double pos_noise, double vel_noise, double ori_noise, 
    double gyr_bias_noise, double acc_bias_noise, double pos_std, double ori_std,
    double gyr_noise_density, double acc_noise_density)
{
    m_g = Eigen::Vector3d(0.0, 0.0, gravity);
    m_P.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_POSI) = Eigen::Matrix3d::Identity() * pos_noise * pos_noise;
    m_P.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity() * vel_noise * vel_noise;
    m_P.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = Eigen::Matrix3d::Identity() * ori_noise * ori_noise;
    m_P.block<3, 3>(INDEX_STATE_GYRO_BIAS, INDEX_STATE_GYRO_BIAS) =
            Eigen::Matrix3d::Identity() * gyr_bias_noise * gyr_bias_noise;
    m_P.block<3, 3>(INDEX_STATE_ACC_BIAS, INDEX_STATE_ACC_BIAS) =
            Eigen::Matrix3d::Identity() * acc_bias_noise * acc_bias_noise;
    m_R(0, 0) = pos_std * pos_std;
    m_R(1, 1) = pos_std * pos_std;
    m_R(2, 2) = pos_std * pos_std;
    m_R(3, 3) = ori_std * ori_std;
    m_R(4, 4) = ori_std * ori_std;
    m_R(5, 5) = ori_std * ori_std;
    // Continuous-time IMU noise density covariance.
    m_Q.block<3, 3>(0, 0) =
        Eigen::Matrix3d::Identity() * gyr_noise_density * gyr_noise_density;
    m_Q.block<3, 3>(3, 3) =
        Eigen::Matrix3d::Identity() * acc_noise_density * acc_noise_density;
    m_X = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
    m_F = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    m_C = Eigen::Matrix<double, DIM_MEASUREMENT_NOISE, DIM_MEASUREMENT_NOISE>::Identity();
    m_G.block<3, 3>(INDEX_MEASUREMENT_POSI, INDEX_MEASUREMENT_POSI) = Eigen::Matrix3d::Identity();
    m_G.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
}

bool ErrorStateKalmanFilter::Init(Eigen::Matrix4d initPose, Eigen::Vector3d initVel, long long tc)
{
    m_pose = initPose;
    m_velocity = initVel;
    m_last_imu_tc = tc;
    m_last_unbias_acc = Eigen::Vector3d::Zero();
    m_last_unbias_gyr = Eigen::Vector3d::Zero();
    m_have_last_imu_measurement = false;
    return true;
}

bool ErrorStateKalmanFilter::Predict(Eigen::Vector3d imu_acc, Eigen::Vector3d imu_gyr, Eigen::Vector3d& pos, Eigen::Vector3d& vel, Eigen::Vector3d& angle_vel, Eigen::Quaterniond& q,  long long tc)
{
    double delta_t = (tc - m_last_imu_tc) / 1000000000.0;
    if(delta_t < 0.005 || delta_t > 0.015)
    {
        m_last_imu_tc = tc;
        return false;
    }
    std::lock_guard<std::mutex> lock(m_mtx);
    Eigen::Vector3d unbias_gyr = imu_gyr - m_gyro_bias;
    if (!m_have_last_imu_measurement)
    {
        m_last_unbias_gyr = unbias_gyr;
    }
    Eigen::Vector3d phi = (unbias_gyr +m_last_unbias_gyr) / 2.0 * delta_t;
    double phi_norm = phi.norm();
    Eigen::Matrix3d skew_sym_m_phi;
    skew_sym_m_phi << 
        0.0, -phi[2], phi[1], 
        phi[2], 0.0, -phi[0],
        -phi[1], phi[0], 0.0;
    Eigen::Matrix3d R_i0_i1 = Eigen::Matrix3d::Identity();
    if (phi_norm < 1.0e-8)
    {
        R_i0_i1 += skew_sym_m_phi;
    }
    else
    {
        R_i0_i1 += sinf64(phi_norm) / phi_norm * skew_sym_m_phi +
            (1 - cosf64(phi_norm)) / (phi_norm * phi_norm) *
                skew_sym_m_phi * skew_sym_m_phi;
    }
    Eigen::Matrix3d last_pose = m_pose.block<3, 3>(0, 0);
    m_pose.block<3, 3>(0, 0) =  last_pose * R_i0_i1;
    Eigen::Vector3d unbias_acc = imu_acc - m_accel_bias;
    Eigen::Vector3d last_world_acc = last_pose * m_last_unbias_acc - m_g;
    Eigen::Vector3d world_acc = m_pose.block<3, 3>(0, 0) * unbias_acc - m_g;
    if (!m_have_last_imu_measurement)
    {
        m_last_unbias_acc = unbias_acc;
        last_world_acc = world_acc;
    }
    Eigen::Vector3d last_vel = m_velocity;
    m_velocity = last_vel + (world_acc + last_world_acc) / 2.0 * delta_t;
    m_pose.block<3, 1>(0, 3) += (last_vel + m_velocity) / 2.0 * delta_t;
    Eigen::Matrix3d tmpr = m_pose.block<3, 3>(0, 0);
    Eigen::Quaterniond tmpq(tmpr);
    const Eigen::Matrix3d R = m_pose.block<3, 3>(0, 0);
    const Eigen::Vector3d f_b = unbias_acc;
    const Eigen::Vector3d w_b = unbias_gyr;
    m_F.setZero();
    m_B.setZero();
    m_F.block<3, 3>(INDEX_STATE_POSI, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity();
    m_F.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ORI) = R * Skew(f_b);
    m_F.block<3, 3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS) = -R;
    m_F.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = -Skew(w_b);
    m_F.block<3, 3>(INDEX_STATE_ORI, INDEX_STATE_GYRO_BIAS) = Eigen::Matrix3d::Identity();
    m_B.block<3, 3>(INDEX_STATE_VEL, 3) = -R;
    m_B.block<3, 3>(INDEX_STATE_ORI, 0) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> Fk =  Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() + m_F * delta_t;
    m_X = Fk * m_X;
    // m_Q is treated as continuous-time IMU noise density.
    m_P = Fk * m_P * Fk.transpose() + m_B * m_Q * m_B.transpose() * delta_t;
    m_P = 0.5 * (m_P + m_P.transpose());

    m_last_imu_tc = tc;
    m_last_unbias_gyr = unbias_gyr;
    m_last_unbias_acc = unbias_acc;
    m_have_last_imu_measurement = true;

    pos = m_pose.block<3, 1>(0, 3);
    vel = m_velocity;
    angle_vel = m_last_unbias_gyr;
    q = tmpq;

    return true;
}

bool ErrorStateKalmanFilter::correct(Eigen::Vector3d gps_pos, Eigen::Quaterniond gps_q)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    m_Y.block(0, 0, 3, 1) = gps_pos - m_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d measure_q = gps_q.matrix();
    Eigen::Matrix3d err_q = measure_q.inverse() * m_pose.block<3, 3>(0, 0);
    Eigen::AngleAxisd rotation_vector(err_q);
    m_Y.block(3, 0, 3, 1) = rotation_vector.axis() * rotation_vector.angle();
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> P_prior = m_P;
    const Eigen::Matrix<double, DIM_MEASUREMENT, DIM_MEASUREMENT> R =
        m_C * m_R * m_C.transpose();
    m_K = P_prior * m_G.transpose() * (m_G * P_prior * m_G.transpose() + R).inverse();
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> A = I - m_K * m_G;
    m_P = A * P_prior * A.transpose() + m_K * R * m_K.transpose();
    m_P = 0.5 * (m_P + m_P.transpose());
    m_X = m_X + m_K *(m_Y - m_G*m_X);
    ApplyErrorState();
    return true;
}

bool ErrorStateKalmanFilter::correctPosition(Eigen::Vector3d gps_pos)
{
    std::lock_guard<std::mutex> lock(m_mtx);

    Eigen::Matrix<double, 3, DIM_STATE> H = Eigen::Matrix<double, 3, DIM_STATE>::Zero();
    H.block<3, 3>(0, INDEX_STATE_POSI) = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R = m_R.block<3, 3>(INDEX_MEASUREMENT_POSI, INDEX_MEASUREMENT_POSI);
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> P_prior = m_P;
    Eigen::Matrix<double, DIM_STATE, 3> K =
        P_prior * H.transpose() * (H * P_prior * H.transpose() + R).inverse();
    Eigen::Vector3d residual = gps_pos - m_pose.block<3, 1>(0, 3);

    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> I =
        Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    const Eigen::Matrix<double, DIM_STATE, DIM_STATE> A = I - K * H;
    m_P = A * P_prior * A.transpose() + K * R * K.transpose();
    m_P = 0.5 * (m_P + m_P.transpose());
    m_X = m_X + K * (residual - H * m_X);
    ApplyErrorState();
    return true;
}

Eigen::Vector3d ErrorStateKalmanFilter::GetPosition()
{
    std::lock_guard<std::mutex> lock(m_mtx);
    return m_pose.block<3, 1>(0, 3);
}

void ErrorStateKalmanFilter::ApplyErrorState()
{
    m_pose.block<3, 1>(0, 3) = m_pose.block<3, 1>(0, 3) + m_X.block<3, 1>(INDEX_STATE_POSI, 0);
    m_velocity = (m_velocity + m_X.block<3, 1>(INDEX_STATE_VEL, 0));

    Eigen::Vector3d delta_theta = m_X.block<3, 1>(INDEX_STATE_ORI, 0);
    double delta_theta_norm = delta_theta.norm();
    if (delta_theta_norm > 1.0e-12)
    {
        Eigen::AngleAxisd err_r(delta_theta_norm, delta_theta / delta_theta_norm);
        m_pose.block<3, 3>(0, 0) = m_pose.block<3, 3>(0, 0) * err_r.inverse();
    }

    m_gyro_bias += m_X.block<3, 1>(INDEX_STATE_GYRO_BIAS, 0);
    m_accel_bias += m_X.block<3, 1>(INDEX_STATE_ACC_BIAS, 0);
    m_X = Eigen::Matrix<double, DIM_STATE, 1>::Zero();
}
