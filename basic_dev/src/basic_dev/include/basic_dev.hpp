#ifndef _BASIC_DEV_HPP_
#define _BASIC_DEV_HPP_

#include <mutex>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "airsim_ros/VelCmd.h"
#include "airsim_ros/Takeoff.h"
#include "airsim_ros/Land.h"
#include "airsim_ros/Reset.h"

class BasicDev
{
public:
    explicit BasicDev(ros::NodeHandle& nh);
    ~BasicDev() = default;

private:
    void poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void frontLeftImageCb(const sensor_msgs::ImageConstPtr& msg);
    void controlLoop(const ros::TimerEvent& event);

    void publishBodyVel(double vx, double vy, double vz, double yaw_rate);

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    ros::Subscriber pose_sub_;
    image_transport::Subscriber front_left_image_sub_;

    ros::Publisher vel_pub_;

    ros::ServiceClient takeoff_client_;
    ros::ServiceClient land_client_;
    ros::ServiceClient reset_client_;

    ros::Timer control_timer_;

    airsim_ros::Takeoff takeoff_srv_;
    airsim_ros::Land land_srv_;
    airsim_ros::Reset reset_srv_;

    std::mutex image_mutex_;
    cv::Mat front_left_img_;

    bool got_pose_ = false;
    bool got_front_left_image_ = false;
    bool takeoff_called_ = false;

    ros::Time takeoff_time_;

    double x_ = 0.0;
    double y_ = 0.0;
    double z_ = 0.0;
    double yaw_ = 0.0;
};

#endif