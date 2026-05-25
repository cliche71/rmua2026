#include "basic_dev.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "basic_dev");

    ros::NodeHandle nh;
    BasicDev node(nh);

    ros::spin();
    return 0;
}

BasicDev::BasicDev(ros::NodeHandle& nh)
    : nh_(nh),
      it_(nh_)
{
    takeoff_srv_.request.waitOnLastTask = 1;
    land_srv_.request.waitOnLastTask = 1;

    pose_sub_ = nh_.subscribe(
        "/airsim_node/drone_1/debug/pose_gt",
        1,
        &BasicDev::poseCb,
        this
    );

    front_left_image_sub_ = it_.subscribe(
        "/airsim_node/drone_1/front_left/Scene",
        1,
        &BasicDev::frontLeftImageCb,
        this
    );

    vel_pub_ = nh_.advertise<airsim_ros::VelCmd>(
        "/airsim_node/drone_1/vel_body_cmd",
        1
    );

    takeoff_client_ = nh_.serviceClient<airsim_ros::Takeoff>(
        "/airsim_node/drone_1/takeoff"
    );

    land_client_ = nh_.serviceClient<airsim_ros::Land>(
        "/airsim_node/drone_1/land"
    );

    reset_client_ = nh_.serviceClient<airsim_ros::Reset>(
        "/airsim_node/reset"
    );

    control_timer_ = nh_.createTimer(
        ros::Duration(0.05),
        &BasicDev::controlLoop,
        this
    );

    ROS_INFO("basic_dev main node started.");
}

void BasicDev::poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    x_ = msg->pose.position.x;
    y_ = msg->pose.position.y;
    z_ = msg->pose.position.z;

    Eigen::Quaterniond q(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z
    );

    Eigen::Vector3d euler = q.matrix().eulerAngles(2, 1, 0);
    yaw_ = euler[0];

    got_pose_ = true;

    ROS_INFO_THROTTLE(
        1.0,
        "pose: x=%.2f y=%.2f z=%.2f yaw=%.2f",
        x_, y_, z_, yaw_
    );
}

void BasicDev::frontLeftImageCb(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImageConstPtr cv_ptr =
            cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_8UC3);

        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            front_left_img_ = cv_ptr->image.clone();
        }

        got_front_left_image_ = true;

        ROS_INFO_THROTTLE(
            1.0,
            "front_left image received: width=%d height=%d encoding=%s",
            msg->width,
            msg->height,
            msg->encoding.c_str()
        );
    }
    catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR_THROTTLE(1.0, "cv_bridge error: %s", e.what());
    }
}

void BasicDev::controlLoop(const ros::TimerEvent& event)
{
    if (!got_pose_)
    {
        ROS_WARN_THROTTLE(1.0, "waiting for pose...");
        return;
    }

    if (!got_front_left_image_)
    {
        ROS_WARN_THROTTLE(1.0, "waiting for front left image...");
        return;
    }

    if (!takeoff_called_)
    {
        if (takeoff_client_.call(takeoff_srv_) && takeoff_srv_.response.success)
        {
            takeoff_called_ = true;
            takeoff_time_ = ros::Time::now();
            ROS_INFO("takeoff success.");
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "takeoff failed, retrying...");
        }

        return;
    }

    double t = (ros::Time::now() - takeoff_time_).toSec();

    if (t < 3.0)
    {
        publishBodyVel(0.0, 0.0, 0.0, 0.0);
        ROS_INFO_THROTTLE(1.0, "hover after takeoff.");
        return;
    }

    if (t < 5.0)
    {
        publishBodyVel(0.3, 0.0, 0.0, 0.0);
        ROS_INFO_THROTTLE(1.0, "control test: moving forward slowly.");
        return;
    }

    publishBodyVel(0.0, 0.0, 0.0, 0.0);
    ROS_INFO_THROTTLE(1.0, "hover.");
}

void BasicDev::publishBodyVel(double vx, double vy, double vz, double yaw_rate)
{
    airsim_ros::VelCmd cmd;

    cmd.header.stamp = ros::Time::now();
    cmd.header.frame_id = "drone_1";

    cmd.vx = vx;
    cmd.vy = vy;
    cmd.vz = vz;
    cmd.yawRate = yaw_rate;

    cmd.va = 4;
    cmd.stop = 0;

    vel_pub_.publish(cmd);
}