#include <Eigen/Core>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>

geometry_msgs::PoseStamped initial_pose;
bool initial_pose_received = false;

void initialPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  initial_pose = *msg;
  initial_pose_received = true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "hovering_example");
  ros::NodeHandle nh;
  // Create a private node handle for accessing node parameters.
  ros::NodeHandle nh_private("~");
  ros::Publisher trajectory_pub = nh.advertise<geometry_msgs::PoseStamped>("command/pose", 10);
  ros::Subscriber initial_pose_sub = nh.subscribe("/airsim_node/initial_pose", 10, initialPoseCallback);
  ROS_INFO("Take Off. waiting for initial_pose");

  // Wait for 5 seconds to let the Airsim Run.
  ros::Duration(3.0).sleep();

  // Wait until the initial pose is received
  ros::Time start_time = ros::Time::now();
  ros::Duration max_wait_time(5.0);
  while (ros::ok() && !initial_pose_received) {
    ros::spinOnce(); // Wait for a valid initial pose
    ros::Duration(0.1).sleep(); // Sleep to prevent busy-waiting
    if (ros::Time::now() - start_time > max_wait_time) {
      ROS_WARN("Timeout reached. Initial pose not received.");
      break;
    }
  }

  if (initial_pose_received) {
    geometry_msgs::PoseStamped posecmd_msg;
    posecmd_msg = initial_pose;
    posecmd_msg.header.stamp = ros::Time::now();
    posecmd_msg.pose.position.z -= 1.5; // The positive direction of the Z-axis is downward.
    // posecmd_msg.pose.position.y += 1;
    // posecmd_msg.pose.position.x += 1;
    posecmd_msg.pose.orientation = tf::createQuaternionMsgFromYaw(0);

    ROS_INFO("Publishing waypoint on namespace %s: [%f, %f, %f].",
            nh.getNamespace().c_str(), posecmd_msg.pose.position.x,
            posecmd_msg.pose.position.y, posecmd_msg.pose.position.z);
    trajectory_pub.publish(posecmd_msg);
  }

  ros::spinOnce();
  ros::shutdown();

  return 0;
}
