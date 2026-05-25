#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path


class OdomToTf:
    def __init__(self):
        rospy.init_node("odom_to_tf")

        self.odom_topic = rospy.get_param("~odom_topic", "/se3_truth_odom")
        self.parent_frame = rospy.get_param("~parent_frame", "")
        self.child_frame = rospy.get_param("~child_frame", "")
        self.path_topic = rospy.get_param("~path_topic", "/se3_truth_path")
        self.max_path_size = int(rospy.get_param("~max_path_size", 20000))

        self.br = tf2_ros.TransformBroadcaster()
        self.path = Path()
        self.last_stamp = None
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1)
        self.sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=10)

        rospy.loginfo("odom_to_tf started.")
        rospy.loginfo("odom_topic: %s", self.odom_topic)
        rospy.loginfo("path_topic: %s", self.path_topic)

    def odom_cb(self, msg):
        parent = self.parent_frame or msg.header.frame_id or "world"
        child = self.child_frame or msg.child_frame_id or "drone_1"

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(tf_msg)

        if self.last_stamp is not None and tf_msg.header.stamp < self.last_stamp:
            rospy.loginfo("ROS time moved backwards; clearing accumulated path.")
            self.path = Path()
        self.last_stamp = tf_msg.header.stamp

        pose = PoseStamped()
        pose.header = tf_msg.header
        pose.pose = msg.pose.pose
        self.path.header = tf_msg.header
        self.path.poses.append(pose)
        if self.max_path_size > 0 and len(self.path.poses) > self.max_path_size:
            self.path.poses = self.path.poses[-self.max_path_size:]
        self.path_pub.publish(self.path)


if __name__ == "__main__":
    OdomToTf()
    rospy.spin()
