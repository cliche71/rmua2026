#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class OdomToPose:
    def __init__(self):
        rospy.init_node("odom_to_pose")

        self.odom_topic = rospy.get_param("~odom_topic", "/eskf_odom")
        self.pose_topic = rospy.get_param("~pose_topic", "/current_pose")

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_cb,
            queue_size=1,
            tcp_nodelay=True,
        )

        rospy.loginfo(
            "odom_to_pose started: %s -> %s",
            self.odom_topic,
            self.pose_topic,
        )

    def odom_cb(self, msg):
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.pose_pub.publish(pose_msg)


if __name__ == "__main__":
    OdomToPose()
    rospy.spin()
