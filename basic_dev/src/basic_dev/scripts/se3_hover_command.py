#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PoseStamped


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def fill_quaternion_from_yaw(rotation, yaw):
    rotation.x = 0.0
    rotation.y = 0.0
    rotation.z = math.sin(0.5 * yaw)
    rotation.w = math.cos(0.5 * yaw)


class Se3HoverCommand:
    def __init__(self):
        rospy.init_node("se3_hover_command")

        self.initial_pose_topic = rospy.get_param(
            "~initial_pose_topic", "/airsim_node/initial_pose"
        )
        self.command_pose_topic = rospy.get_param("~command_pose_topic", "/command/pose")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", "drone_1")
        self.hover_height_m = rospy.get_param("~hover_height_m", 1.0)
        self.step_x_m = rospy.get_param("~step_x_m", 0.0)
        self.step_y_m = rospy.get_param("~step_y_m", 0.0)
        self.step_z_m = rospy.get_param("~step_z_m", 0.0)
        self.step_start_time = rospy.get_param("~step_start_time", 4.0)
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)
        self.start_delay = rospy.get_param("~start_delay", 1.0)

        self.initial_pose = None
        self.start_time = None
        self.command_pub = rospy.Publisher(
            self.command_pose_topic, PoseStamped, queue_size=1
        )
        self.initial_pose_sub = rospy.Subscriber(
            self.initial_pose_topic, PoseStamped, self.initial_pose_cb, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)), self.timer_cb
        )

        rospy.loginfo("se3_hover_command started.")
        rospy.loginfo("command_pose_topic: %s", self.command_pose_topic)
        rospy.loginfo("hover_height_m: %.2f", self.hover_height_m)

    def initial_pose_cb(self, msg):
        if self.initial_pose is not None:
            return

        self.initial_pose = msg
        self.start_time = rospy.Time.now()
        rospy.loginfo(
            "got initial pose: x=%.2f y=%.2f z=%.2f",
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )

    def timer_cb(self, _event):
        if self.initial_pose is None:
            rospy.logwarn_throttle(1.0, "waiting for initial pose")
            return

        if (rospy.Time.now() - self.start_time).to_sec() < self.start_delay:
            return

        cmd = PoseStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.frame_id
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        step_active = elapsed >= self.step_start_time
        step_x = self.step_x_m if step_active else 0.0
        step_y = self.step_y_m if step_active else 0.0
        step_z = self.step_z_m if step_active else 0.0
        cmd.pose.position.x = self.initial_pose.pose.position.x
        cmd.pose.position.y = self.initial_pose.pose.position.y
        cmd.pose.position.z = self.initial_pose.pose.position.z - self.hover_height_m
        cmd.pose.position.x += step_x
        cmd.pose.position.y += step_y
        cmd.pose.position.z += step_z
        fill_quaternion_from_yaw(
            cmd.pose.orientation, yaw_from_quaternion(self.initial_pose.pose.orientation)
        )

        self.command_pub.publish(cmd)
        rospy.loginfo_throttle(
            1.0,
            "publish hover pose: x=%.2f y=%.2f z=%.2f",
            cmd.pose.position.x,
            cmd.pose.position.y,
            cmd.pose.position.z,
        )


if __name__ == "__main__":
    node = Se3HoverCommand()
    rospy.spin()
