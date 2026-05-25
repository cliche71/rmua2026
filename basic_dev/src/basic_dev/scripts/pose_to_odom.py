#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def quat_conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def normalize_quat(q):
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm < 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


class PoseToOdom:
    def __init__(self):
        rospy.init_node("pose_to_odom")

        self.pose_topic = rospy.get_param(
            "~pose_topic", "/airsim_node/drone_1/debug/pose_gt"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/se3_truth_odom")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", "drone_1")
        self.max_dt = rospy.get_param("~max_dt", 0.2)
        self.angular_velocity_alpha = rospy.get_param("~angular_velocity_alpha", 0.5)

        self.last_stamp = None
        self.last_position = None
        self.last_quat = None
        self.filtered_angular_velocity = (0.0, 0.0, 0.0)

        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)
        self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)

        rospy.loginfo("pose_to_odom started.")
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("odom_topic: %s", self.odom_topic)

    def pose_cb(self, msg):
        stamp = msg.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()

        position = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        quat = normalize_quat(
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        )

        linear_velocity = (0.0, 0.0, 0.0)
        angular_velocity = (0.0, 0.0, 0.0)

        if self.last_stamp is not None:
            dt = (stamp - self.last_stamp).to_sec()
            if 1.0e-4 < dt <= self.max_dt:
                linear_velocity = (
                    (position[0] - self.last_position[0]) / dt,
                    (position[1] - self.last_position[1]) / dt,
                    (position[2] - self.last_position[2]) / dt,
                )

                # Lee's controller expects body-frame angular velocity. For an
                # orientation q_W_B, the body-frame increment is q_last^-1*q_now.
                dq = quat_multiply(quat_conjugate(self.last_quat), quat)
                dq = normalize_quat(dq)
                if dq[3] < 0.0:
                    dq = (-dq[0], -dq[1], -dq[2], -dq[3])
                angle = 2.0 * math.atan2(
                    math.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2]), dq[3]
                )
                if angle > math.pi:
                    angle -= 2.0 * math.pi
                axis_norm = math.sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2])
                if axis_norm > 1.0e-9:
                    raw_angular_velocity = (
                        dq[0] / axis_norm * angle / dt,
                        dq[1] / axis_norm * angle / dt,
                        dq[2] / axis_norm * angle / dt,
                    )
                    alpha = max(0.0, min(1.0, self.angular_velocity_alpha))
                    angular_velocity = (
                        alpha * raw_angular_velocity[0]
                        + (1.0 - alpha) * self.filtered_angular_velocity[0],
                        alpha * raw_angular_velocity[1]
                        + (1.0 - alpha) * self.filtered_angular_velocity[1],
                        alpha * raw_angular_velocity[2]
                        + (1.0 - alpha) * self.filtered_angular_velocity[2],
                    )
                    self.filtered_angular_velocity = angular_velocity

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = msg.pose
        odom.twist.twist.linear.x = linear_velocity[0]
        odom.twist.twist.linear.y = linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        odom.twist.twist.angular.x = angular_velocity[0]
        odom.twist.twist.angular.y = angular_velocity[1]
        odom.twist.twist.angular.z = angular_velocity[2]
        self.pub.publish(odom)

        self.last_stamp = stamp
        self.last_position = position
        self.last_quat = quat


if __name__ == "__main__":
    node = PoseToOdom()
    rospy.spin()
