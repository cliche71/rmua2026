#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PoseStamped, Transform, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def fill_quaternion_from_yaw(rotation, yaw):
    rotation.x = 0.0
    rotation.y = 0.0
    rotation.z = math.sin(0.5 * yaw)
    rotation.w = math.cos(0.5 * yaw)


class Se3TrajectoryStepCommand:
    def __init__(self):
        rospy.init_node("se3_trajectory_step_command")

        self.initial_pose_topic = rospy.get_param(
            "~initial_pose_topic", "/airsim_node/initial_pose"
        )
        self.command_trajectory_topic = rospy.get_param(
            "~command_trajectory_topic", "/command/trajectory"
        )
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
            self.command_trajectory_topic, MultiDOFJointTrajectory, queue_size=1
        )
        self.initial_pose_sub = rospy.Subscriber(
            self.initial_pose_topic, PoseStamped, self.initial_pose_cb, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)), self.timer_cb
        )

        rospy.loginfo("se3_trajectory_step_command started.")
        rospy.loginfo("command_trajectory_topic: %s", self.command_trajectory_topic)
        rospy.loginfo("hover_height_m: %.2f", self.hover_height_m)
        rospy.loginfo(
            "step after %.2fs: dx=%.2f dy=%.2f dz=%.2f",
            self.step_start_time,
            self.step_x_m,
            self.step_y_m,
            self.step_z_m,
        )

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

        now = rospy.Time.now()
        elapsed = (now - self.start_time).to_sec()
        if elapsed < self.start_delay:
            return

        step_active = elapsed >= self.step_start_time
        step_x = self.step_x_m if step_active else 0.0
        step_y = self.step_y_m if step_active else 0.0
        step_z = self.step_z_m if step_active else 0.0

        position = (
            self.initial_pose.pose.position.x + step_x,
            self.initial_pose.pose.position.y + step_y,
            self.initial_pose.pose.position.z - self.hover_height_m + step_z,
        )
        yaw = yaw_from_quaternion(self.initial_pose.pose.orientation)

        msg = MultiDOFJointTrajectory()
        msg.header.stamp = now
        msg.header.frame_id = self.frame_id
        msg.joint_names = [self.child_frame_id]

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = position[0]
        transform.translation.y = position[1]
        transform.translation.z = position[2]
        fill_quaternion_from_yaw(transform.rotation, yaw)

        velocity = Twist()
        acceleration = Twist()
        point.transforms.append(transform)
        point.velocities.append(velocity)
        point.accelerations.append(acceleration)
        point.time_from_start = rospy.Duration(0.0)
        msg.points.append(point)

        self.command_pub.publish(msg)
        rospy.loginfo_throttle(
            1.0,
            "publish trajectory step: x=%.2f y=%.2f z=%.2f active=%s",
            position[0],
            position[1],
            position[2],
            str(step_active),
        )


if __name__ == "__main__":
    node = Se3TrajectoryStepCommand()
    rospy.spin()
