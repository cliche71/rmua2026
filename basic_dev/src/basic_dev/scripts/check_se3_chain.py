#!/usr/bin/env python3
import math

import rospy
from airsim_ros.msg import RotorPWM
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory


def pos_tuple_from_pose(pose):
    if hasattr(pose, "position"):
        return (pose.position.x, pose.position.y, pose.position.z)
    return (pose.x, pose.y, pose.z)


def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class Se3ChainChecker:
    def __init__(self):
        rospy.init_node("check_se3_chain")

        self.command_topic = rospy.get_param(
            "~command_topic", "/command/trajectory"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/se3_truth_odom")
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/airsim_node/drone_1/debug/pose_gt"
        )
        self.motor_topic = rospy.get_param(
            "~motor_topic", "/airsim_node/drone_1/rotor_pwm_cmd"
        )
        self.min_motion_m = rospy.get_param("~min_motion_m", 0.05)
        self.motion_window_s = rospy.get_param("~motion_window_s", 5.0)
        self.target_error_warn_m = rospy.get_param("~target_error_warn_m", 0.35)
        self.pwm_active_threshold = rospy.get_param("~pwm_active_threshold", 0.05)

        self.last_command = None
        self.last_command_stamp = None
        self.last_odom = None
        self.last_odom_stamp = None
        self.last_pose = None
        self.last_pose_stamp = None
        self.last_pwm = None
        self.last_pwm_stamp = None

        self.motion_anchor_pose = None
        self.motion_anchor_stamp = None

        rospy.Subscriber(
            self.command_topic,
            MultiDOFJointTrajectory,
            self.command_cb,
            queue_size=1,
        )
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber(self.motor_topic, RotorPWM, self.pwm_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0), self.report_cb)

        rospy.loginfo("[check_se3] command_topic=%s", self.command_topic)
        rospy.loginfo("[check_se3] odom_topic=%s", self.odom_topic)
        rospy.loginfo("[check_se3] pose_topic=%s", self.pose_topic)
        rospy.loginfo("[check_se3] motor_topic=%s", self.motor_topic)

    def command_cb(self, msg):
        if not msg.points or not msg.points[0].transforms:
            return
        self.last_command = pos_tuple_from_pose(msg.points[0].transforms[0].translation)
        self.last_command_stamp = rospy.Time.now()

    def odom_cb(self, msg):
        self.last_odom = pos_tuple_from_pose(msg.pose.pose)
        self.last_odom_stamp = rospy.Time.now()

    def pose_cb(self, msg):
        self.last_pose = pos_tuple_from_pose(msg.pose)
        self.last_pose_stamp = rospy.Time.now()
        if self.motion_anchor_pose is None:
            self.motion_anchor_pose = self.last_pose
            self.motion_anchor_stamp = self.last_pose_stamp

    def pwm_cb(self, msg):
        self.last_pwm = (
            msg.rotorPWM0,
            msg.rotorPWM1,
            msg.rotorPWM2,
            msg.rotorPWM3,
        )
        self.last_pwm_stamp = rospy.Time.now()

    def age(self, stamp):
        if stamp is None:
            return None
        return (rospy.Time.now() - stamp).to_sec()

    def report_cb(self, _event):
        now = rospy.Time.now()
        command_age = self.age(self.last_command_stamp)
        odom_age = self.age(self.last_odom_stamp)
        pose_age = self.age(self.last_pose_stamp)
        pwm_age = self.age(self.last_pwm_stamp)

        if self.last_command is None:
            rospy.logwarn("[check_se3] no trajectory command received")
        if self.last_odom is None:
            rospy.logwarn("[check_se3] no odom received")
        if self.last_pose is None:
            rospy.logwarn("[check_se3] no AirSim pose_gt received")
        if self.last_pwm is None:
            rospy.logwarn("[check_se3] no rotor PWM received")

        pwm_min = None
        pwm_max = None
        if self.last_pwm is not None:
            pwm_min = min(self.last_pwm)
            pwm_max = max(self.last_pwm)

        target_error = None
        if self.last_command is not None and self.last_pose is not None:
            target_error = distance(self.last_command, self.last_pose)

        moved = None
        if self.last_pose is not None and self.motion_anchor_pose is not None:
            moved = distance(self.last_pose, self.motion_anchor_pose)
            if (now - self.motion_anchor_stamp).to_sec() >= self.motion_window_s:
                if (
                    target_error is not None
                    and target_error > self.target_error_warn_m
                    and moved < self.min_motion_m
                ):
                    if self.last_pwm is None:
                        rospy.logwarn(
                            "[check_se3] target error %.2fm but no PWM output",
                            target_error,
                        )
                    elif pwm_max < self.pwm_active_threshold:
                        rospy.logwarn(
                            "[check_se3] target error %.2fm but PWM is near zero: max=%.3f",
                            target_error,
                            pwm_max,
                        )
                    else:
                        rospy.logwarn(
                            "[check_se3] target error %.2fm and PWM max=%.3f, but pose moved only %.2fm in %.1fs",
                            target_error,
                            pwm_max,
                            moved,
                            self.motion_window_s,
                        )
                self.motion_anchor_pose = self.last_pose
                self.motion_anchor_stamp = now

        rospy.loginfo(
            "[check_se3] cmd=%s age=%s odom=%s age=%s pose=%s age=%s pwm_min=%s pwm_max=%s age=%s target_error=%s moved_window=%s",
            self.format_vec(self.last_command),
            self.format_age(command_age),
            self.format_vec(self.last_odom),
            self.format_age(odom_age),
            self.format_vec(self.last_pose),
            self.format_age(pose_age),
            self.format_float(pwm_min),
            self.format_float(pwm_max),
            self.format_age(pwm_age),
            self.format_float(target_error),
            self.format_float(moved),
        )

    @staticmethod
    def format_vec(value):
        if value is None:
            return "None"
        return "(%.2f, %.2f, %.2f)" % value

    @staticmethod
    def format_age(value):
        if value is None:
            return "None"
        return "%.2fs" % value

    @staticmethod
    def format_float(value):
        if value is None:
            return "None"
        return "%.3f" % value


if __name__ == "__main__":
    checker = Se3ChainChecker()
    rospy.spin()
