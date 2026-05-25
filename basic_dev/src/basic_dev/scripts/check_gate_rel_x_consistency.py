#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PoseStamped


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class GateRelXConsistencyCheck:
    def __init__(self):
        rospy.init_node("check_gate_rel_x_consistency")

        self.pose_topic = rospy.get_param(
            "~pose_topic", "/airsim_node/drone_1/debug/pose_gt"
        )
        self.gate_topic = rospy.get_param("~gate_topic", "/gate/relative_pose")
        self.required_forward_delta_m = rospy.get_param("~required_forward_delta_m", 1.0)
        self.error_warn_m = rospy.get_param("~error_warn_m", 0.35)
        self.yaw_warn_deg = rospy.get_param("~yaw_warn_deg", 8.0)
        self.lateral_warn_m = rospy.get_param("~lateral_warn_m", 0.35)

        self.current_pos = None
        self.current_yaw = None
        self.current_rel_x = None

        self.start_pos = None
        self.start_yaw = None
        self.start_rel_x = None
        self.reported = False

        self.pose_sub = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_cb, queue_size=1
        )
        self.gate_sub = rospy.Subscriber(
            self.gate_topic, PoseStamped, self.gate_cb, queue_size=1
        )

        rospy.loginfo("[check_gate_rel_x] pose_topic=%s", self.pose_topic)
        rospy.loginfo("[check_gate_rel_x] gate_topic=%s", self.gate_topic)

    def pose_cb(self, msg):
        self.current_pos = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        self.current_yaw = yaw_from_quaternion(msg.pose.orientation)
        self.maybe_set_start()
        self.maybe_report()

    def gate_cb(self, msg):
        self.current_rel_x = msg.pose.position.x
        self.maybe_set_start()
        self.maybe_report()

    def maybe_set_start(self):
        if self.start_pos is not None:
            return
        if self.current_pos is None or self.current_yaw is None or self.current_rel_x is None:
            return

        self.start_pos = self.current_pos
        self.start_yaw = self.current_yaw
        self.start_rel_x = self.current_rel_x
        rospy.loginfo(
            "[check_gate_rel_x] start_pos=(%.3f, %.3f, %.3f)",
            self.start_pos[0],
            self.start_pos[1],
            self.start_pos[2],
        )
        rospy.loginfo("[check_gate_rel_x] start_yaw=%.2f deg", math.degrees(self.start_yaw))
        rospy.loginfo("[check_gate_rel_x] start_rel_x=%.3f", self.start_rel_x)

    def maybe_report(self):
        if self.reported or self.start_pos is None:
            return
        if self.current_pos is None or self.current_yaw is None or self.current_rel_x is None:
            return

        c = math.cos(self.start_yaw)
        s = math.sin(self.start_yaw)
        dx = self.current_pos[0] - self.start_pos[0]
        dy = self.current_pos[1] - self.start_pos[1]
        forward_delta = c * dx + s * dy
        lateral_delta = -s * dx + c * dy

        if forward_delta < self.required_forward_delta_m:
            return

        rel_x_delta = self.current_rel_x - self.start_rel_x
        expected_rel_x_delta = -forward_delta
        error = abs(rel_x_delta - expected_rel_x_delta)
        yaw_delta_deg = math.degrees(normalize_angle(self.current_yaw - self.start_yaw))

        rospy.loginfo(
            "[check_gate_rel_x] current_pos=(%.3f, %.3f, %.3f)",
            self.current_pos[0],
            self.current_pos[1],
            self.current_pos[2],
        )
        rospy.loginfo(
            "[check_gate_rel_x] forward_delta=%.2f lateral_delta=%.2f yaw_delta_deg=%.2f rel_x_delta=%.2f expected_rel_x_delta=%.2f error=%.2f",
            forward_delta,
            lateral_delta,
            yaw_delta_deg,
            rel_x_delta,
            expected_rel_x_delta,
            error,
        )

        if abs(yaw_delta_deg) > self.yaw_warn_deg:
            rospy.logwarn(
                "[check_gate_rel_x] WARN: yaw changed %.2fdeg, rel_x consistency is weak",
                yaw_delta_deg,
            )
        if abs(lateral_delta) > self.lateral_warn_m:
            rospy.logwarn(
                "[check_gate_rel_x] WARN: lateral motion %.2fm, rel_x consistency is weak",
                lateral_delta,
            )
        if error > self.error_warn_m:
            rospy.logwarn("[check_gate_rel_x] WARN: rel_x is not consistent with drone motion")

        self.reported = True


if __name__ == "__main__":
    node = GateRelXConsistencyCheck()
    rospy.spin()
