#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import MultiDOFJointTrajectory
from visualization_msgs.msg import Marker


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def quat_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def point_from_xyz(x, y, z):
    point = Point()
    point.x = x
    point.y = y
    point.z = z
    return point


def color(r, g, b, a=1.0):
    value = ColorRGBA()
    value.r = r
    value.g = g
    value.b = b
    value.a = a
    return value


class LiveDebugViz:
    def __init__(self):
        rospy.init_node("live_debug_viz")

        self.frame_id = rospy.get_param("~frame_id", "world")
        self.odom_topic = rospy.get_param("~odom_topic", "/se3_truth_odom")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/command/trajectory")
        self.gate_topic = rospy.get_param("~gate_topic", "/gate/relative_pose")
        self.visual_error_topic = rospy.get_param(
            "~visual_error_topic", "/gate/visual_error"
        )
        self.publish_rate = rospy.get_param("~publish_rate", 20.0)
        self.max_path_length = int(rospy.get_param("~max_path_length", 4000))
        self.gate_timeout_s = rospy.get_param("~gate_timeout_s", 0.5)
        self.visual_error_timeout_s = rospy.get_param(
            "~visual_error_timeout_s", 0.5
        )
        self.gate_relative_z_sign = rospy.get_param("~gate_relative_z_sign", -1.0)
        self.gate_marker_scale = rospy.get_param("~gate_marker_scale", 0.7)
        self.visual_arrow_scale = rospy.get_param("~visual_arrow_scale", 3.0)
        self.yaw_arrow_length = rospy.get_param("~yaw_arrow_length", 2.2)
        self.min_path_step_m = rospy.get_param("~min_path_step_m", 0.02)

        self.odom = None
        self.odom_yaw = 0.0
        self.latest_cmd_pose = None
        self.expected_path_yaw = None
        self.gate_rel = None
        self.gate_stamp = None
        self.visual_error = None
        self.visual_error_stamp = None

        self.odom_path = Path()
        self.odom_path.header.frame_id = self.frame_id
        self.cmd_path = Path()
        self.cmd_path.header.frame_id = self.frame_id

        self.odom_path_pub = rospy.Publisher(
            "/debug/odom_path", Path, queue_size=1
        )
        self.cmd_path_pub = rospy.Publisher("/debug/cmd_path", Path, queue_size=1)
        self.gate_marker_pub = rospy.Publisher(
            "/debug/current_gate_marker", Marker, queue_size=1
        )
        self.visual_marker_pub = rospy.Publisher(
            "/debug/visual_error_marker", Marker, queue_size=1
        )
        self.yaw_marker_pub = rospy.Publisher(
            "/debug/yaw_marker", Marker, queue_size=1
        )

        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_cb, queue_size=10
        )
        self.cmd_sub = rospy.Subscriber(
            self.cmd_topic, MultiDOFJointTrajectory, self.cmd_cb, queue_size=10
        )
        self.gate_sub = rospy.Subscriber(
            self.gate_topic, PoseStamped, self.gate_cb, queue_size=10
        )
        self.visual_error_sub = rospy.Subscriber(
            self.visual_error_topic, Vector3Stamped, self.visual_error_cb, queue_size=10
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)), self.timer_cb
        )

        rospy.loginfo(
            "live_debug_viz started: odom=%s cmd=%s gate=%s visual=%s frame=%s rate=%.1f",
            self.odom_topic,
            self.cmd_topic,
            self.gate_topic,
            self.visual_error_topic,
            self.frame_id,
            self.publish_rate,
        )

    def odom_cb(self, msg):
        self.odom = msg
        self.odom_yaw = quat_yaw(msg.pose.pose.orientation)

    def cmd_cb(self, msg):
        if not msg.points:
            return
        point = msg.points[0]
        if not point.transforms:
            return

        transform = point.transforms[0]
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        if pose.header.stamp == rospy.Time(0):
            pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = transform.translation.x
        pose.pose.position.y = transform.translation.y
        pose.pose.position.z = transform.translation.z
        pose.pose.orientation = transform.rotation

        previous_pose = self.latest_cmd_pose
        self.latest_cmd_pose = pose
        self.expected_path_yaw = self.compute_expected_path_yaw(point, previous_pose)

    def gate_cb(self, msg):
        self.gate_rel = msg.pose.position
        self.gate_stamp = rospy.Time.now()

    def visual_error_cb(self, msg):
        self.visual_error = msg.vector
        self.visual_error_stamp = rospy.Time.now()

    def compute_expected_path_yaw(self, point, previous_pose):
        if point.velocities:
            velocity = point.velocities[0].linear
            speed_xy = math.hypot(velocity.x, velocity.y)
            if speed_xy > 0.05:
                return math.atan2(velocity.y, velocity.x)

        if previous_pose is not None and self.latest_cmd_pose is not None:
            dx = self.latest_cmd_pose.pose.position.x - previous_pose.pose.position.x
            dy = self.latest_cmd_pose.pose.position.y - previous_pose.pose.position.y
            if math.hypot(dx, dy) > 0.05:
                return math.atan2(dy, dx)

        if point.transforms:
            return quat_yaw(point.transforms[0].rotation)
        return None

    def timer_cb(self, _event):
        now = rospy.Time.now()
        self.update_paths(now)
        self.publish_gate_marker(now)
        self.publish_visual_error_marker(now)
        self.publish_yaw_marker(now)

    def update_paths(self, now):
        if self.odom is not None:
            pose = PoseStamped()
            pose.header.stamp = self.odom.header.stamp
            if pose.header.stamp == rospy.Time(0):
                pose.header.stamp = now
            pose.header.frame_id = self.frame_id
            pose.pose = self.odom.pose.pose
            self.append_pose(self.odom_path, pose)
            self.odom_path.header.stamp = now
            self.odom_path_pub.publish(self.odom_path)

        if self.latest_cmd_pose is not None:
            self.append_pose(self.cmd_path, self.latest_cmd_pose)
            self.cmd_path.header.stamp = now
            self.cmd_path_pub.publish(self.cmd_path)

    def append_pose(self, path, pose):
        pose.header.frame_id = self.frame_id
        if path.poses:
            last = path.poses[-1].pose.position
            current = pose.pose.position
            step = math.sqrt(
                (current.x - last.x) ** 2
                + (current.y - last.y) ** 2
                + (current.z - last.z) ** 2
            )
            if step < self.min_path_step_m:
                return

        path.poses.append(pose)
        if self.max_path_length > 0 and len(path.poses) > self.max_path_length:
            del path.poses[: len(path.poses) - self.max_path_length]

    def publish_gate_marker(self, now):
        marker = self.base_marker(now, "current_gate", 0, Marker.SPHERE)
        if self.odom is None or self.gate_rel is None or self.gate_stamp is None:
            marker.action = Marker.DELETE
            self.gate_marker_pub.publish(marker)
            return
        if (now - self.gate_stamp).to_sec() > self.gate_timeout_s:
            marker.action = Marker.DELETE
            self.gate_marker_pub.publish(marker)
            return

        pos = self.odom.pose.pose.position
        c = math.cos(self.odom_yaw)
        s = math.sin(self.odom_yaw)
        marker.pose.position.x = pos.x + c * self.gate_rel.x - s * self.gate_rel.y
        marker.pose.position.y = pos.y + s * self.gate_rel.x + c * self.gate_rel.y
        marker.pose.position.z = pos.z + self.gate_relative_z_sign * self.gate_rel.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.gate_marker_scale
        marker.scale.y = self.gate_marker_scale
        marker.scale.z = self.gate_marker_scale
        marker.color = color(0.1, 1.0, 0.2, 0.85)
        self.gate_marker_pub.publish(marker)

    def publish_visual_error_marker(self, now):
        marker = self.base_marker(now, "visual_error", 0, Marker.ARROW)
        if (
            self.odom is None
            or self.visual_error is None
            or self.visual_error_stamp is None
        ):
            marker.action = Marker.DELETE
            self.visual_marker_pub.publish(marker)
            return
        if (now - self.visual_error_stamp).to_sec() > self.visual_error_timeout_s:
            marker.action = Marker.DELETE
            self.visual_marker_pub.publish(marker)
            return

        pos = self.odom.pose.pose.position
        start = point_from_xyz(pos.x, pos.y, pos.z - 0.8)
        right_x = -math.sin(self.odom_yaw)
        right_y = math.cos(self.odom_yaw)
        vx = clamp(self.visual_error.x, -1.0, 1.0) * self.visual_arrow_scale
        vz = clamp(self.visual_error.y, -1.0, 1.0) * self.visual_arrow_scale
        end = point_from_xyz(
            start.x + right_x * vx,
            start.y + right_y * vx,
            start.z + vz,
        )
        marker.points = [start, end]
        marker.scale.x = 0.08
        marker.scale.y = 0.24
        marker.scale.z = 0.35
        marker.color = color(1.0, 0.1, 1.0, 0.9)
        self.visual_marker_pub.publish(marker)

    def publish_yaw_marker(self, now):
        marker = self.base_marker(now, "yaw", 0, Marker.LINE_LIST)
        if self.odom is None:
            marker.action = Marker.DELETE
            self.yaw_marker_pub.publish(marker)
            return

        pos = self.odom.pose.pose.position
        origin = point_from_xyz(pos.x, pos.y, pos.z - 0.35)
        actual_end = point_from_xyz(
            origin.x + math.cos(self.odom_yaw) * self.yaw_arrow_length,
            origin.y + math.sin(self.odom_yaw) * self.yaw_arrow_length,
            origin.z,
        )
        marker.points = [origin, actual_end]
        marker.colors = [
            color(1.0, 0.7, 0.0, 1.0),
            color(1.0, 0.7, 0.0, 1.0),
        ]

        if self.expected_path_yaw is not None:
            expected_end = point_from_xyz(
                origin.x + math.cos(self.expected_path_yaw) * self.yaw_arrow_length,
                origin.y + math.sin(self.expected_path_yaw) * self.yaw_arrow_length,
                origin.z,
            )
            marker.points.extend([origin, expected_end])
            marker.colors.extend(
                [
                    color(0.0, 0.8, 1.0, 1.0),
                    color(0.0, 0.8, 1.0, 1.0),
                ]
            )

        marker.scale.x = 0.08
        marker.color = color(1.0, 1.0, 1.0, 1.0)
        self.yaw_marker_pub.publish(marker)

    def base_marker(self, stamp, namespace, marker_id, marker_type):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.25)
        return marker


if __name__ == "__main__":
    try:
        LiveDebugViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
