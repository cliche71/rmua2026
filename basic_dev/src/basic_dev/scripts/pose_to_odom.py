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
        self.min_dt = float(rospy.get_param("~min_dt", 0.005))
        self.max_dt = float(rospy.get_param("~max_dt", 0.05))
        self.linear_velocity_cutoff_hz = float(
            rospy.get_param("~linear_velocity_cutoff_hz", 8.0)
        )
        self.angular_velocity_cutoff_hz = float(
            rospy.get_param("~angular_velocity_cutoff_hz", 10.0)
        )
        self.max_linear_velocity = float(rospy.get_param("~max_linear_velocity", 3.0))
        self.max_linear_acceleration = float(
            rospy.get_param("~max_linear_acceleration", 100.0)
        )
        self.max_angular_velocity = float(
            rospy.get_param("~max_angular_velocity", 8.0)
        )
        self.reset_after_invalid_count = int(
            rospy.get_param("~reset_after_invalid_count", 5)
        )
        self.diagnostics_period = float(rospy.get_param("~diagnostics_period", 1.0))

        self.last_receive_time = None
        self.last_position = None
        self.last_quat = None
        self.filtered_linear_velocity = (0.0, 0.0, 0.0)
        self.filtered_angular_velocity = (0.0, 0.0, 0.0)
        self.consecutive_invalid_count = 0

        self.stats_start_time = None
        self.dt_min = None
        self.dt_max = None
        self.valid_dt_count = 0
        self.invalid_dt_count = 0
        self.invalid_linear_velocity_count = 0
        self.invalid_angular_velocity_count = 0
        self.raw_linear_velocity_max = 0.0
        self.filtered_linear_velocity_max = 0.0
        self.raw_angular_velocity_max = 0.0
        self.published_angular_velocity_max = 0.0
        self.source_stamp_age_sum = 0.0
        self.source_stamp_age_max = 0.0
        self.source_stamp_age_count = 0

        self.pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)
        self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)

        rospy.loginfo("pose_to_odom started.")
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("odom_topic: %s", self.odom_topic)
        rospy.loginfo(
            "pose_to_odom params: dt=[%.4f, %.4f] linear_cutoff=%.2fHz angular_cutoff=%.2fHz max_linear_velocity=%.2f max_linear_acceleration=%.2f max_angular_velocity=%.2f",
            self.min_dt,
            self.max_dt,
            self.linear_velocity_cutoff_hz,
            self.angular_velocity_cutoff_hz,
            self.max_linear_velocity,
            self.max_linear_acceleration,
            self.max_angular_velocity,
        )

    @staticmethod
    def vec_add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def vec_sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def vec_scale(a, scale):
        return (a[0] * scale, a[1] * scale, a[2] * scale)

    @staticmethod
    def vec_norm(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    @staticmethod
    def low_pass_alpha(dt, cutoff_hz):
        if cutoff_hz <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        return max(0.0, min(1.0, dt / (tau + dt)))

    def low_pass_vec(self, previous, current, dt, cutoff_hz):
        alpha = self.low_pass_alpha(dt, cutoff_hz)
        delta = self.vec_sub(current, previous)
        return self.vec_add(previous, self.vec_scale(delta, alpha))

    def update_dt_stats(self, dt):
        if self.dt_min is None:
            self.dt_min = dt
            self.dt_max = dt
        else:
            self.dt_min = min(self.dt_min, dt)
            self.dt_max = max(self.dt_max, dt)
        self.valid_dt_count += 1

    def update_source_stamp_stats(self, receive_time, source_stamp):
        if source_stamp == rospy.Time(0):
            return
        age = (receive_time - source_stamp).to_sec()
        if not math.isfinite(age):
            return
        self.source_stamp_age_sum += age
        self.source_stamp_age_max = max(self.source_stamp_age_max, age)
        self.source_stamp_age_count += 1

    def log_stats_if_needed(self, receive_time):
        if self.stats_start_time is None:
            self.stats_start_time = receive_time
            return

        elapsed = (receive_time - self.stats_start_time).to_sec()
        if elapsed < self.diagnostics_period:
            return

        source_age_avg = (
            self.source_stamp_age_sum / self.source_stamp_age_count
            if self.source_stamp_age_count > 0
            else 0.0
        )
        rospy.loginfo(
            "pose_to_odom stats: receive_dt_min=%.4f receive_dt_max=%.4f valid_dt=%d invalid_dt=%d invalid_linear=%d invalid_angular=%d raw_linear_velocity_max=%.3f filtered_linear_velocity_max=%.3f raw_angular_velocity_max=%.3f published_angular_velocity_max=%.3f source_stamp_age_avg=%.3f source_stamp_age_max=%.3f",
            self.dt_min if self.dt_min is not None else 0.0,
            self.dt_max if self.dt_max is not None else 0.0,
            self.valid_dt_count,
            self.invalid_dt_count,
            self.invalid_linear_velocity_count,
            self.invalid_angular_velocity_count,
            self.raw_linear_velocity_max,
            self.filtered_linear_velocity_max,
            self.raw_angular_velocity_max,
            self.published_angular_velocity_max,
            source_age_avg,
            self.source_stamp_age_max,
        )

        self.stats_start_time = receive_time
        self.dt_min = None
        self.dt_max = None
        self.valid_dt_count = 0
        self.invalid_dt_count = 0
        self.invalid_linear_velocity_count = 0
        self.invalid_angular_velocity_count = 0
        self.raw_linear_velocity_max = 0.0
        self.filtered_linear_velocity_max = 0.0
        self.raw_angular_velocity_max = 0.0
        self.published_angular_velocity_max = 0.0
        self.source_stamp_age_sum = 0.0
        self.source_stamp_age_max = 0.0
        self.source_stamp_age_count = 0

    def pose_cb(self, msg):
        receive_time = rospy.Time.now()
        source_stamp = msg.header.stamp
        self.update_source_stamp_stats(receive_time, source_stamp)

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
        if self.last_quat is not None:
            dot = (
                quat[0] * self.last_quat[0]
                + quat[1] * self.last_quat[1]
                + quat[2] * self.last_quat[2]
                + quat[3] * self.last_quat[3]
            )
            if dot < 0.0:
                quat = (-quat[0], -quat[1], -quat[2], -quat[3])

        linear_velocity = self.filtered_linear_velocity
        angular_velocity = self.filtered_angular_velocity

        if self.last_receive_time is not None:
            dt = (receive_time - self.last_receive_time).to_sec()
            if not math.isfinite(dt) or dt <= self.min_dt or dt > self.max_dt:
                self.invalid_dt_count += 1
                self.consecutive_invalid_count += 1
                if self.consecutive_invalid_count >= self.reset_after_invalid_count:
                    self.filtered_linear_velocity = (0.0, 0.0, 0.0)
                    self.filtered_angular_velocity = (0.0, 0.0, 0.0)
                    linear_velocity = self.filtered_linear_velocity
                    angular_velocity = self.filtered_angular_velocity
            else:
                self.update_dt_stats(dt)
                self.consecutive_invalid_count = 0
                raw_linear_velocity = self.vec_scale(
                    self.vec_sub(position, self.last_position), 1.0 / dt
                )
                raw_linear_norm = self.vec_norm(raw_linear_velocity)
                self.raw_linear_velocity_max = max(
                    self.raw_linear_velocity_max, raw_linear_norm
                )
                raw_linear_acceleration = self.vec_scale(
                    self.vec_sub(raw_linear_velocity, self.filtered_linear_velocity),
                    1.0 / dt,
                )
                linear_accel_valid = (
                    self.max_linear_acceleration <= 0.0
                    or self.vec_norm(raw_linear_acceleration)
                    <= self.max_linear_acceleration
                )
                linear_valid = (
                    raw_linear_norm <= self.max_linear_velocity and linear_accel_valid
                )
                if linear_valid:
                    self.filtered_linear_velocity = self.low_pass_vec(
                        self.filtered_linear_velocity,
                        raw_linear_velocity,
                        dt,
                        self.linear_velocity_cutoff_hz,
                    )
                else:
                    self.invalid_linear_velocity_count += 1
                linear_velocity = self.filtered_linear_velocity

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
                raw_angular_velocity = (0.0, 0.0, 0.0)
                if axis_norm > 1.0e-9:
                    raw_angular_velocity = (
                        dq[0] / axis_norm * angle / dt,
                        dq[1] / axis_norm * angle / dt,
                        dq[2] / axis_norm * angle / dt,
                    )
                raw_angular_norm = self.vec_norm(raw_angular_velocity)
                self.raw_angular_velocity_max = max(
                    self.raw_angular_velocity_max, raw_angular_norm
                )
                if raw_angular_norm <= self.max_angular_velocity:
                    self.filtered_angular_velocity = self.low_pass_vec(
                        self.filtered_angular_velocity,
                        raw_angular_velocity,
                        dt,
                        self.angular_velocity_cutoff_hz,
                    )
                else:
                    self.invalid_angular_velocity_count += 1
                angular_velocity = self.filtered_angular_velocity

        self.filtered_linear_velocity_max = max(
            self.filtered_linear_velocity_max, self.vec_norm(linear_velocity)
        )
        self.published_angular_velocity_max = max(
            self.published_angular_velocity_max, self.vec_norm(angular_velocity)
        )

        odom = Odometry()
        odom.header.stamp = receive_time
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position = msg.pose.position
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        odom.twist.twist.linear.x = linear_velocity[0]
        odom.twist.twist.linear.y = linear_velocity[1]
        odom.twist.twist.linear.z = linear_velocity[2]
        odom.twist.twist.angular.x = angular_velocity[0]
        odom.twist.twist.angular.y = angular_velocity[1]
        odom.twist.twist.angular.z = angular_velocity[2]
        self.pub.publish(odom)

        self.last_receive_time = receive_time
        self.last_position = position
        self.last_quat = quat
        self.log_stats_if_needed(receive_time)


if __name__ == "__main__":
    node = PoseToOdom()
    rospy.spin()
