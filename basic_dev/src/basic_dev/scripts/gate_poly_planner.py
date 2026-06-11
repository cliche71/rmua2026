#!/usr/bin/env python3
import json
import math
import threading

import numpy as np
import rospy
from airsim_ros.msg import VelCmd
from geometry_msgs.msg import PoseStamped, Transform, Twist, Vector3Stamped
from std_msgs.msg import String
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


TAKEOFF = "TAKEOFF"
SEARCH_GATE = "SEARCH_GATE"
TRACK_GATE = "TRACK_GATE"
CLEAR_GATE = "CLEAR_GATE"
CRUISE_SEARCH_NEXT_GATE = "CRUISE_SEARCH_NEXT_GATE"
ADVANCE_SEARCH_NEXT_GATE = "ADVANCE_SEARCH_NEXT_GATE"
CONFIRM_NEXT_GATE = "CONFIRM_NEXT_GATE"
TRACK_NEXT_GATE = "TRACK_NEXT_GATE"


def clamp(value, limit, upper=None):
    if upper is None:
        return max(-limit, min(limit, value))
    return max(limit, min(upper, value))


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) * (a[0] - b[0])
        + (a[1] - b[1]) * (a[1] - b[1])
        + (a[2] - b[2]) * (a[2] - b[2])
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def normalize_vector(vec, fallback):
    length = norm(vec)
    if length < 1.0e-6:
        return fallback
    return (vec[0] / length, vec[1] / length, vec[2] / length)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_lerp(start, end, weight):
    weight = max(0.0, min(1.0, weight))
    return wrap_angle(start + wrap_angle(end - start) * weight)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def roll_pitch_from_quaternion(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(0.5 * math.pi, sinp)
    else:
        pitch = math.asin(sinp)
    return roll, pitch


def fill_quaternion_from_yaw(rotation, yaw):
    rotation.x = 0.0
    rotation.y = 0.0
    rotation.z = math.sin(0.5 * yaw)
    rotation.w = math.cos(0.5 * yaw)


def basis_tau(degree, derivative_order, tau):
    if degree < 0 or derivative_order < 0 or derivative_order > degree:
        raise ValueError("invalid degree or derivative order")

    values = np.zeros(degree + 1)
    for k in range(derivative_order, degree + 1):
        coef = math.factorial(k) / math.factorial(k - derivative_order)
        values[k] = coef * (tau ** (k - derivative_order))
    return values


def build_q_unit(degree, snap_order):
    if degree < 0 or snap_order < 0 or snap_order > degree:
        raise ValueError("invalid degree or snap order")

    size = degree + 1
    q = np.zeros((size, size))
    for i in range(snap_order, size):
        for j in range(snap_order, size):
            coef = 1.0
            for kk in range(snap_order):
                coef *= float(i - kk) * float(j - kk)
            power = (i - snap_order) + (j - snap_order) + 1
            q[i, j] = coef / float(power)
    return q


def solve_min_snap_1d(
    waypoints_1d,
    times,
    degree=7,
    snap_order=4,
    regularization=1.0e-9,
    boundary_derivs_mode="free",
    waypoint_derivatives=None,
):
    if len(waypoints_1d) < 2 or len(waypoints_1d) != len(times):
        raise ValueError("waypoints and times must have the same size >= 2")
    if snap_order < 1 or snap_order > degree:
        raise ValueError("invalid snap order")
    if boundary_derivs_mode not in ("free", "fix", "match"):
        raise ValueError("boundary_derivs_mode must be free, fix, or match")

    n_wp = len(waypoints_1d)
    n_seg = n_wp - 1
    n_coeff = n_seg * (degree + 1)
    n_total = n_wp * snap_order

    def deriv_index(waypoint_index, derivative_order):
        return derivative_order * n_wp + waypoint_index

    q_unit = build_q_unit(degree, snap_order)
    q_big = np.zeros((n_coeff, n_coeff))
    durations = []
    for seg in range(n_seg):
        duration = times[seg + 1] - times[seg]
        if duration <= 0.0:
            raise ValueError("segment durations must be positive")
        durations.append(duration)
        scale = 1.0 / (duration ** (2 * snap_order - 1))
        i0 = seg * (degree + 1)
        q_big[i0 : i0 + degree + 1, i0 : i0 + degree + 1] = q_unit * scale

    rows = []
    maps = []
    for seg in range(n_seg):
        i0 = seg * (degree + 1)
        duration = durations[seg]

        for waypoint_index, tau in ((seg, 0.0), (seg + 1, 1.0)):
            for deriv in range(snap_order):
                scale = 1.0 if deriv == 0 else 1.0 / (duration**deriv)
                row = np.zeros(n_coeff)
                mapping = np.zeros(n_total)
                row[i0 : i0 + degree + 1] = scale * basis_tau(
                    degree, deriv, tau
                )
                mapping[deriv_index(waypoint_index, deriv)] = 1.0
                rows.append(row.copy())
                maps.append(mapping.copy())

    fixed_values = {}
    for waypoint_index, value in enumerate(waypoints_1d):
        fixed_values[deriv_index(waypoint_index, 0)] = float(value)

    if boundary_derivs_mode == "fix":
        for deriv in range(1, snap_order):
            fixed_values.setdefault(deriv_index(0, deriv), 0.0)
            fixed_values.setdefault(deriv_index(n_wp - 1, deriv), 0.0)

    if waypoint_derivatives is not None:
        for waypoint_index, derivatives in enumerate(waypoint_derivatives):
            if derivatives is None:
                continue
            for deriv, value in derivatives.items():
                if deriv <= 0 or deriv >= snap_order or value is None:
                    continue
                fixed_values[deriv_index(waypoint_index, deriv)] = float(value)

    if boundary_derivs_mode == "match":
        # Without explicit endpoint derivative constraints, keep the start/end
        # derivatives free. Fixed pass-through velocities are supplied through
        # waypoint_derivatives when flying gates.
        pass

    if len(rows) != n_coeff:
        raise RuntimeError("constraint matrix size mismatch")

    a = np.vstack(rows)
    ct = np.vstack(maps)

    a_inv = np.linalg.pinv(a)
    ai_mc = a_inv.dot(ct)
    r = ai_mc.T.dot(q_big).dot(ai_mc)

    fixed_idx = sorted(fixed_values.keys())
    fixed_vals = np.array([fixed_values[idx] for idx in fixed_idx])
    free_idx = [idx for idx in range(n_total) if idx not in fixed_values]

    d = np.zeros(n_total)
    if fixed_idx:
        d[fixed_idx] = fixed_vals

    if free_idx:
        rff = r[np.ix_(free_idx, free_idx)]
        if regularization > 0.0:
            rff = rff + regularization * np.eye(len(free_idx))
        if fixed_idx:
            r_fixed_free = r[np.ix_(fixed_idx, free_idx)]
            d_free = -np.linalg.pinv(rff).dot(r_fixed_free.T).dot(fixed_vals)
        else:
            d_free = np.zeros(len(free_idx))
        d[free_idx] = d_free

    p = ai_mc.dot(d)
    coeffs = np.zeros((n_seg, degree + 1))
    for seg in range(n_seg):
        coeffs[seg, :] = p[seg * (degree + 1) : (seg + 1) * (degree + 1)]
    return coeffs


class PolynomialTrajectory:
    def __init__(self, coeffs_xyz, times, degree):
        self.coeffs_xyz = coeffs_xyz
        self.times = times
        self.degree = degree
        self.n_seg = len(times) - 1

    def locate(self, t):
        t_start = self.times[0]
        t_end = self.times[-1]
        t_clipped = max(t_start, min(t_end - 1.0e-8, t))
        seg = 0
        for idx in range(self.n_seg):
            if self.times[idx] <= t_clipped < self.times[idx + 1]:
                seg = idx
                break

        duration = self.times[seg + 1] - self.times[seg]
        tau = (t_clipped - self.times[seg]) / duration
        return seg, tau, duration

    def eval(self, t, order):
        if order < 0:
            raise ValueError("order must be non-negative")
        if order > self.degree:
            return (0.0, 0.0, 0.0)

        seg, tau, duration = self.locate(t)
        scale = 1.0 if order == 0 else 1.0 / (duration**order)
        out = []
        basis = basis_tau(self.degree, order, tau)
        for dim in range(3):
            out.append(float(scale * self.coeffs_xyz[dim][seg, :].dot(basis)))
        return tuple(out)


class FramePolynomialTrajectory:
    def __init__(self, local_trajectory, origin, axes):
        self.local_trajectory = local_trajectory
        self.origin = origin
        self.axes = axes

    def eval(self, t, order):
        local = self.local_trajectory.eval(t, order)
        out = (
            self.axes[0][0] * local[0]
            + self.axes[1][0] * local[1]
            + self.axes[2][0] * local[2],
            self.axes[0][1] * local[0]
            + self.axes[1][1] * local[1]
            + self.axes[2][1] * local[2],
            self.axes[0][2] * local[0]
            + self.axes[1][2] * local[1]
            + self.axes[2][2] * local[2],
        )
        if order == 0:
            return (
                self.origin[0] + out[0],
                self.origin[1] + out[1],
                self.origin[2] + out[2],
            )
        return out


def world_to_frame(point, origin, axes):
    rel = (
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    )
    return (
        dot(rel, axes[0]),
        dot(rel, axes[1]),
        dot(rel, axes[2]),
    )


def build_min_snap_trajectory(
    waypoints,
    times,
    degree,
    snap_order,
    regularization,
    boundary_derivs_mode,
    waypoint_velocities=None,
    waypoint_accelerations=None,
):
    coeffs_xyz = []
    for dim in range(3):
        wp_1d = [point[dim] for point in waypoints]
        waypoint_derivatives = []
        for idx in range(len(waypoints)):
            derivatives = {}
            if waypoint_velocities is not None and waypoint_velocities[idx] is not None:
                value = waypoint_velocities[idx][dim]
                if value is not None:
                    derivatives[1] = value
            if (
                waypoint_accelerations is not None
                and waypoint_accelerations[idx] is not None
            ):
                value = waypoint_accelerations[idx][dim]
                if value is not None:
                    derivatives[2] = value
            waypoint_derivatives.append(derivatives)
        coeffs_xyz.append(
            solve_min_snap_1d(
                wp_1d,
                times,
                degree=degree,
                snap_order=snap_order,
                regularization=regularization,
                boundary_derivs_mode=boundary_derivs_mode,
                waypoint_derivatives=waypoint_derivatives,
            )
        )
    return PolynomialTrajectory(coeffs_xyz, times, degree)


def build_min_snap_trajectory_in_frame(
    waypoints,
    times,
    degree,
    snap_order,
    regularization,
    boundary_derivs_mode,
    waypoint_velocities,
    waypoint_accelerations,
    origin,
    axes,
):
    local_waypoints = [world_to_frame(point, origin, axes) for point in waypoints]
    local_trajectory = build_min_snap_trajectory(
        local_waypoints,
        times,
        degree,
        snap_order,
        regularization,
        boundary_derivs_mode,
        waypoint_velocities,
        waypoint_accelerations,
    )
    return FramePolynomialTrajectory(local_trajectory, origin, axes)


def eval_quintic(p0, p1, duration, elapsed):
    if duration <= 1.0e-3:
        return p1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    u = max(0.0, min(1.0, elapsed / duration))
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    ds = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    dds = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration * duration)

    delta = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
    pos = (p0[0] + delta[0] * s, p0[1] + delta[1] * s, p0[2] + delta[2] * s)
    vel = (delta[0] * ds, delta[1] * ds, delta[2] * ds)
    acc = (delta[0] * dds, delta[1] * dds, delta[2] * dds)
    return pos, vel, acc


class GatePolyPlanner:
    def __init__(self):
        rospy.init_node("gate_poly_planner")

        self.command_mode = rospy.get_param("~command_mode", "vel").strip().lower()
        if self.command_mode not in ("vel", "trajectory"):
            rospy.logwarn(
                "unsupported command_mode '%s', falling back to vel", self.command_mode
            )
            self.command_mode = "vel"

        default_cmd_topic = (
            "/command/trajectory"
            if self.command_mode == "trajectory"
            else "/airsim_node/drone_1/vel_body_cmd"
        )
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/airsim_node/drone_1/debug/pose_gt"
        )
        self.gate_topic = rospy.get_param("~gate_topic", "/gate/relative_pose")
        self.gate_visual_error_topic = rospy.get_param(
            "~gate_visual_error_topic", "/gate/visual_error"
        )
        self.cmd_topic = rospy.get_param("~cmd_topic", default_cmd_topic)
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.child_frame_id = rospy.get_param("~child_frame_id", "drone_1")
        self.enable_takeoff_hold = rospy.get_param("~enable_takeoff_hold", True)
        self.takeoff_height_m = rospy.get_param("~takeoff_height_m", 1.0)
        self.takeoff_reached_tolerance_m = rospy.get_param(
            "~takeoff_reached_tolerance_m", 0.15
        )

        self.control_rate = rospy.get_param("~control_rate", 30.0)
        self.cruise_speed = rospy.get_param("~cruise_speed", 0.65)
        self.trajectory_cruise_speed = rospy.get_param(
            "~trajectory_cruise_speed", 1.7
        )
        self.pass_speed_mps = rospy.get_param("~pass_speed_mps", 2.0)
        self.cruise_search_speed_mps = rospy.get_param(
            "~cruise_search_speed_mps", 1.8
        )
        self.max_speed_mps = rospy.get_param("~max_speed_mps", 2.5)
        self.max_acc_mps2 = rospy.get_param("~max_acc_mps2", 2.0)
        self.trajectory_max_lateral_speed_mps = rospy.get_param(
            "~trajectory_max_lateral_speed_mps", 0.9
        )
        self.trajectory_max_vertical_speed_mps = rospy.get_param(
            "~trajectory_max_vertical_speed_mps", 0.55
        )
        self.trajectory_constrain_lateral_vertical_derivatives = rospy.get_param(
            "~trajectory_constrain_lateral_vertical_derivatives", True
        )
        self.min_segment_time = rospy.get_param("~min_segment_time", 0.25)
        self.max_segment_time = rospy.get_param("~max_segment_time", 30.0)
        self.front_offset_m = rospy.get_param("~front_offset_m", 1.0)
        self.min_front_x_m = rospy.get_param("~min_front_x_m", 0.35)
        self.exit_offset_m = rospy.get_param("~exit_offset_m", 0.8)
        self.gate_trigger_width_m = rospy.get_param("~gate_trigger_width_m", 10.0)
        self.gate_trigger_height_m = rospy.get_param("~gate_trigger_height_m", 5.0)
        self.gate_trigger_length_m = rospy.get_param("~gate_trigger_length_m", 7.0)
        self.gate_trigger_exit_offset_m = rospy.get_param(
            "~gate_trigger_exit_offset_m", -1.0
        )
        self.gate_trigger_clear_margin_m = rospy.get_param(
            "~gate_trigger_clear_margin_m", 0.3
        )
        self.gate_target_z_offset_m = rospy.get_param("~gate_target_z_offset_m", 0.3)
        self.target_flight_z = rospy.get_param("~target_flight_z", -1.2)
        self.height_manager_enabled = rospy.get_param("~height_manager_enabled", True)
        self.height_reference_mode = rospy.get_param(
            "~height_reference_mode", "gate_center"
        )
        self.height_min_z = rospy.get_param("~height_min_z", -30.0)
        self.height_max_z = rospy.get_param("~height_max_z", -0.7)
        self.first_gate_height_min_z = rospy.get_param("~first_gate_height_min_z", -3.8)
        self.height_full_range_after_gate_index = int(
            rospy.get_param("~height_full_range_after_gate_index", 1)
        )
        self.height_ref_alpha = rospy.get_param("~height_ref_alpha", 1.0)
        self.height_ref_max_step_m = rospy.get_param("~height_ref_max_step_m", 2.0)
        self.height_ref_rate_mps = rospy.get_param("~height_ref_rate_mps", 3.0)
        self.first_gate_height_ref_max_step_m = rospy.get_param(
            "~first_gate_height_ref_max_step_m", 1.8
        )
        self.first_gate_height_ref_rate_mps = rospy.get_param(
            "~first_gate_height_ref_rate_mps", 2.5
        )
        self.height_gate_z_weight = rospy.get_param("~height_gate_z_weight", 1.0)
        self.height_gate_z_offset_m = rospy.get_param("~height_gate_z_offset_m", 0.0)
        self.height_gate_z_filter_alpha = rospy.get_param(
            "~height_gate_z_filter_alpha", 0.45
        )
        self.height_gate_z_max_step_m = rospy.get_param(
            "~height_gate_z_max_step_m", 2.00
        )
        self.height_gate_z_rate_mps = rospy.get_param(
            "~height_gate_z_rate_mps", 1.20
        )
        self.height_gate_z_freeze_on_recovery = rospy.get_param(
            "~height_gate_z_freeze_on_recovery", True
        )
        self.height_gate_z_freeze_on_yaw_mismatch = rospy.get_param(
            "~height_gate_z_freeze_on_yaw_mismatch", True
        )
        self.height_gate_z_freeze_on_large_attitude = rospy.get_param(
            "~height_gate_z_freeze_on_large_attitude", True
        )
        self.height_visual_error_gain_m = rospy.get_param(
            "~height_visual_error_gain_m", 0.45
        )
        self.height_visual_distance_gain = rospy.get_param(
            "~height_visual_distance_gain", 0.10
        )
        self.height_visual_error_sign = rospy.get_param(
            "~height_visual_error_sign", 1.0
        )
        self.height_down_correction_scale = rospy.get_param(
            "~height_down_correction_scale", 0.25
        )
        self.height_visual_deadband = rospy.get_param("~height_visual_deadband", 0.05)
        self.height_visual_timeout = rospy.get_param("~height_visual_timeout", 0.5)
        self.allow_height_replan_while_tracking = rospy.get_param(
            "~allow_height_replan_while_tracking", True
        )
        self.allow_height_down_replan_while_tracking = rospy.get_param(
            "~allow_height_down_replan_while_tracking", False
        )
        self.height_replan_visual_threshold = rospy.get_param(
            "~height_replan_visual_threshold", 0.12
        )
        self.height_replan_z_threshold_m = rospy.get_param(
            "~height_replan_z_threshold_m", 0.45
        )
        self.height_replan_max_step_m = rospy.get_param(
            "~height_replan_max_step_m", 0.70
        )
        self.height_replan_min_interval_s = rospy.get_param(
            "~height_replan_min_interval_s", 0.8
        )
        self.height_replan_max_abs_rel_z_m = rospy.get_param(
            "~height_replan_max_abs_rel_z_m", 10.0
        )
        self.height_replan_min_gate_distance_m = rospy.get_param(
            "~height_replan_min_gate_distance_m", 4.0
        )
        self.height_replan_active_gate_xy_tolerance_m = rospy.get_param(
            "~height_replan_active_gate_xy_tolerance_m", 6.0
        )
        self.height_replan_z_only_ignore_xy_error = rospy.get_param(
            "~height_replan_z_only_ignore_xy_error", True
        )
        self.height_replan_after_gate_index = int(
            rospy.get_param("~height_replan_after_gate_index", 1)
        )
        self.new_gate_direct_climb_height = rospy.get_param(
            "~new_gate_direct_climb_height", True
        )
        self.new_gate_direct_climb_max_step_m = rospy.get_param(
            "~new_gate_direct_climb_max_step_m", 1.0
        )
        self.gate_relative_z_sign = rospy.get_param("~gate_relative_z_sign", 1.0)
        self.max_vision_tilt_deg = rospy.get_param("~max_vision_tilt_deg", 45.0)
        self.use_min_snap = rospy.get_param("~use_min_snap", True)
        self.trajectory_min_snap_path_frame = rospy.get_param(
            "~trajectory_min_snap_path_frame", True
        )
        self.trajectory_lateral_overshoot_limit_m = rospy.get_param(
            "~trajectory_lateral_overshoot_limit_m", 1.0
        )
        self.gate_center_diag_interval_s = rospy.get_param(
            "~gate_center_diag_interval_s", 0.5
        )
        self.track_error_warn_forward_m = rospy.get_param(
            "~track_error_warn_forward_m", 3.0
        )
        self.track_error_warn_lateral_m = rospy.get_param(
            "~track_error_warn_lateral_m", 1.5
        )
        self.track_error_warn_vertical_m = rospy.get_param(
            "~track_error_warn_vertical_m", 1.2
        )
        self.poly_degree = rospy.get_param("~poly_degree", 7)
        self.snap_order = rospy.get_param("~snap_order", 4)
        self.regularization = rospy.get_param("~regularization", 1.0e-9)
        self.boundary_derivs_mode = rospy.get_param("~boundary_derivs_mode", "match")

        self.kp_pos = rospy.get_param("~kp_pos", 0.8)
        self.kp_x = rospy.get_param("~kp_x", self.kp_pos)
        self.kp_z = rospy.get_param("~kp_z", 0.45)
        self.max_vx = rospy.get_param("~max_vx", 0.9)
        self.max_vy = rospy.get_param("~max_vy", 0.6)
        self.max_vz = rospy.get_param("~max_vz", 0.4)
        self.trajectory_max_vx = rospy.get_param("~trajectory_max_vx", self.max_vx)
        self.trajectory_max_vy = rospy.get_param("~trajectory_max_vy", self.max_vy)
        self.trajectory_max_vz = rospy.get_param("~trajectory_max_vz", self.max_vz)
        self.trajectory_max_acc = rospy.get_param("~trajectory_max_acc", 1.2)
        self.trajectory_use_feedforward = rospy.get_param(
            "~trajectory_use_feedforward", True
        )
        self.trajectory_waypoint_hold = rospy.get_param(
            "~trajectory_waypoint_hold", False
        )
        self.trajectory_waypoint_tolerance_m = rospy.get_param(
            "~trajectory_waypoint_tolerance_m", 0.25
        )
        self.trajectory_waypoint_timeout = rospy.get_param(
            "~trajectory_waypoint_timeout", 8.0
        )
        self.trajectory_track_lookahead_m = rospy.get_param(
            "~trajectory_track_lookahead_m", 1.2
        )
        self.trajectory_sample_guard_enabled = rospy.get_param(
            "~trajectory_sample_guard_enabled", False
        )
        self.trajectory_sample_guard_max_deviation_m = rospy.get_param(
            "~trajectory_sample_guard_max_deviation_m", 1.0
        )
        self.trajectory_sample_guard_corridor_m = rospy.get_param(
            "~trajectory_sample_guard_corridor_m", 1.0
        )
        self.require_gate_clear_before_finish = rospy.get_param(
            "~require_gate_clear_before_finish", False
        )
        self.gate_clear_x_m = rospy.get_param("~gate_clear_x_m", 0.6)
        self.gate_clear_extra_m = rospy.get_param("~gate_clear_extra_m", 0.8)
        self.gate_clear_max_extensions = int(
            rospy.get_param("~gate_clear_max_extensions", 3)
        )
        self.clear_distance_m = rospy.get_param("~clear_distance_m", 0.6)
        self.gate_clear_require_center_check = rospy.get_param(
            "~gate_clear_require_center_check", True
        )
        self.gate_clear_lateral_margin_m = rospy.get_param(
            "~gate_clear_lateral_margin_m", 0.6
        )
        self.gate_clear_vertical_margin_m = rospy.get_param(
            "~gate_clear_vertical_margin_m", 0.4
        )
        self.gate_clear_center_miss_m = rospy.get_param(
            "~gate_clear_center_miss_m", 4.2
        )
        self.gate_clear_block_during_recovery = rospy.get_param(
            "~gate_clear_block_during_recovery", True
        )
        self.gate_clear_recovery_strict_enabled = rospy.get_param(
            "~gate_clear_recovery_strict_enabled", True
        )
        self.gate_clear_recovery_lateral_limit_m = rospy.get_param(
            "~gate_clear_recovery_lateral_limit_m", 2.5
        )
        self.gate_clear_recovery_vertical_limit_m = rospy.get_param(
            "~gate_clear_recovery_vertical_limit_m", 1.2
        )
        self.gate_clear_recovery_center_miss_m = rospy.get_param(
            "~gate_clear_recovery_center_miss_m", 2.8
        )
        self.gate_clear_recovery_behind_margin_m = rospy.get_param(
            "~gate_clear_recovery_behind_margin_m", 2.0
        )
        self.gate_clear_block_on_large_attitude = rospy.get_param(
            "~gate_clear_block_on_large_attitude", True
        )
        self.gate_clear_block_on_yaw_mismatch = rospy.get_param(
            "~gate_clear_block_on_yaw_mismatch", True
        )
        self.gate_clear_reject_replan_interval_s = rospy.get_param(
            "~gate_clear_reject_replan_interval_s", 0.8
        )
        self.planner_debug_topic = rospy.get_param(
            "~planner_debug_topic", "/gate_planner/debug"
        )
        self.search_next_forward_step_m = rospy.get_param(
            "~search_next_forward_step_m", 1.0
        )
        self.post_gate_settle_time = rospy.get_param("~post_gate_settle_time", 0.0)
        self.search_next_max_distance_m = rospy.get_param(
            "~search_next_max_distance_m", 0.0
        )
        self.search_next_speed_mps = rospy.get_param(
            "~search_next_speed_mps", self.cruise_search_speed_mps
        )
        self.new_gate_stable_frames = int(
            rospy.get_param("~new_gate_stable_frames", 3)
        )
        self.min_new_gate_distance_m = rospy.get_param("~min_new_gate_distance_m", 0.8)
        self.min_new_gate_world_forward_m = rospy.get_param(
            "~min_new_gate_world_forward_m", 1.0
        )
        self.search_gate_corridor_half_width_m = rospy.get_param(
            "~search_gate_corridor_half_width_m", 8.0
        )
        self.gate_reacquire_timeout = rospy.get_param("~gate_reacquire_timeout", 3.0)
        self.cleared_gate_ignore_radius_m = rospy.get_param(
            "~cleared_gate_ignore_radius_m", 2.0
        )
        self.new_gate_candidate_tolerance_m = rospy.get_param(
            "~new_gate_candidate_tolerance_m", 0.8
        )
        self.new_gate_requires_visual_error = rospy.get_param(
            "~new_gate_requires_visual_error", True
        )
        self.new_gate_reject_zero_visual_error = rospy.get_param(
            "~new_gate_reject_zero_visual_error", True
        )
        self.new_gate_zero_visual_epsilon = rospy.get_param(
            "~new_gate_zero_visual_epsilon", 1.0e-4
        )
        self.near_gate_zone_enabled = rospy.get_param(
            "~near_gate_zone_enabled", True
        )
        self.near_gate_zone_forward_m = rospy.get_param(
            "~near_gate_zone_forward_m", 6.0
        )
        self.near_gate_prefetch_forward_m = rospy.get_param(
            "~near_gate_prefetch_forward_m", 12.0
        )
        self.near_gate_next_min_distance_m = rospy.get_param(
            "~near_gate_next_min_distance_m", 6.0
        )
        self.near_gate_forbid_current_gate_yaw = rospy.get_param(
            "~near_gate_forbid_current_gate_yaw", True
        )
        self.advance_search_lookahead_m = rospy.get_param(
            "~advance_search_lookahead_m", 6.0
        )
        self.advance_search_speed_mps = rospy.get_param(
            "~advance_search_speed_mps", 1.6
        )
        self.max_gate_lateral_m = rospy.get_param("~max_gate_lateral_m", 1.2)
        self.max_gate_vertical_m = rospy.get_param("~max_gate_vertical_m", 10.0)
        self.vx_filter_alpha = rospy.get_param("~vx_filter_alpha", 0.35)
        self.vx_deadband = rospy.get_param("~vx_deadband", 0.04)
        self.max_vx_step_per_sec = rospy.get_param("~max_vx_step_per_sec", 0.8)
        self.vz_filter_alpha = rospy.get_param("~vz_filter_alpha", 0.25)
        self.vz_deadband = rospy.get_param("~vz_deadband", 0.03)
        self.accel_cmd = int(rospy.get_param("~accel_cmd", 4))

        self.k_yaw = rospy.get_param("~k_yaw", 1.8)
        self.max_yaw_rate = rospy.get_param("~max_yaw_rate", 25.0)
        self.yaw_tracking_sign = rospy.get_param("~yaw_tracking_sign", 1.0)
        self.search_yaw_rate = rospy.get_param("~search_yaw_rate", 10.0)
        self.trajectory_search_yaw_rate = rospy.get_param(
            "~trajectory_search_yaw_rate", 3.0
        )
        self.search_visual_yaw_enabled = rospy.get_param(
            "~search_visual_yaw_enabled", True
        )
        self.search_visual_yaw_gain_deg = rospy.get_param(
            "~search_visual_yaw_gain_deg", 35.0
        )
        self.search_visual_yaw_max_rate_deg = rospy.get_param(
            "~search_visual_yaw_max_rate_deg", 25.0
        )
        self.search_visual_yaw_deadband = rospy.get_param(
            "~search_visual_yaw_deadband", 0.08
        )
        self.search_visual_yaw_sign = rospy.get_param(
            "~search_visual_yaw_sign", 1.0
        )
        self.search_visual_min_area_ratio = rospy.get_param(
            "~search_visual_min_area_ratio", 0.005
        )
        self.search_visual_path_turn_rate_deg = rospy.get_param(
            "~search_visual_path_turn_rate_deg", 12.0
        )
        self.yaw_debug_log_enabled = rospy.get_param("~yaw_debug_log_enabled", True)
        self.yaw_debug_log_period = rospy.get_param("~yaw_debug_log_period", 0.5)
        self.trajectory_lock_yaw_to_path = rospy.get_param(
            "~trajectory_lock_yaw_to_path", True
        )
        self.gate_center_yaw_enabled = rospy.get_param(
            "~gate_center_yaw_enabled", True
        )
        self.gate_center_yaw_far_m = rospy.get_param(
            "~gate_center_yaw_far_m", 12.0
        )
        self.gate_center_yaw_near_m = rospy.get_param(
            "~gate_center_yaw_near_m", 4.0
        )
        self.trajectory_start_velocity_to_gate_center = rospy.get_param(
            "~trajectory_start_velocity_to_gate_center", True
        )
        self.max_vision_yaw_error_deg = rospy.get_param(
            "~max_vision_yaw_error_deg", 60.0
        )
        self.enable_heading_recovery = rospy.get_param(
            "~enable_heading_recovery", True
        )
        self.heading_recovery_yaw_error_deg = rospy.get_param(
            "~heading_recovery_yaw_error_deg", 95.0
        )
        self.heading_recovery_tilt_deg = rospy.get_param(
            "~heading_recovery_tilt_deg", 45.0
        )
        self.heading_recovery_forward_step_m = rospy.get_param(
            "~heading_recovery_forward_step_m", 1.5
        )
        self.heading_recovery_speed_mps = rospy.get_param(
            "~heading_recovery_speed_mps", 1.2
        )
        self.heading_recovery_level_speed_mps = rospy.get_param(
            "~heading_recovery_level_speed_mps", 0.8
        )
        self.heading_recovery_far_level_min_speed_mps = rospy.get_param(
            "~heading_recovery_far_level_min_speed_mps", 0.8
        )
        self.heading_recovery_hold_xy_when_yaw_disabled = rospy.get_param(
            "~heading_recovery_hold_xy_when_yaw_disabled", False
        )
        self.heading_recovery_max_vertical_speed_mps = rospy.get_param(
            "~heading_recovery_max_vertical_speed_mps", 0.25
        )
        self.heading_recovery_hold_altitude_while_leveling = rospy.get_param(
            "~heading_recovery_hold_altitude_while_leveling", True
        )
        self.heading_recovery_climb_while_leveling = rospy.get_param(
            "~heading_recovery_climb_while_leveling", False
        )
        self.heading_recovery_pause_plan_time = rospy.get_param(
            "~heading_recovery_pause_plan_time", True
        )
        self.heading_recovery_yaw_enable_tilt_deg = rospy.get_param(
            "~heading_recovery_yaw_enable_tilt_deg", 40.0
        )
        self.heading_recovery_partial_yaw_tilt_deg = rospy.get_param(
            "~heading_recovery_partial_yaw_tilt_deg", 45.0
        )
        self.heading_recovery_partial_yaw_rate_deg = rospy.get_param(
            "~heading_recovery_partial_yaw_rate_deg", 15.0
        )
        self.heading_recovery_yaw_enable_dwell_s = rospy.get_param(
            "~heading_recovery_yaw_enable_dwell_s", 0.4
        )
        self.heading_recovery_full_yaw_rate_tilt_deg = rospy.get_param(
            "~heading_recovery_full_yaw_rate_tilt_deg", 8.0
        )
        self.heading_recovery_yaw_gain = rospy.get_param(
            "~heading_recovery_yaw_gain", 1.0
        )
        self.heading_recovery_max_yaw_rate_deg = rospy.get_param(
            "~heading_recovery_max_yaw_rate_deg", 60.0
        )
        self.heading_recovery_exit_yaw_error_deg = rospy.get_param(
            "~heading_recovery_exit_yaw_error_deg", 35.0
        )
        self.heading_recovery_exit_tilt_deg = rospy.get_param(
            "~heading_recovery_exit_tilt_deg", 30.0
        )
        self.heading_recovery_target_mode = rospy.get_param(
            "~heading_recovery_target_mode", "gate_center"
        ).strip().lower()
        self.heading_recovery_min_forward_component = rospy.get_param(
            "~heading_recovery_min_forward_component", 0.2
        )
        self.heading_recovery_gate_center_weight = rospy.get_param(
            "~heading_recovery_gate_center_weight", 1.0
        )
        self.heading_recovery_max_target_angle_deg = rospy.get_param(
            "~heading_recovery_max_target_angle_deg", 35.0
        )
        self.enable_search = rospy.get_param("~enable_search", True)
        self.enable_yaw_tracking = rospy.get_param("~enable_yaw_tracking", True)
        self.gate_timeout = rospy.get_param("~gate_timeout", 1.0)
        self.min_replan_interval = rospy.get_param("~min_replan_interval", 0.8)
        self.replan_distance_m = rospy.get_param("~replan_distance_m", 0.7)
        self.allow_replan_while_tracking = rospy.get_param(
            "~allow_replan_while_tracking", True
        )
        self.stop_after_first_gate = rospy.get_param("~stop_after_first_gate", False)
        self.pending_gate_max_age = rospy.get_param("~pending_gate_max_age", 8.0)
        self.wait_for_gate_before_takeoff = rospy.get_param(
            "~wait_for_gate_before_takeoff", True
        )
        self.initial_gate_wait_timeout = rospy.get_param(
            "~initial_gate_wait_timeout", 5.0
        )

        self.got_pose = False
        self.pos_world = (0.0, 0.0, 0.0)
        self.initial_pos_world = None
        self.takeoff_target_world = None
        self.takeoff_reached = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.last_gate_rel = None
        self.last_gate_stamp = None
        self.last_visual_error = None
        self.last_visual_error_stamp = None
        self.height_ref_world = self.clamp_height(self.target_flight_z)
        self.height_ref_stamp = None
        self.filtered_gate_z_world = None
        self.filtered_gate_z_stamp = None
        self.last_height_replan_stamp = rospy.Time(0)
        self.state = (
            TAKEOFF
            if self.command_mode == "trajectory" and self.enable_takeoff_hold
            else SEARCH_GATE
        )
        self.target_gate_world = None
        self.active_gate_world = None
        self.aim_gate_world = None
        self.target_exit_world = None
        self.target_forward_dir_world = None
        self.pending_front_world = None
        self.pending_gate_world = None
        self.pending_exit_world = None
        self.pending_gate_rel = None
        self.pending_gate_stamp = None
        self.gate_completed = False
        self.gate_sequence_index = 0
        self.target_front_world = None
        self.cleared_gate_world_positions = []
        self.missed_gate_world_positions = []
        self.last_cleared_gate_world = None
        self.last_missed_gate_world = None
        self.search_forward_dir_world = None
        self.search_start_world = None
        self.search_next_altitude_world = None
        self.new_gate_candidate_world = None
        self.new_gate_candidate_count = 0
        self.new_gate_candidate_stamp = None
        self.next_gate_candidate_world = None
        self.next_gate_candidate_count = 0
        self.next_gate_candidate_stamp = None
        self.search_next_target_world = None
        self.search_next_target_stamp = None
        self.search_next_enter_time = None
        self.search_next_settle_target_world = None
        self.last_search_steer_stamp = None
        self.last_trajectory_sample_source = "none"
        self.initial_gate_wait_start = None
        self.initial_gate_wait_done = False
        self.heading_recovery_active = False
        self.heading_recovery_last_stamp = None
        self.heading_recovery_entry_z = None
        self.heading_recovery_altitude_z = None
        self.heading_recovery_yaw_level_since = None
        self.last_gate_clear_reject_stamp = rospy.Time(0)
        self.last_clear_decision = None
        self.last_command_position = None
        self.last_command_velocity = None
        self.last_command_acceleration = None
        self.last_command_yaw = None
        self.last_command_yaw_rate = None

        self.plan_lock = threading.RLock()
        self.plan = []
        self.plan_waypoints = []
        self.active_waypoint_index = 0
        self.active_waypoint_start = None
        self.gate_clear_extensions = 0
        self.trajectory = None
        self.plan_start = None
        self.plan_total_time = 0.0
        self.hold_position_world = None
        self.last_replan = rospy.Time(0)
        self.filtered_vx_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.last_cmd_time = None
        self.yaw_reference = None
        self.last_yaw_time = None

        if self.command_mode == "trajectory":
            self.cmd_pub = rospy.Publisher(
                self.cmd_topic, MultiDOFJointTrajectory, queue_size=1
            )
        else:
            self.cmd_pub = rospy.Publisher(self.cmd_topic, VelCmd, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, PoseStamped, self.pose_cb, queue_size=1
        )
        self.gate_sub = rospy.Subscriber(
            self.gate_topic, PoseStamped, self.gate_cb, queue_size=1
        )
        self.visual_error_sub = rospy.Subscriber(
            self.gate_visual_error_topic,
            Vector3Stamped,
            self.visual_error_cb,
            queue_size=1,
        )
        self.planner_debug_pub = rospy.Publisher(
            self.planner_debug_topic, String, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self.control_loop
        )
        self.diagnostic_timer = rospy.Timer(
            rospy.Duration(2.0), self.diagnostic_loop
        )

        rospy.loginfo("gate_poly_planner started.")
        rospy.loginfo("command_mode: %s", self.command_mode)
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("gate_topic: %s", self.gate_topic)
        rospy.loginfo("gate_visual_error_topic: %s", self.gate_visual_error_topic)
        rospy.loginfo("cmd_topic: %s", self.cmd_topic)
        rospy.loginfo(
            "gate plausibility limits: min_front_x=%.2f max_lateral=%.2f max_vertical=%.2f",
            self.min_front_x_m,
            self.max_gate_lateral_m,
            self.max_gate_vertical_m,
        )
        rospy.loginfo(
            "search next gate: step=%.2f settle=%.2f max_distance=%.2f speed=%.2f advance=(lookahead %.2f speed %.2f) near_gate=(enabled %s zone %.2f prefetch %.2f next_min %.2f forbid_current_yaw %s) stable_frames=%d ignore_radius=%.2f corridor=%.2f new_gate_visual=(required %s reject_zero %s eps %.1e)",
            self.search_next_forward_step_m,
            self.post_gate_settle_time,
            self.search_next_max_distance_m,
            self.search_next_speed_mps,
            self.advance_search_lookahead_m,
            self.advance_search_speed_mps,
            str(self.near_gate_zone_enabled),
            self.near_gate_zone_forward_m,
            self.near_gate_prefetch_forward_m,
            self.near_gate_next_min_distance_m,
            str(self.near_gate_forbid_current_gate_yaw),
            self.new_gate_stable_frames,
            self.cleared_gate_ignore_radius_m,
            self.search_gate_corridor_half_width_m,
            str(self.new_gate_requires_visual_error),
            str(self.new_gate_reject_zero_visual_error),
            self.new_gate_zero_visual_epsilon,
        )
        rospy.loginfo(
            "trajectory safety: track_lookahead=%.2f sample_guard=%s guard_deviation=%.2f lateral_overshoot_limit=%.2f search_yaw_rate=%.2f yaw_tracking=%s track_yaw_gain=%.2f track_yaw_max=%.1f track_yaw_sign=%.1f visual_search=%s visual_yaw_gain=%.1f visual_yaw_max=%.1f visual_yaw_sign=%.1f visual_path_turn=%.1f visual_min_area=%.3f lock_yaw_to_path=%s gate_center_yaw=%s yaw_blend=[near %.1f far %.1f] start_vel_to_gate=%s vision_yaw_error=%.1f heading_recovery=%s yaw_error=%.1f tilt=%.1f exit_yaw=%.1f exit_tilt=%.1f yaw_enable_tilt=%.1f partial_yaw_tilt=%.1f partial_yaw_rate=%.1f yaw_dwell=%.2f full_yaw_tilt=%.1f speed=%.2f level_speed=%.2f far_level_min_speed=%.2f hold_xy_yaw_disabled=%s vertical_speed=%.2f hold_level_alt=%s climb_while_level=%s pause_plan=%s yaw_rate_max=%.1f",
            self.trajectory_track_lookahead_m,
            str(self.trajectory_sample_guard_enabled),
            self.trajectory_sample_guard_max_deviation_m,
            self.trajectory_lateral_overshoot_limit_m,
            self.trajectory_search_yaw_rate,
            str(self.enable_yaw_tracking),
            self.k_yaw,
            self.max_yaw_rate,
            self.yaw_tracking_sign,
            str(self.search_visual_yaw_enabled),
            self.search_visual_yaw_gain_deg,
            self.search_visual_yaw_max_rate_deg,
            self.search_visual_yaw_sign,
            self.search_visual_path_turn_rate_deg,
            self.search_visual_min_area_ratio,
            str(self.trajectory_lock_yaw_to_path),
            str(self.gate_center_yaw_enabled),
            self.gate_center_yaw_near_m,
            self.gate_center_yaw_far_m,
            str(self.trajectory_start_velocity_to_gate_center),
            self.max_vision_yaw_error_deg,
            str(self.enable_heading_recovery),
            self.heading_recovery_yaw_error_deg,
            self.heading_recovery_tilt_deg,
            self.heading_recovery_exit_yaw_error_deg,
            self.heading_recovery_exit_tilt_deg,
            self.heading_recovery_yaw_enable_tilt_deg,
            self.heading_recovery_partial_yaw_tilt_deg,
            self.heading_recovery_partial_yaw_rate_deg,
            self.heading_recovery_yaw_enable_dwell_s,
            self.heading_recovery_full_yaw_rate_tilt_deg,
            self.heading_recovery_speed_mps,
            self.heading_recovery_level_speed_mps,
            self.heading_recovery_far_level_min_speed_mps,
            str(self.heading_recovery_hold_xy_when_yaw_disabled),
            self.heading_recovery_max_vertical_speed_mps,
            str(self.heading_recovery_hold_altitude_while_leveling),
            str(self.heading_recovery_climb_while_leveling),
            str(self.heading_recovery_pause_plan_time),
            self.heading_recovery_max_yaw_rate_deg,
        )
        rospy.loginfo(
            "yaw debug: enabled=%s period=%.2f",
            str(self.yaw_debug_log_enabled),
            self.yaw_debug_log_period,
        )
        rospy.loginfo(
            "height manager: enabled=%s mode=%s ref_z=%.2f limits=[%.2f, %.2f] first_gate_min=%.2f full_range_after_gate=%d alpha=%.2f gate_weight=%.2f gate_z_offset=%.2f gate_z_filter=(alpha=%.2f step=%.2f rate=%.2f freeze_recovery=%s freeze_yaw=%s freeze_att=%s) visual_gain=%.2f distance_gain=%.2f visual_sign=%.1f down_scale=%.2f max_step=%.2f rate=%.2f first_step=%.2f first_rate=%.2f direct_new_gate_climb=%s height_replan=%s down_replan=%s replan_after_gate=%d replan_step=%.2f replan_interval=%.2f replan_rel_z=%.2f replan_xy_tol=%.2f replan_ignore_xy=%s vision_tilt=%.1f",
            str(self.height_manager_enabled),
            self.height_reference_mode,
            self.height_ref_world,
            self.height_min_z,
            self.height_max_z,
            self.first_gate_height_min_z,
            self.height_full_range_after_gate_index,
            self.height_ref_alpha,
            self.height_gate_z_weight,
            self.height_gate_z_offset_m,
            self.height_gate_z_filter_alpha,
            self.height_gate_z_max_step_m,
            self.height_gate_z_rate_mps,
            str(self.height_gate_z_freeze_on_recovery),
            str(self.height_gate_z_freeze_on_yaw_mismatch),
            str(self.height_gate_z_freeze_on_large_attitude),
            self.height_visual_error_gain_m,
            self.height_visual_distance_gain,
            self.height_visual_error_sign,
            self.height_down_correction_scale,
            self.height_ref_max_step_m,
            self.height_ref_rate_mps,
            self.first_gate_height_ref_max_step_m,
            self.first_gate_height_ref_rate_mps,
            str(self.new_gate_direct_climb_height),
            str(self.allow_height_replan_while_tracking),
            str(self.allow_height_down_replan_while_tracking),
            self.height_replan_after_gate_index,
            self.height_replan_max_step_m,
            self.height_replan_min_interval_s,
            self.height_replan_max_abs_rel_z_m,
            self.height_replan_active_gate_xy_tolerance_m,
            str(self.height_replan_z_only_ignore_xy_error),
            self.max_vision_tilt_deg,
        )
        rospy.loginfo(
            "trajectory mode: use_min_snap=%s path_frame=%s waypoint_hold=%s trajectory_cruise_speed=%.2f pass_speed=%.2f cruise_search_speed=%.2f max_speed=%.2f max_acc=%.2f target_flight_z=%.2f min_segment_time=%.2f max_segment_time=%.2f",
            str(self.use_min_snap),
            str(self.trajectory_min_snap_path_frame),
            str(self.trajectory_waypoint_hold),
            self.trajectory_cruise_speed,
            self.pass_speed_mps,
            self.cruise_search_speed_mps,
            self.max_speed_mps,
            self.max_acc_mps2,
            self.target_flight_z,
            self.min_segment_time,
            self.max_segment_time,
        )
        rospy.loginfo(
            "trajectory axis limits: lateral_speed=%.2f vertical_speed=%.2f constrain_lat_vert_derivs=%s direct_climb_max_step=%.2f",
            self.trajectory_max_lateral_speed_mps,
            self.trajectory_max_vertical_speed_mps,
            str(self.trajectory_constrain_lateral_vertical_derivatives),
            self.new_gate_direct_climb_max_step_m,
        )
        rospy.loginfo(
            "gate center diagnostics: interval=%.2f warn_forward=%.2f warn_lateral=%.2f warn_vertical=%.2f",
            self.gate_center_diag_interval_s,
            self.track_error_warn_forward_m,
            self.track_error_warn_lateral_m,
            self.track_error_warn_vertical_m,
        )
        rospy.loginfo(
            "gate trigger volume: width=%.2f height=%.2f length=%.2f effective_exit_offset=%.2f effective_clear_distance=%.2f ignore_radius=%.2f",
            self.gate_trigger_width_m,
            self.gate_trigger_height_m,
            self.gate_trigger_length_m,
            self.effective_exit_offset_m(),
            self.effective_clear_distance_m(),
            self.effective_cleared_gate_ignore_radius_m(),
        )
        rospy.loginfo(
            "gate clear safety: center_check=%s lateral_limit=%.2f vertical_limit=%.2f center_miss=%.2f block_recovery=%s strict_recovery=%s strict_limits=(lat %.2f vert %.2f miss %.2f behind_margin %.2f) block_attitude=%s block_yaw=%s reject_replan_interval=%.2f recovery_target_mode=%s min_forward=%.2f gate_weight=%.2f max_target_angle=%.1f debug_topic=%s",
            str(self.gate_clear_require_center_check),
            self.gate_clear_lateral_limit_m(),
            self.gate_clear_vertical_limit_m(),
            self.gate_clear_center_miss_m,
            str(self.gate_clear_block_during_recovery),
            str(self.gate_clear_recovery_strict_enabled),
            self.gate_clear_recovery_lateral_limit_m,
            self.gate_clear_recovery_vertical_limit_m,
            self.gate_clear_recovery_center_miss_m,
            self.gate_clear_recovery_behind_margin_m,
            str(self.gate_clear_block_on_large_attitude),
            str(self.gate_clear_block_on_yaw_mismatch),
            self.gate_clear_reject_replan_interval_s,
            self.heading_recovery_target_mode,
            self.heading_recovery_min_forward_component,
            self.heading_recovery_gate_center_weight,
            self.heading_recovery_max_target_angle_deg,
            self.planner_debug_topic,
        )
        if (
            self.command_mode == "trajectory"
            and self.use_min_snap
            and self.trajectory_waypoint_hold
        ):
            rospy.logwarn(
                "trajectory_waypoint_hold is ignored because use_min_snap=True; sampling trajectory instead"
            )
        rospy.loginfo("state: %s", self.state)

    def pose_cb(self, msg):
        self.pos_world = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        if self.initial_pos_world is None:
            self.initial_pos_world = self.pos_world
            self.takeoff_target_world = (
                self.initial_pos_world[0],
                self.initial_pos_world[1],
                self.initial_pos_world[2] - self.takeoff_height_m,
            )
        self.roll, self.pitch = roll_pitch_from_quaternion(msg.pose.orientation)
        self.yaw = yaw_from_quaternion(msg.pose.orientation)
        if self.yaw_reference is None:
            self.yaw_reference = self.yaw
        if not self.is_attitude_valid_for_vision():
            rospy.logwarn_throttle(
                0.5,
                "large attitude angle: roll=%.1f pitch=%.1f deg, gate vision world transform gated because relative_to_world is yaw-only",
                math.degrees(self.roll),
                math.degrees(self.pitch),
            )
        self.got_pose = True

    def visual_error_cb(self, msg):
        self.last_visual_error = (
            msg.vector.x,
            msg.vector.y,
            msg.vector.z,
        )
        # AirSim image stamps are not always in the same time base as rospy.Time.now().
        # Use receipt time for freshness, otherwise valid visual errors can look stale.
        self.last_visual_error_stamp = rospy.Time.now()

    def gate_cb(self, msg):
        if self.gate_completed:
            return

        gate_rel = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        gate_stamp = msg.header.stamp
        if gate_stamp == rospy.Time(0):
            gate_stamp = rospy.Time.now()

        if not self.is_gate_measurement_plausible(gate_rel):
            rospy.logwarn_throttle(
                0.5,
                "reject implausible gate rel: x=%.2f y=%.2f z=%.2f reason=%s",
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
                self.gate_measurement_reject_reason(gate_rel),
            )
            return

        if not self.got_pose:
            self.last_gate_rel = gate_rel
            self.last_gate_stamp = gate_stamp
            return

        if not self.is_attitude_valid_for_vision():
            rospy.logwarn_throttle(
                0.5,
                "ignore gate detection during large attitude: roll=%.1f pitch=%.1f rel=(%.2f, %.2f, %.2f)",
                math.degrees(self.roll),
                math.degrees(self.pitch),
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
            )
            return

        heading_valid, expected_yaw, yaw_error = self.is_heading_valid_for_vision()
        allow_yaw_mismatch_search = self.state == ADVANCE_SEARCH_NEXT_GATE
        if not heading_valid and not allow_yaw_mismatch_search:
            rospy.logwarn_throttle(
                0.5,
                "ignore gate detection during yaw mismatch: yaw=%.1f expected=%.1f error=%.1f limit=%.1f rel=(%.2f, %.2f, %.2f)",
                math.degrees(self.yaw),
                math.degrees(expected_yaw),
                math.degrees(yaw_error),
                self.max_vision_yaw_error_deg,
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
            )
            return
        if not heading_valid and allow_yaw_mismatch_search:
            rospy.logwarn_throttle(
                0.5,
                "accept front-gate search candidate during yaw mismatch: yaw=%.1f expected=%.1f error=%.1f rel=(%.2f, %.2f, %.2f)",
                math.degrees(self.yaw),
                math.degrees(expected_yaw),
                math.degrees(yaw_error),
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
            )

        now = rospy.Time.now()
        gate_world_raw = self.relative_to_world(gate_rel)
        candidate_filter_world = self.apply_gate_target_z_offset(
            gate_world_raw,
            self.height_ref_world if self.command_mode == "trajectory" else None,
        )

        if self.is_in_cleared_gate_ignore_zone(candidate_filter_world):
            self.reset_new_gate_candidate()
            rospy.loginfo_throttle(
                1.0,
                "ignore gate candidate in cleared gate zone: world=(%.2f, %.2f, %.2f)",
                candidate_filter_world[0],
                candidate_filter_world[1],
                candidate_filter_world[2],
            )
            return

        if self.state in (
            CRUISE_SEARCH_NEXT_GATE,
            ADVANCE_SEARCH_NEXT_GATE,
            CONFIRM_NEXT_GATE,
        ):
            if self.is_search_next_settling(now):
                self.reset_new_gate_candidate()
                rospy.loginfo_throttle(
                    0.5,
                    "ignore gate candidate during optional post-gate settle",
                )
                return
            if not self.is_new_gate_detection_fresh(now):
                self.reset_new_gate_candidate()
                return

            if gate_rel[0] < self.min_new_gate_distance_m:
                self.reset_new_gate_candidate()
                rospy.loginfo_throttle(
                    1.0,
                    "ignore near gate candidate while searching next: rel_x=%.2f < %.2f",
                    gate_rel[0],
                    self.min_new_gate_distance_m,
                )
                return
            if not self.is_search_next_candidate_allowed(candidate_filter_world):
                self.reset_new_gate_candidate()
                return

            gate_world, front_world, exit_world = self.compute_gate_waypoints(
                gate_rel, now, update_height=True
            )
            if not self.is_gate_plan_candidate_valid(
                front_world,
                gate_world,
                exit_world,
                self.current_planning_forward_dir(),
                context="gate candidate",
            ):
                self.reset_new_gate_candidate()
                return
            self.update_search_altitude_preview(gate_world, gate_rel)
            self.last_gate_rel = gate_rel
            self.last_gate_stamp = gate_stamp
            self.set_state(CONFIRM_NEXT_GATE)
            if self.update_new_gate_candidate(gate_world, now):
                self.build_plan_from_waypoints(
                    front_world,
                    gate_world,
                    exit_world,
                    now,
                    is_next_gate=True,
                )
            return

        if self.command_mode == "trajectory" and self.is_takeoff_pending():
            gate_world, front_world, exit_world = self.compute_gate_waypoints(
                gate_rel, now, update_height=False
            )
            if not self.is_gate_plan_candidate_valid(
                front_world,
                gate_world,
                exit_world,
                self.current_planning_forward_dir(),
                context="gate candidate",
            ):
                return
            self.pending_front_world = front_world
            self.pending_gate_world = gate_world
            self.pending_exit_world = exit_world
            self.pending_gate_rel = gate_rel
            self.pending_gate_stamp = now
            rospy.loginfo_throttle(
                1.0,
                "cache gate detection until se3 takeoff hold is reached; height manager locked at %.2f",
                self.height_ref_world,
            )
            return

        if self.state == CLEAR_GATE:
            return

        self.last_gate_rel = gate_rel
        self.last_gate_stamp = gate_stamp

        if (
            self.command_mode == "trajectory"
            and self.plan
            and self.state in (TRACK_GATE, TRACK_NEXT_GATE)
        ):
            prefetched_next = self.try_prefetch_next_gate(gate_rel, now)
            if prefetched_next:
                return
            new_z = self.compute_active_gate_height_replan_z(gate_world_raw, gate_rel, now)
            if new_z is not None:
                self.replan_active_gate_height_z_only(
                    new_z,
                    now,
                    is_next_gate=(self.state == TRACK_NEXT_GATE),
                )
            return

        gate_world, front_world, exit_world = self.compute_gate_waypoints(
            gate_rel, now, update_height=True
        )
        if not self.is_gate_plan_candidate_valid(
            front_world,
            gate_world,
            exit_world,
            self.current_planning_forward_dir(),
            context="gate candidate",
        ):
            return
        should_replan = not self.plan
        if self.plan and not self.allow_replan_while_tracking:
            should_replan = False
        elif self.target_gate_world is not None:
            moved = distance(gate_world, self.target_gate_world)
            should_replan = should_replan or moved > self.replan_distance_m

        if should_replan and (now - self.last_replan).to_sec() >= self.min_replan_interval:
            self.build_plan_from_waypoints(front_world, gate_world, exit_world, now)

    def build_plan_from_waypoints(
        self, front_world, gate_world, exit_world, now, is_next_gate=False
    ):
        start_world = self.pos_world
        waypoints = [start_world, front_world, gate_world]
        if exit_world is not None:
            waypoints.append(exit_world)
        times = [0.0]
        forward_dir = self.compute_gate_forward_dir(front_world, gate_world, exit_world)
        approach_dir = self.compute_gate_center_approach_dir(
            start_world, gate_world, forward_dir
        )

        new_plan = []
        path_axes = self.build_path_frame_axes(forward_dir)
        path_origin = start_world
        for index in range(len(waypoints) - 1):
            p0 = waypoints[index]
            p1 = waypoints[index + 1]
            segment_distance = distance(p0, p1)
            if self.command_mode == "trajectory":
                speed = (
                    self.pass_speed_mps
                    if is_next_gate or index > 0
                    else self.trajectory_cruise_speed
                )
            else:
                speed = self.cruise_speed
            duration = segment_distance / max(0.05, speed)
            if self.command_mode == "trajectory":
                p0_local = world_to_frame(p0, path_origin, path_axes)
                p1_local = world_to_frame(p1, path_origin, path_axes)
                lateral_delta = abs(p1_local[1] - p0_local[1])
                vertical_delta = abs(p1_local[2] - p0_local[2])
                if self.trajectory_max_lateral_speed_mps > 0.0:
                    duration = max(
                        duration,
                        lateral_delta / self.trajectory_max_lateral_speed_mps,
                    )
                if self.trajectory_max_vertical_speed_mps > 0.0:
                    duration = max(
                        duration,
                        vertical_delta / self.trajectory_max_vertical_speed_mps,
                    )
            duration = max(self.min_segment_time, min(self.max_segment_time, duration))
            new_plan.append((p0, p1, duration))
            times.append(times[-1] + duration)

        waypoint_velocities, waypoint_accelerations = self.build_gate_derivatives(
            waypoints,
            forward_dir,
            is_next_gate,
            local_frame=self.trajectory_min_snap_path_frame,
            approach_dir=approach_dir,
            local_axes=path_axes,
        )
        new_trajectory = None
        if self.use_min_snap:
            try:
                if self.trajectory_min_snap_path_frame:
                    axes = self.build_path_frame_axes(forward_dir)
                    new_trajectory = build_min_snap_trajectory_in_frame(
                        waypoints,
                        times,
                        self.poly_degree,
                        self.snap_order,
                        self.regularization,
                        self.boundary_derivs_mode,
                        waypoint_velocities,
                        waypoint_accelerations,
                        start_world,
                        axes,
                    )
                    if not self.log_min_snap_path_frame_diagnostics(
                        waypoints, axes, new_trajectory, times
                    ):
                        raise RuntimeError("min-snap lateral overshoot exceeds limit")
                else:
                    new_trajectory = build_min_snap_trajectory(
                        waypoints,
                        times,
                        self.poly_degree,
                        self.snap_order,
                        self.regularization,
                        self.boundary_derivs_mode,
                        waypoint_velocities,
                        waypoint_accelerations,
                    )
            except Exception as exc:
                rospy.logerr("min-snap solve failed, abort trajectory plan: %s", str(exc))
                if self.command_mode == "trajectory":
                    with self.plan_lock:
                        if not self.plan:
                            self.plan = []
                            self.plan_waypoints = []
                            self.trajectory = None
                            self.plan_start = None
                            self.plan_total_time = 0.0
                            self.reset_new_gate_candidate()
                    return False

        new_plan_total_time = sum(segment[2] for segment in new_plan)
        yaw_forward_dir = None
        yaw_reference = None
        if self.command_mode == "trajectory" and self.trajectory_lock_yaw_to_path:
            yaw_forward_dir = self.horizontal_forward_dir(forward_dir)
            yaw_reference = math.atan2(yaw_forward_dir[1], yaw_forward_dir[0])

        with self.plan_lock:
            self.plan = new_plan
            self.plan_waypoints = waypoints[1:]
            self.active_waypoint_index = 0
            self.active_waypoint_start = now
            self.gate_clear_extensions = 0
            self.trajectory = new_trajectory
            self.plan_start = now
            self.plan_total_time = new_plan_total_time
            self.hold_position_world = None
            self.last_replan = now
            self.target_front_world = front_world
            self.target_gate_world = gate_world
            self.active_gate_world = gate_world
            self.aim_gate_world = gate_world
            self.target_exit_world = exit_world
            self.target_forward_dir_world = forward_dir
            if yaw_reference is not None:
                if self.enable_yaw_tracking:
                    self.yaw_reference = self.yaw
                else:
                    self.yaw_reference = yaw_reference
                self.last_yaw_time = now
            self.search_next_target_world = None
            self.search_next_target_stamp = None
            self.reset_new_gate_candidate()
            self.next_gate_candidate_world = None
            self.next_gate_candidate_count = 0
            self.next_gate_candidate_stamp = None
            self.set_state(TRACK_NEXT_GATE if is_next_gate else TRACK_GATE)

        if yaw_reference is not None:
            gate_bearing_yaw = math.atan2(approach_dir[1], approach_dir[0])
            velocity_yaw = (
                gate_bearing_yaw
                if self.trajectory_start_velocity_to_gate_center
                else yaw_reference
            )
            velocity_to_gate_error = wrap_angle(velocity_yaw - gate_bearing_yaw)
            rospy.loginfo(
                "trajectory yaw locked to path: yaw=%.1f forward_dir=(%.2f, %.2f) gate_bearing_yaw=%.1f velocity_yaw=%.1f velocity_to_gate_error=%.1f",
                math.degrees(yaw_reference),
                yaw_forward_dir[0],
                yaw_forward_dir[1],
                math.degrees(gate_bearing_yaw),
                math.degrees(velocity_yaw),
                math.degrees(velocity_to_gate_error),
            )

        rospy.loginfo(
            "planned gate trajectory: start=(%.2f, %.2f, %.2f) front=(%.2f, %.2f, %.2f) gate=(%.2f, %.2f, %.2f) exit=(%.2f, %.2f, %.2f) total=%.2fs",
            start_world[0],
            start_world[1],
            start_world[2],
            front_world[0],
            front_world[1],
            front_world[2],
            gate_world[0],
            gate_world[1],
            gate_world[2],
            exit_world[0] if exit_world is not None else gate_world[0],
            exit_world[1] if exit_world is not None else gate_world[1],
            exit_world[2] if exit_world is not None else gate_world[2],
            new_plan_total_time,
        )
        if is_next_gate:
            rospy.loginfo("planned next gate trajectory")
        return True

    def build_gate_derivatives(
        self,
        waypoints,
        forward_dir,
        is_next_gate=False,
        local_frame=False,
        approach_dir=None,
        local_axes=None,
    ):
        if self.command_mode != "trajectory":
            return None, None

        velocities = []
        accelerations = []
        for index, _point in enumerate(waypoints):
            if index > 0 or is_next_gate:
                speed = self.pass_speed_mps
            else:
                speed = 0.0

            velocity_dir = forward_dir
            use_approach_velocity = False
            if (
                index == 0
                and self.trajectory_start_velocity_to_gate_center
                and approach_dir is not None
            ):
                velocity_dir = approach_dir
                use_approach_velocity = True

            if local_frame:
                if self.trajectory_constrain_lateral_vertical_derivatives:
                    if use_approach_velocity and local_axes is not None:
                        world_velocity = (
                            velocity_dir[0] * speed,
                            velocity_dir[1] * speed,
                            0.0,
                        )
                        velocities.append(
                            (
                                dot(world_velocity, local_axes[0]),
                                dot(world_velocity, local_axes[1]),
                                0.0,
                            )
                        )
                    else:
                        velocities.append((speed, 0.0, 0.0))
                    accelerations.append((0.0, 0.0, 0.0))
                else:
                    if use_approach_velocity and local_axes is not None:
                        world_velocity = (
                            velocity_dir[0] * speed,
                            velocity_dir[1] * speed,
                            0.0,
                        )
                        velocities.append(
                            (
                                dot(world_velocity, local_axes[0]),
                                dot(world_velocity, local_axes[1]),
                                None,
                            )
                        )
                    else:
                        velocities.append((speed, None, None))
                    accelerations.append((0.0, None, None))
            else:
                velocities.append(
                    (
                        velocity_dir[0] * speed,
                        velocity_dir[1] * speed,
                        velocity_dir[2] * speed,
                    )
                )
                accelerations.append((0.0, 0.0, 0.0))
        return velocities, accelerations

    def build_path_frame_axes(self, forward_dir):
        forward_axis = self.horizontal_forward_dir(forward_dir)
        lateral_axis = (-forward_axis[1], forward_axis[0], 0.0)
        vertical_axis = (0.0, 0.0, 1.0)
        return forward_axis, lateral_axis, vertical_axis

    def log_min_snap_path_frame_diagnostics(self, waypoints, axes, trajectory, times):
        if trajectory is None or len(waypoints) < 2:
            return True

        origin = waypoints[0]
        waypoint_lateral = [world_to_frame(point, origin, axes)[1] for point in waypoints]
        min_allowed = min(waypoint_lateral)
        max_allowed = max(waypoint_lateral)
        min_sample = float("inf")
        max_sample = float("-inf")
        sample_count = 40
        for index in range(sample_count + 1):
            t = times[-1] * float(index) / float(sample_count)
            sample = trajectory.eval(t, 0)
            lateral = world_to_frame(sample, origin, axes)[1]
            min_sample = min(min_sample, lateral)
            max_sample = max(max_sample, lateral)

        overshoot = max(
            0.0,
            min_allowed - min_sample,
            max_sample - max_allowed,
        )
        if (
            self.trajectory_lateral_overshoot_limit_m > 0.0
            and overshoot > self.trajectory_lateral_overshoot_limit_m
        ):
            rospy.logerr(
                "min-snap lateral overshoot %.2fm exceeds path corridor: waypoint_l=[%.2f, %.2f] sample_l=[%.2f, %.2f]",
                overshoot,
                min_allowed,
                max_allowed,
                min_sample,
                max_sample,
            )
            return False
        else:
            rospy.loginfo(
                "min-snap path-frame lateral check: waypoint_l=[%.2f, %.2f] sample_l=[%.2f, %.2f] overshoot=%.2f",
                min_allowed,
                max_allowed,
                min_sample,
                max_sample,
                overshoot,
            )
            return True

    def set_state(self, state):
        if self.state == state:
            return
        self.state = state
        rospy.loginfo("state: %s", self.state)

    def compute_gate_waypoints(self, gate_rel, now=None, update_height=True):
        gate_world_raw = self.relative_to_world(gate_rel)
        if update_height:
            target_z = self.update_height_reference(gate_world_raw, gate_rel, now)
            target_z = self.apply_new_gate_direct_climb_height(
                target_z, gate_world_raw, gate_rel, now
            )
        elif self.command_mode == "trajectory":
            target_z = self.height_ref_world
        else:
            target_z = gate_world_raw[2]
        gate_world = self.apply_gate_target_z_offset(gate_world_raw, target_z)
        front_x = max(self.min_front_x_m, gate_rel[0] - self.front_offset_m)
        front_world = self.apply_gate_target_z_offset(
            self.relative_to_world((front_x, gate_rel[1], gate_rel[2])),
            target_z,
        )

        exit_world = None
        exit_offset = self.effective_exit_offset_m()
        if exit_offset > 0.0:
            exit_world = self.apply_gate_target_z_offset(
                self.relative_to_world(
                    (
                        gate_rel[0] + exit_offset,
                        gate_rel[1],
                        gate_rel[2],
                    )
                ),
                target_z,
            )
        return gate_world, front_world, exit_world

    def apply_new_gate_direct_climb_height(self, target_z, gate_world_raw, gate_rel, now):
        if self.command_mode != "trajectory":
            return target_z
        if not self.new_gate_direct_climb_height:
            return target_z
        if self.gate_sequence_index < self.height_full_range_after_gate_index:
            return target_z
        if self.state not in (
            CRUISE_SEARCH_NEXT_GATE,
            ADVANCE_SEARCH_NEXT_GATE,
            CONFIRM_NEXT_GATE,
        ):
            return target_z

        freeze_reason = self.height_gate_z_freeze_reason()
        if freeze_reason is not None:
            rospy.loginfo_throttle(
                0.5,
                "skip new gate direct climb height while PnP gate z is frozen: reason=%s rel_x=%.2f",
                freeze_reason,
                gate_rel[0],
            )
            return target_z

        direct_z = (
            self.filtered_gate_z_world
            if self.filtered_gate_z_world is not None
            else self.clamp_height(gate_world_raw[2] + self.height_gate_z_offset_m)
        )
        if direct_z >= target_z:
            return target_z

        applied_z = direct_z
        if self.new_gate_direct_climb_max_step_m > 0.0:
            applied_z = max(
                direct_z, target_z - self.new_gate_direct_climb_max_step_m
            )

        self.height_ref_world = applied_z
        self.height_ref_stamp = now if now is not None else rospy.Time.now()
        rospy.loginfo(
            "new gate direct climb height: limited_z=%.2f applied_z=%.2f direct_gate_z=%.2f raw_gate_z=%.2f rel_x=%.2f max_step=%.2f",
            target_z,
            applied_z,
            direct_z,
            gate_world_raw[2],
            gate_rel[0],
            self.new_gate_direct_climb_max_step_m,
        )
        return applied_z

    def apply_gate_target_z_offset(self, point, target_z=None):
        if self.command_mode == "trajectory":
            if target_z is None:
                target_z = self.height_ref_world
            return (point[0], point[1], target_z)
        if self.gate_target_z_offset_m == 0.0:
            return point
        return (point[0], point[1], point[2] - self.gate_target_z_offset_m)

    def clamp_height(self, z_value):
        active_min_z = self.active_height_min_z()
        lower = min(active_min_z, self.height_max_z)
        upper = max(active_min_z, self.height_max_z)
        return max(lower, min(upper, z_value))

    def active_height_min_z(self):
        gate_index = getattr(self, "gate_sequence_index", 0)
        if gate_index < self.height_full_range_after_gate_index:
            return max(self.height_min_z, self.first_gate_height_min_z)
        return self.height_min_z

    def search_next_altitude_target(self):
        if self.command_mode != "trajectory":
            return self.pos_world[2]

        candidates = []
        for z_value in (
            self.pos_world[2] if self.pos_world is not None else None,
            self.height_ref_world,
            self.search_next_altitude_world,
        ):
            if z_value is not None and math.isfinite(z_value):
                candidates.append(z_value)

        if not candidates:
            return self.clamp_height(self.target_flight_z)

        # AirSim/NED z is more negative at higher altitude. During lost-gate
        # advance search, never command a descent back to an older low gate.
        return self.clamp_height(min(candidates))

    def update_search_next_altitude_target(self, now=None):
        target_z = self.search_next_altitude_target()
        old_z = self.search_next_altitude_world
        if old_z is not None and target_z >= old_z - 1.0e-4:
            return old_z

        self.search_next_altitude_world = target_z
        if self.search_next_target_world is not None:
            self.search_next_target_world = (
                self.search_next_target_world[0],
                self.search_next_target_world[1],
                target_z,
            )
        if self.search_next_settle_target_world is not None:
            self.search_next_settle_target_world = (
                self.search_next_settle_target_world[0],
                self.search_next_settle_target_world[1],
                target_z,
            )

        if old_z is not None:
            rospy.loginfo_throttle(
                0.5,
                "raise search next altitude target: old_z=%.2f new_z=%.2f current_z=%.2f height_ref_z=%.2f",
                old_z,
                target_z,
                self.pos_world[2] if self.pos_world is not None else float("nan"),
                self.height_ref_world
                if self.height_ref_world is not None
                else float("nan"),
            )
        return target_z

    def active_height_ref_step_limit_m(self):
        gate_index = getattr(self, "gate_sequence_index", 0)
        if gate_index < self.height_full_range_after_gate_index:
            return self.first_gate_height_ref_max_step_m
        return self.height_ref_max_step_m

    def active_height_ref_rate_mps(self):
        gate_index = getattr(self, "gate_sequence_index", 0)
        if gate_index < self.height_full_range_after_gate_index:
            return self.first_gate_height_ref_rate_mps
        return self.height_ref_rate_mps

    def height_gate_z_freeze_reason(self):
        if self.height_gate_z_freeze_on_recovery:
            if self.heading_recovery_active:
                return "heading recovery"
            if self.enable_heading_recovery:
                needed, _expected_yaw, _yaw_error, _roll_deg, _pitch_deg = (
                    self.heading_recovery_status()
                )
                if needed:
                    return "heading recovery needed"

        if self.height_gate_z_freeze_on_large_attitude:
            if not self.is_attitude_valid_for_vision():
                return "large attitude"

        if self.height_gate_z_freeze_on_yaw_mismatch:
            heading_valid, _expected_yaw, yaw_error = self.is_heading_valid_for_vision()
            if not heading_valid:
                return "yaw mismatch %.1f > %.1f" % (
                    abs(math.degrees(yaw_error)),
                    self.max_vision_yaw_error_deg,
                )

        return None

    def filtered_gate_height_z(self, measured_gate_z, now, commit=True):
        freeze_reason = self.height_gate_z_freeze_reason()
        if freeze_reason is not None:
            rospy.loginfo_throttle(
                0.5,
                "freeze PnP gate z height update: reason=%s measured_z=%.2f filtered_z=%s ref_z=%.2f",
                freeze_reason,
                measured_gate_z,
                "none"
                if self.filtered_gate_z_world is None
                else "%.2f" % self.filtered_gate_z_world,
                self.height_ref_world,
            )
            return self.height_ref_world, "freeze: " + freeze_reason

        base_z = (
            self.filtered_gate_z_world
            if self.filtered_gate_z_world is not None
            else self.height_ref_world
        )
        limited_z = measured_gate_z
        status = "ok"

        max_step = max(0.0, self.height_gate_z_max_step_m)
        if max_step > 0.0:
            step_limited_z = clamp(limited_z, base_z - max_step, base_z + max_step)
            if abs(step_limited_z - limited_z) > 1.0e-6:
                status = "step limited"
            limited_z = step_limited_z

        max_rate = max(0.0, self.height_gate_z_rate_mps)
        if (
            max_rate > 0.0
            and self.filtered_gate_z_world is not None
            and self.filtered_gate_z_stamp is not None
        ):
            dt = max(0.0, (now - self.filtered_gate_z_stamp).to_sec())
            rate_step = max_rate * dt
            rate_limited_z = clamp(
                limited_z,
                self.filtered_gate_z_world - rate_step,
                self.filtered_gate_z_world + rate_step,
            )
            if abs(rate_limited_z - limited_z) > 1.0e-6:
                status = "rate limited" if status == "ok" else status + " + rate"
            limited_z = rate_limited_z

        alpha = max(0.0, min(1.0, self.height_gate_z_filter_alpha))
        if self.filtered_gate_z_world is None:
            filtered_z = limited_z
        else:
            filtered_z = (
                alpha * limited_z + (1.0 - alpha) * self.filtered_gate_z_world
            )
        filtered_z = self.clamp_height(filtered_z)

        if commit:
            self.filtered_gate_z_world = filtered_z
            self.filtered_gate_z_stamp = now

        return filtered_z, status

    def effective_exit_offset_m(self):
        if self.gate_trigger_exit_offset_m >= 0.0:
            return self.gate_trigger_exit_offset_m
        if self.gate_trigger_length_m <= 0.0:
            return self.exit_offset_m
        trigger_exit = 0.5 * self.gate_trigger_length_m + self.exit_offset_m
        return max(self.exit_offset_m, trigger_exit)

    def effective_clear_distance_m(self):
        if self.gate_trigger_length_m <= 0.0:
            return self.clear_distance_m
        trigger_clear = 0.5 * self.gate_trigger_length_m + self.gate_trigger_clear_margin_m
        return max(self.clear_distance_m, trigger_clear)

    def effective_cleared_gate_ignore_radius_m(self):
        if self.gate_trigger_length_m <= 0.0:
            return self.cleared_gate_ignore_radius_m
        return max(self.cleared_gate_ignore_radius_m, 0.5 * self.gate_trigger_length_m)

    def fresh_visual_error(self, now):
        if self.last_visual_error is None or self.last_visual_error_stamp is None:
            return None
        if (now - self.last_visual_error_stamp).to_sec() > self.height_visual_timeout:
            return None
        return self.last_visual_error

    def fresh_search_visual_error(self, now):
        if not self.enable_search:
            return None
        if not self.search_visual_yaw_enabled:
            return None
        if self.state not in (
            CRUISE_SEARCH_NEXT_GATE,
            ADVANCE_SEARCH_NEXT_GATE,
            CONFIRM_NEXT_GATE,
        ):
            return None
        visual_error = self.fresh_visual_error(now)
        if visual_error is None:
            return None
        if visual_error[2] < self.search_visual_min_area_ratio:
            return None
        return visual_error

    def is_new_gate_detection_fresh(self, now):
        visual_error = self.fresh_visual_error(now)
        if self.new_gate_requires_visual_error and visual_error is None:
            rospy.loginfo_throttle(
                0.5,
                "ignore new gate candidate without fresh visual_error",
            )
            return False
        if visual_error is None:
            return True
        if visual_error[2] < self.search_visual_min_area_ratio:
            rospy.loginfo_throttle(
                0.5,
                "ignore new gate candidate with small visual area: area=%.3f < %.3f",
                visual_error[2],
                self.search_visual_min_area_ratio,
            )
            return False
        eps = max(0.0, self.new_gate_zero_visual_epsilon)
        if (
            self.new_gate_reject_zero_visual_error
            and eps > 0.0
            and abs(visual_error[0]) <= eps
            and abs(visual_error[1]) <= eps
        ):
            rospy.logwarn_throttle(
                0.5,
                "ignore new gate candidate that looks like hold-last visual_error: visual=(%.5f, %.5f area=%.3f)",
                visual_error[0],
                visual_error[1],
                visual_error[2],
            )
            return False
        return True

    def compute_search_yaw_rate(self, now, default_yaw_rate_deg):
        visual_error = self.fresh_search_visual_error(now)
        if visual_error is None:
            return default_yaw_rate_deg

        visual_x = visual_error[0]
        if abs(visual_x) < self.search_visual_yaw_deadband:
            return 0.0

        yaw_rate = (
            self.search_visual_yaw_sign
            * self.search_visual_yaw_gain_deg
            * visual_x
        )
        yaw_rate = clamp(yaw_rate, self.search_visual_yaw_max_rate_deg)
        rospy.loginfo_throttle(
            0.5,
            "search visual yaw: visual_x=%.2f area=%.3f yaw_rate=%.1f",
            visual_x,
            visual_error[2],
            yaw_rate,
        )
        return yaw_rate

    def steer_search_forward_dir_from_visual(self, now, forward_dir):
        visual_error = self.fresh_search_visual_error(now)
        if visual_error is None or self.search_visual_path_turn_rate_deg <= 0.0:
            self.last_search_steer_stamp = now
            return forward_dir

        visual_x = visual_error[0]
        if abs(visual_x) < self.search_visual_yaw_deadband:
            self.last_search_steer_stamp = now
            return forward_dir

        if self.last_search_steer_stamp is None:
            dt = 1.0 / max(1.0, self.control_rate)
        else:
            dt = max(0.0, min(0.2, (now - self.last_search_steer_stamp).to_sec()))
        self.last_search_steer_stamp = now

        turn_rate = (
            self.search_visual_yaw_sign
            * self.search_visual_path_turn_rate_deg
            * visual_x
        )
        turn_rate = clamp(turn_rate, self.search_visual_path_turn_rate_deg)
        if abs(turn_rate) < 1.0e-3 or dt <= 0.0:
            return forward_dir

        current_yaw = math.atan2(forward_dir[1], forward_dir[0])
        new_yaw = current_yaw + math.radians(turn_rate) * dt
        new_forward_dir = (math.cos(new_yaw), math.sin(new_yaw), 0.0)
        self.search_forward_dir_world = new_forward_dir
        self.search_start_world = self.pos_world
        self.search_next_target_world = None
        self.search_next_target_stamp = None
        rospy.loginfo_throttle(
            0.5,
            "search visual steering: visual_x=%.2f area=%.3f turn_rate=%.1f forward_dir=(%.2f, %.2f)",
            visual_x,
            visual_error[2],
            turn_rate,
            new_forward_dir[0],
            new_forward_dir[1],
        )
        return new_forward_dir

    def update_height_reference(self, gate_world_raw, gate_rel=None, now=None, commit=True):
        if self.command_mode != "trajectory":
            return gate_world_raw[2]
        if self.is_takeoff_pending():
            return self.height_ref_world
        if not self.height_manager_enabled:
            target = self.clamp_height(self.target_flight_z)
            if commit:
                self.height_ref_world = target
            return target
        if now is None:
            now = rospy.Time.now()

        current_ref = self.height_ref_world
        unclamped_gate_z = gate_world_raw[2] + self.height_gate_z_offset_m
        measured_gate_z = self.clamp_height(unclamped_gate_z)
        filtered_gate_z, gate_z_status = self.filtered_gate_height_z(
            measured_gate_z, now, commit=commit
        )

        visual_ref = current_ref
        visual_error = self.fresh_visual_error(now)
        visual_y = None
        visual_correction = 0.0
        if self.height_reference_mode == "gate_center":
            candidate = filtered_gate_z
        elif visual_error is not None:
            visual_y = visual_error[1]
            if abs(visual_y) > self.height_visual_deadband:
                distance_scale = 0.0
                if gate_rel is not None:
                    distance_scale = max(0.0, gate_rel[0]) * self.height_visual_distance_gain
                visual_correction = (
                    self.height_visual_error_sign
                    * visual_y
                    * (self.height_visual_error_gain_m + distance_scale)
                )
                # Positive correction means larger NED z, i.e. commanding lower altitude.
                # Downward corrections are deliberately softer because bbox/edge detections
                # near a gate often jump low in the image for one or two frames.
                if visual_correction > 0.0:
                    visual_correction *= max(
                        0.0, min(1.0, self.height_down_correction_scale)
                    )
                visual_ref = current_ref + visual_correction
            gate_weight = max(0.0, min(1.0, self.height_gate_z_weight))
            candidate = (1.0 - gate_weight) * visual_ref + gate_weight * filtered_gate_z
        else:
            gate_weight = max(0.0, min(1.0, self.height_gate_z_weight))
            candidate = (1.0 - gate_weight) * visual_ref + gate_weight * filtered_gate_z
        candidate = self.clamp_height(candidate)

        alpha = max(0.0, min(1.0, self.height_ref_alpha))
        target = (1.0 - alpha) * current_ref + alpha * candidate
        active_step_limit = self.active_height_ref_step_limit_m()
        active_rate_limit = self.active_height_ref_rate_mps()
        max_step = max(0.0, active_step_limit)
        if self.height_ref_stamp is not None and active_rate_limit > 0.0:
            dt = max(0.0, (now - self.height_ref_stamp).to_sec())
            rate_step = active_rate_limit * dt
            max_step = min(max_step, rate_step) if max_step > 0.0 else rate_step
        if max_step > 0.0:
            target = clamp(target, current_ref - max_step, current_ref + max_step)

        target = self.clamp_height(target)
        if commit:
            self.height_ref_world = target
            self.height_ref_stamp = now
            rospy.loginfo_throttle(
                0.5,
                "height ref manager: mode=%s ref_z=%.2f raw_gate_z=%.2f gate_z=%.2f filtered_gate_z=%.2f gate_z_status=%s rel_x=%s visual_y=%s visual_dz=%.2f candidate=%.2f",
                self.height_reference_mode,
                self.height_ref_world,
                gate_world_raw[2],
                measured_gate_z,
                filtered_gate_z,
                gate_z_status,
                "none" if gate_rel is None else "%.2f" % gate_rel[0],
                "none" if visual_y is None else "%.2f" % visual_y,
                visual_correction,
                candidate,
            )
            if abs(measured_gate_z - unclamped_gate_z) > 1.0e-3:
                rospy.logwarn_throttle(
                    0.5,
                    "height ref clamped gate center: raw_z=%.2f clamped_z=%.2f limits=[%.2f, %.2f]",
                    unclamped_gate_z,
                    measured_gate_z,
                    self.active_height_min_z(),
                    self.height_max_z,
                )
        return target

    def compute_gate_forward_dir(self, front_world, gate_world, exit_world):
        fallback = (math.cos(self.yaw), math.sin(self.yaw), 0.0)
        if exit_world is not None:
            return normalize_vector(
                (
                    exit_world[0] - gate_world[0],
                    exit_world[1] - gate_world[1],
                    exit_world[2] - gate_world[2],
                ),
                fallback,
            )
        return normalize_vector(
            (
                gate_world[0] - front_world[0],
                gate_world[1] - front_world[1],
                gate_world[2] - front_world[2],
            ),
            fallback,
        )

    def compute_gate_center_approach_dir(self, start_world, gate_world, fallback_dir):
        return normalize_vector(
            (
                gate_world[0] - start_world[0],
                gate_world[1] - start_world[1],
                0.0,
            ),
            self.horizontal_forward_dir(fallback_dir),
        )

    def horizontal_forward_dir(self, forward_dir=None):
        fallback = normalize_vector(
            (math.cos(self.yaw), math.sin(self.yaw), 0.0),
            (1.0, 0.0, 0.0),
        )
        if forward_dir is None:
            return fallback
        return normalize_vector((forward_dir[0], forward_dir[1], 0.0), fallback)

    def active_gate_bearing_yaw(self):
        aim_gate = self.current_aim_gate_world()
        if aim_gate is None:
            return None, None
        dx = aim_gate[0] - self.pos_world[0]
        dy = aim_gate[1] - self.pos_world[1]
        horizontal_dist = math.sqrt(dx * dx + dy * dy)
        if horizontal_dist < 1.0e-3:
            return None, horizontal_dist
        return math.atan2(dy, dx), horizontal_dist

    def gate_center_yaw_weight(self, horizontal_dist):
        near_m = max(0.0, self.gate_center_yaw_near_m)
        far_m = max(near_m + 1.0e-3, self.gate_center_yaw_far_m)
        weight = (horizontal_dist - near_m) / (far_m - near_m)
        weight = max(0.0, min(1.0, weight))
        return weight * weight * (3.0 - 2.0 * weight)

    def tracking_target_yaw(self):
        forward_dir = self.horizontal_forward_dir(self.current_planning_forward_dir())
        path_yaw = math.atan2(forward_dir[1], forward_dir[0])
        if (
            not self.gate_center_yaw_enabled
            or self.state not in (TRACK_GATE, TRACK_NEXT_GATE)
        ):
            return path_yaw, path_yaw, path_yaw, "path", 0.0

        gate_center_yaw, horizontal_dist = self.active_gate_bearing_yaw()
        if gate_center_yaw is None or horizontal_dist is None:
            return path_yaw, path_yaw, path_yaw, "path_no_gate", 0.0

        center_weight = self.gate_center_yaw_weight(horizontal_dist)
        target_yaw = angle_lerp(path_yaw, gate_center_yaw, center_weight)
        yaw_source = "next_gate_blend" if self.is_near_gate_zone() else "gate_center_blend"
        return (
            target_yaw,
            gate_center_yaw,
            path_yaw,
            yaw_source,
            center_weight,
        )

    def is_attitude_valid_for_vision(self):
        if self.max_vision_tilt_deg <= 0.0:
            return True
        return (
            abs(math.degrees(self.roll)) <= self.max_vision_tilt_deg
            and abs(math.degrees(self.pitch)) <= self.max_vision_tilt_deg
        )

    def expected_vision_yaw(self):
        if self.command_mode == "trajectory":
            forward_dir = self.current_planning_forward_dir()
            return math.atan2(forward_dir[1], forward_dir[0])
        return self.yaw

    def heading_recovery_expected_yaw(self):
        path_dir = self.current_planning_forward_dir()
        path_yaw = math.atan2(path_dir[1], path_dir[0])
        if self.heading_recovery_target_mode != "gate_center":
            return path_yaw
        gate_yaw, horizontal_dist = self.active_gate_bearing_yaw()
        if gate_yaw is None or horizontal_dist is None:
            return path_yaw
        return gate_yaw

    def is_heading_valid_for_vision(self):
        expected_yaw = self.expected_vision_yaw()
        yaw_error = wrap_angle(self.yaw - expected_yaw)
        if self.max_vision_yaw_error_deg <= 0.0:
            return True, expected_yaw, yaw_error
        return (
            abs(math.degrees(yaw_error)) <= self.max_vision_yaw_error_deg,
            expected_yaw,
            yaw_error,
        )

    def heading_recovery_status(self):
        expected_yaw = self.heading_recovery_expected_yaw()
        yaw_error = wrap_angle(self.yaw - expected_yaw)
        roll_deg = math.degrees(self.roll)
        pitch_deg = math.degrees(self.pitch)
        yaw_error_deg = math.degrees(yaw_error)
        yaw_threshold = (
            self.heading_recovery_exit_yaw_error_deg
            if self.heading_recovery_active
            else self.heading_recovery_yaw_error_deg
        )
        tilt_threshold = (
            self.heading_recovery_exit_tilt_deg
            if self.heading_recovery_active
            else self.heading_recovery_tilt_deg
        )
        tilt_bad = (
            tilt_threshold > 0.0
            and (
                abs(roll_deg) > tilt_threshold
                or abs(pitch_deg) > tilt_threshold
            )
        )
        yaw_bad = (
            yaw_threshold > 0.0
            and abs(yaw_error_deg) > yaw_threshold
        )
        return tilt_bad or yaw_bad, expected_yaw, yaw_error, roll_deg, pitch_deg

    def should_publish_heading_recovery(self, now):
        _ = now
        if not self.enable_heading_recovery:
            return False
        if self.command_mode != "trajectory":
            return False
        if self.is_takeoff_pending() or self.state == TAKEOFF:
            return False
        needed, _expected_yaw, _yaw_error, _roll_deg, _pitch_deg = (
            self.heading_recovery_status()
        )
        if not needed:
            if self.heading_recovery_active:
                self.replan_after_heading_recovery_if_needed(now)
            self.reset_heading_recovery()
        return needed

    def reset_heading_recovery(self):
        self.heading_recovery_active = False
        self.heading_recovery_last_stamp = None
        self.heading_recovery_entry_z = None
        self.heading_recovery_altitude_z = None
        self.heading_recovery_yaw_level_since = None

    def pause_active_plan_time_for_recovery(self, now):
        if not self.heading_recovery_pause_plan_time:
            return 0.0
        if self.heading_recovery_last_stamp is None:
            return 0.0
        dt = max(0.0, (now - self.heading_recovery_last_stamp).to_sec())
        if dt <= 0.0:
            return 0.0
        with self.plan_lock:
            if self.plan_start is not None:
                self.plan_start += rospy.Duration(dt)
        return dt

    def heading_recovery_altitude_target(self, raw_planned_z=None):
        if self.heading_recovery_entry_z is None:
            self.heading_recovery_entry_z = self.pos_world[2]

        candidates = []
        for z_value in (
            self.pos_world[2] if self.pos_world is not None else None,
            self.heading_recovery_entry_z,
            raw_planned_z,
            self.height_ref_world,
            self.search_next_altitude_world,
            self.target_gate_world[2] if self.target_gate_world is not None else None,
        ):
            if z_value is not None and math.isfinite(z_value):
                candidates.append(self.clamp_height(z_value))

        if candidates:
            # AirSim/NED z is more negative at higher altitude. Recovery must
            # not lock to the low entry altitude when the planned route climbs.
            target_z = self.clamp_height(min(candidates))
        else:
            target_z = self.clamp_height(self.heading_recovery_entry_z)

        self.heading_recovery_altitude_z = target_z

        max_vz = max(0.0, self.trajectory_max_vertical_speed_mps)
        if max_vz > 0.0 and self.pos_world is not None:
            target_vz = clamp(target_z - self.pos_world[2], max_vz)
        else:
            target_vz = 0.0
        return target_z, target_vz, "planned-follow"

    def heading_recovery_yaw_ready(self, now, abs_tilt_deg):
        enable_tilt = self.heading_recovery_yaw_enable_tilt_deg
        if enable_tilt <= 0.0:
            return True, self.heading_recovery_max_yaw_rate_deg, 1.0

        if abs_tilt_deg > enable_tilt:
            partial_tilt = self.heading_recovery_partial_yaw_tilt_deg
            partial_rate = self.heading_recovery_partial_yaw_rate_deg
            if (
                partial_tilt > enable_tilt
                and partial_rate > 0.0
                and abs_tilt_deg <= partial_tilt
            ):
                self.heading_recovery_yaw_level_since = None
                yaw_rate_limit = min(
                    self.heading_recovery_max_yaw_rate_deg,
                    partial_rate,
                )
                yaw_scale = yaw_rate_limit / max(
                    1.0e-6, self.heading_recovery_max_yaw_rate_deg
                )
                return yaw_rate_limit > 1.0e-3, yaw_rate_limit, yaw_scale
            self.heading_recovery_yaw_level_since = None
            return False, 0.0, 0.0

        if self.heading_recovery_yaw_level_since is None:
            self.heading_recovery_yaw_level_since = now
        level_time = (now - self.heading_recovery_yaw_level_since).to_sec()
        if level_time < self.heading_recovery_yaw_enable_dwell_s:
            return False, 0.0, 0.0

        full_rate_tilt = max(0.0, self.heading_recovery_full_yaw_rate_tilt_deg)
        if enable_tilt <= full_rate_tilt:
            yaw_scale = 1.0
        else:
            yaw_scale = (enable_tilt - abs_tilt_deg) / (enable_tilt - full_rate_tilt)
            yaw_scale = max(0.0, min(1.0, yaw_scale))
        yaw_rate_limit = self.heading_recovery_max_yaw_rate_deg * yaw_scale
        return yaw_rate_limit > 1.0e-3, yaw_rate_limit, yaw_scale

    def heading_recovery_target_direction(self):
        path_dir = self.current_planning_forward_dir()
        path_yaw = math.atan2(path_dir[1], path_dir[0])
        gate_yaw, horizontal_dist = self.active_gate_bearing_yaw()
        if (
            self.heading_recovery_target_mode != "gate_center"
            or gate_yaw is None
            or horizontal_dist is None
            or horizontal_dist < 1.0e-3
        ):
            return path_dir, "path", None, path_yaw

        gate_dir = (math.cos(gate_yaw), math.sin(gate_yaw), 0.0)
        target_source = "next_gate" if self.is_near_gate_zone() else "gate_center"
        weight = max(0.0, min(1.0, self.heading_recovery_gate_center_weight))
        target_dir = normalize_vector(
            (
                (1.0 - weight) * path_dir[0] + weight * gate_dir[0],
                (1.0 - weight) * path_dir[1] + weight * gate_dir[1],
                0.0,
            ),
            path_dir,
        )

        min_forward = max(0.0, min(1.0, self.heading_recovery_min_forward_component))
        max_target_angle_deg = self.heading_recovery_max_target_angle_deg
        if 0.0 < max_target_angle_deg < 90.0:
            min_forward = max(
                min_forward,
                math.cos(math.radians(max_target_angle_deg)),
            )
        forward_component = dot(target_dir, path_dir)
        if min_forward > 0.0 and forward_component < min_forward:
            lateral = (
                gate_dir[0] - dot(gate_dir, path_dir) * path_dir[0],
                gate_dir[1] - dot(gate_dir, path_dir) * path_dir[1],
                0.0,
            )
            lateral_norm = norm(lateral)
            if lateral_norm > 1.0e-6:
                lateral_axis = (
                    lateral[0] / lateral_norm,
                    lateral[1] / lateral_norm,
                    0.0,
                )
                lateral_component = math.sqrt(max(0.0, 1.0 - min_forward * min_forward))
                target_dir = normalize_vector(
                    (
                        min_forward * path_dir[0] + lateral_component * lateral_axis[0],
                        min_forward * path_dir[1] + lateral_component * lateral_axis[1],
                        0.0,
                    ),
                    path_dir,
                )
            else:
                target_dir = path_dir

        return target_dir, target_source, gate_yaw, path_yaw

    def current_planning_forward_dir(self):
        if (
            self.state in (TRACK_GATE, TRACK_NEXT_GATE, CLEAR_GATE)
            and self.target_forward_dir_world is not None
        ):
            return self.horizontal_forward_dir(self.target_forward_dir_world)
        if (
            self.state
            in (CRUISE_SEARCH_NEXT_GATE, ADVANCE_SEARCH_NEXT_GATE, CONFIRM_NEXT_GATE)
            and self.search_forward_dir_world is not None
        ):
            return self.horizontal_forward_dir(self.search_forward_dir_world)
        return self.horizontal_forward_dir()

    def is_gate_plan_candidate_valid(
        self,
        front_world,
        gate_world,
        exit_world,
        current_forward_dir,
        context="gate candidate",
    ):
        current_forward_dir = self.horizontal_forward_dir(current_forward_dir)
        candidate_delta = (
            gate_world[0] - self.pos_world[0],
            gate_world[1] - self.pos_world[1],
            gate_world[2] - self.pos_world[2],
        )
        forward_progress = dot(candidate_delta, current_forward_dir)
        if forward_progress < self.min_new_gate_world_forward_m:
            if context == "replan":
                rospy.logwarn(
                    "reject replan behind current position forward_progress=%.2f min=%.2f gate=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
                    forward_progress,
                    self.min_new_gate_world_forward_m,
                    gate_world[0],
                    gate_world[1],
                    gate_world[2],
                    self.pos_world[0],
                    self.pos_world[1],
                    self.pos_world[2],
                )
            else:
                rospy.logwarn_throttle(
                    1.0,
                    "reject gate candidate: behind current position forward_progress=%.2f min=%.2f gate=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
                    forward_progress,
                    self.min_new_gate_world_forward_m,
                    gate_world[0],
                    gate_world[1],
                    gate_world[2],
                    self.pos_world[0],
                    self.pos_world[1],
                    self.pos_world[2],
                )
            return False

        new_forward_dir = self.horizontal_forward_dir(
            self.compute_gate_forward_dir(front_world, gate_world, exit_world)
        )
        direction_dot = dot(new_forward_dir, current_forward_dir)
        if direction_dot < 0.7:
            if context == "replan":
                rospy.logwarn(
                    "reject replan reverse direction: dot=%.2f current_dir=(%.2f, %.2f) new_dir=(%.2f, %.2f)",
                    direction_dot,
                    current_forward_dir[0],
                    current_forward_dir[1],
                    new_forward_dir[0],
                    new_forward_dir[1],
                )
            else:
                rospy.logwarn_throttle(
                    1.0,
                    "reject gate candidate: reverse direction dot=%.2f current_dir=(%.2f, %.2f) new_dir=(%.2f, %.2f)",
                    direction_dot,
                    current_forward_dir[0],
                    current_forward_dir[1],
                    new_forward_dir[0],
                    new_forward_dir[1],
                )
            return False

        return True

    def is_in_cleared_gate_ignore_zone(self, gate_world):
        ignore_radius = self.effective_cleared_gate_ignore_radius_m()
        if ignore_radius <= 0.0:
            return False
        for cleared_world in self.cleared_gate_world_positions:
            if distance(gate_world, cleared_world) <= ignore_radius:
                return True
        for missed_world in self.missed_gate_world_positions:
            if distance(gate_world, missed_world) <= ignore_radius:
                return True
        return False

    def is_search_next_candidate_allowed(self, gate_world):
        if self.last_cleared_gate_world is None or self.search_forward_dir_world is None:
            return True

        delta = (
            gate_world[0] - self.last_cleared_gate_world[0],
            gate_world[1] - self.last_cleared_gate_world[1],
            gate_world[2] - self.last_cleared_gate_world[2],
        )
        forward_distance = dot(delta, self.search_forward_dir_world)
        lateral = (
            delta[0] - forward_distance * self.search_forward_dir_world[0],
            delta[1] - forward_distance * self.search_forward_dir_world[1],
            0.0,
        )
        lateral_distance = norm(lateral)

        if forward_distance < self.min_new_gate_world_forward_m:
            rospy.loginfo_throttle(
                1.0,
                "ignore next gate candidate behind cleared gate: forward=%.2f < %.2f world=(%.2f, %.2f, %.2f)",
                forward_distance,
                self.min_new_gate_world_forward_m,
                gate_world[0],
                gate_world[1],
                gate_world[2],
            )
            return False

        if (
            self.search_gate_corridor_half_width_m > 0.0
            and lateral_distance > self.search_gate_corridor_half_width_m
        ):
            rospy.loginfo_throttle(
                1.0,
                "ignore next gate candidate outside search corridor: lateral=%.2f > %.2f world=(%.2f, %.2f, %.2f)",
                lateral_distance,
                self.search_gate_corridor_half_width_m,
                gate_world[0],
                gate_world[1],
                gate_world[2],
            )
            return False

        return True

    def is_prefetch_next_candidate_allowed(self, gate_world):
        active_gate = self.active_gate_world or self.target_gate_world
        if active_gate is None:
            return False
        forward_dir = self.current_planning_forward_dir()
        axes = self.build_path_frame_axes(forward_dir)
        rel_current = world_to_frame(gate_world, self.pos_world, axes)
        rel_active = world_to_frame(active_gate, self.pos_world, axes)

        if rel_current[0] < self.near_gate_next_min_distance_m:
            rospy.loginfo_throttle(
                0.5,
                "ignore prefetch next gate too near: forward=%.2f < %.2f world=(%.2f, %.2f, %.2f)",
                rel_current[0],
                self.near_gate_next_min_distance_m,
                gate_world[0],
                gate_world[1],
                gate_world[2],
            )
            return False

        min_ahead = max(0.0, self.min_new_gate_world_forward_m)
        if rel_current[0] < rel_active[0] + min_ahead:
            rospy.loginfo_throttle(
                0.5,
                "ignore prefetch candidate not ahead of active gate: candidate_forward=%.2f active_forward=%.2f min_ahead=%.2f",
                rel_current[0],
                rel_active[0],
                min_ahead,
            )
            return False

        if abs(rel_current[1]) > self.search_gate_corridor_half_width_m:
            rospy.loginfo_throttle(
                0.5,
                "ignore prefetch candidate outside corridor: lateral=%.2f > %.2f world=(%.2f, %.2f, %.2f)",
                rel_current[1],
                self.search_gate_corridor_half_width_m,
                gate_world[0],
                gate_world[1],
                gate_world[2],
            )
            return False

        if distance(gate_world, active_gate) <= self.effective_cleared_gate_ignore_radius_m():
            rospy.loginfo_throttle(
                0.5,
                "ignore prefetch candidate near active gate: dist=%.2f world=(%.2f, %.2f, %.2f)",
                distance(gate_world, active_gate),
                gate_world[0],
                gate_world[1],
                gate_world[2],
            )
            return False

        return True

    def try_prefetch_next_gate(self, gate_rel, now):
        metrics = self.current_gate_clear_metrics()
        if self.is_near_gate_zone(metrics) and self.near_gate_forbid_current_gate_yaw:
            # In the near-gate zone the current gate is only a clear/miss object.
            # Until a next gate is stable, yaw/recovery falls back to nominal forward.
            if self.next_gate_candidate_world is None:
                self.aim_gate_world = None
        if not self.should_prefetch_next_gate(metrics):
            return False
        if not self.is_new_gate_detection_fresh(now):
            return False

        candidate_filter_world = self.apply_gate_target_z_offset(
            self.relative_to_world(gate_rel),
            self.height_ref_world if self.command_mode == "trajectory" else None,
        )
        if self.is_in_cleared_gate_ignore_zone(candidate_filter_world):
            return False
        if not self.is_prefetch_next_candidate_allowed(candidate_filter_world):
            return False

        gate_world, front_world, exit_world = self.compute_gate_waypoints(
            gate_rel, now, update_height=True
        )
        if not self.is_gate_plan_candidate_valid(
            front_world,
            gate_world,
            exit_world,
            self.current_planning_forward_dir(),
            context="prefetch next gate",
        ):
            return False

        stable = self.update_next_gate_candidate(gate_world, now)
        if stable:
            rospy.loginfo_throttle(
                0.5,
                "near gate aim switched to prefetched next gate: aim_gate=(%.2f, %.2f, %.2f)",
                self.aim_gate_world[0],
                self.aim_gate_world[1],
                self.aim_gate_world[2],
            )
        return stable

    def reset_new_gate_candidate(self):
        self.new_gate_candidate_world = None
        self.new_gate_candidate_count = 0
        self.new_gate_candidate_stamp = None

    def reset_next_gate_candidate(self):
        self.next_gate_candidate_world = None
        self.next_gate_candidate_count = 0
        self.next_gate_candidate_stamp = None
        if self.is_near_gate_zone():
            self.aim_gate_world = None

    def update_new_gate_candidate(self, gate_world, now):
        if self.new_gate_candidate_stamp is not None:
            age = (now - self.new_gate_candidate_stamp).to_sec()
            if self.gate_reacquire_timeout > 0.0 and age > self.gate_reacquire_timeout:
                self.reset_new_gate_candidate()

        if self.new_gate_candidate_world is None:
            self.new_gate_candidate_world = gate_world
            self.new_gate_candidate_count = 1
        elif distance(gate_world, self.new_gate_candidate_world) <= self.new_gate_candidate_tolerance_m:
            self.new_gate_candidate_count += 1
            alpha = 1.0 / float(self.new_gate_candidate_count)
            self.new_gate_candidate_world = (
                (1.0 - alpha) * self.new_gate_candidate_world[0] + alpha * gate_world[0],
                (1.0 - alpha) * self.new_gate_candidate_world[1] + alpha * gate_world[1],
                (1.0 - alpha) * self.new_gate_candidate_world[2] + alpha * gate_world[2],
            )
        else:
            self.new_gate_candidate_world = gate_world
            self.new_gate_candidate_count = 1

        self.new_gate_candidate_stamp = now
        rospy.loginfo_throttle(
            0.5,
            "new gate candidate stable %d/%d world=(%.2f, %.2f, %.2f)",
            self.new_gate_candidate_count,
            self.new_gate_stable_frames,
            self.new_gate_candidate_world[0],
            self.new_gate_candidate_world[1],
            self.new_gate_candidate_world[2],
        )
        return self.new_gate_candidate_count >= self.new_gate_stable_frames

    def update_next_gate_candidate(self, gate_world, now):
        if self.next_gate_candidate_stamp is not None:
            age = (now - self.next_gate_candidate_stamp).to_sec()
            if self.gate_reacquire_timeout > 0.0 and age > self.gate_reacquire_timeout:
                self.reset_next_gate_candidate()

        if self.next_gate_candidate_world is None:
            self.next_gate_candidate_world = gate_world
            self.next_gate_candidate_count = 1
        elif distance(gate_world, self.next_gate_candidate_world) <= self.new_gate_candidate_tolerance_m:
            self.next_gate_candidate_count += 1
            alpha = 1.0 / float(self.next_gate_candidate_count)
            self.next_gate_candidate_world = (
                (1.0 - alpha) * self.next_gate_candidate_world[0] + alpha * gate_world[0],
                (1.0 - alpha) * self.next_gate_candidate_world[1] + alpha * gate_world[1],
                (1.0 - alpha) * self.next_gate_candidate_world[2] + alpha * gate_world[2],
            )
        else:
            self.next_gate_candidate_world = gate_world
            self.next_gate_candidate_count = 1

        self.next_gate_candidate_stamp = now
        stable = self.next_gate_candidate_count >= self.new_gate_stable_frames
        if stable:
            self.aim_gate_world = self.next_gate_candidate_world
        rospy.loginfo_throttle(
            0.5,
            "prefetch next gate candidate stable %d/%d world=(%.2f, %.2f, %.2f) aim_ready=%s",
            self.next_gate_candidate_count,
            self.new_gate_stable_frames,
            self.next_gate_candidate_world[0],
            self.next_gate_candidate_world[1],
            self.next_gate_candidate_world[2],
            str(stable),
        )
        return stable

    def update_search_altitude_preview(self, gate_world, gate_rel):
        if self.command_mode != "trajectory":
            return
        new_z = self.clamp_height(gate_world[2])
        old_z = self.search_next_altitude_world
        if old_z is None:
            should_update = True
        else:
            z_delta = new_z - old_z
            should_update = z_delta < -self.height_replan_z_threshold_m
            if z_delta > self.height_replan_z_threshold_m:
                should_update = self.allow_height_down_replan_while_tracking

        if not should_update:
            return

        self.search_next_altitude_world = new_z
        if self.search_next_target_world is not None:
            self.search_next_target_world = (
                self.search_next_target_world[0],
                self.search_next_target_world[1],
                new_z,
            )
        rospy.loginfo(
            "search next height preview: old_z=%s new_z=%.2f rel_x=%.2f",
            "none" if old_z is None else "%.2f" % old_z,
            new_z,
            gate_rel[0],
        )

    def compute_active_gate_height_replan_z(self, gate_world_raw, gate_rel, now):
        heading_recovery_needed = False
        if self.enable_heading_recovery:
            heading_recovery_needed, _expected_yaw, _yaw_error, _roll_deg, _pitch_deg = (
                self.heading_recovery_status()
            )
        if self.heading_recovery_active or heading_recovery_needed:
            rospy.loginfo_throttle(
                0.5,
                "skip height replan during heading recovery: rel=(%.2f, %.2f, %.2f)",
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
            )
            return None
        freeze_reason = self.height_gate_z_freeze_reason()
        if freeze_reason is not None:
            rospy.loginfo_throttle(
                0.5,
                "skip height replan while PnP gate z is frozen: reason=%s rel=(%.2f, %.2f, %.2f)",
                freeze_reason,
                gate_rel[0],
                gate_rel[1],
                gate_rel[2],
            )
            return None
        if not self.allow_height_replan_while_tracking:
            return None
        if self.gate_sequence_index < self.height_replan_after_gate_index:
            return None
        if self.target_gate_world is None:
            return None
        if gate_rel[0] < self.height_replan_min_gate_distance_m:
            return None
        if (
            self.height_replan_max_abs_rel_z_m > 0.0
            and abs(gate_rel[2]) > self.height_replan_max_abs_rel_z_m
        ):
            rospy.logwarn_throttle(
                0.5,
                "reject height replan vertical outlier: rel_z=%.2f limit=%.2f rel_x=%.2f",
                gate_rel[2],
                self.height_replan_max_abs_rel_z_m,
                gate_rel[0],
            )
            return None
        xy_error = None
        if self.height_replan_active_gate_xy_tolerance_m > 0.0:
            dx = gate_world_raw[0] - self.target_gate_world[0]
            dy = gate_world_raw[1] - self.target_gate_world[1]
            xy_error = math.sqrt(dx * dx + dy * dy)
            if xy_error > self.height_replan_active_gate_xy_tolerance_m:
                if not self.height_replan_z_only_ignore_xy_error:
                    rospy.loginfo_throttle(
                        0.5,
                        "reject height replan non-active gate: xy_error=%.2f > %.2f detected=(%.2f, %.2f, %.2f) active_gate=(%.2f, %.2f, %.2f) rel=(%.2f, %.2f, %.2f)",
                        xy_error,
                        self.height_replan_active_gate_xy_tolerance_m,
                        gate_world_raw[0],
                        gate_world_raw[1],
                        gate_world_raw[2],
                        self.target_gate_world[0],
                        self.target_gate_world[1],
                        self.target_gate_world[2],
                        gate_rel[0],
                        gate_rel[1],
                        gate_rel[2],
                    )
                    return None
                rospy.loginfo_throttle(
                    0.5,
                    "height z-only replan ignores detected XY mismatch: xy_error=%.2f > %.2f keep_active_xy=true detected_z=%.2f active_z=%.2f rel=(%.2f, %.2f, %.2f)",
                    xy_error,
                    self.height_replan_active_gate_xy_tolerance_m,
                    gate_world_raw[2],
                    self.target_gate_world[2],
                    gate_rel[0],
                    gate_rel[1],
                    gate_rel[2],
                )

        predicted_z = self.update_height_reference(
            gate_world_raw,
            gate_rel,
            now,
            commit=True,
        )
        z_delta = predicted_z - self.target_gate_world[2]
        if z_delta > 0.0 and not self.allow_height_down_replan_while_tracking:
            rospy.loginfo_throttle(
                0.5,
                "ignore downward height replan while tracking: old_z=%.2f predicted_z=%.2f dz=%.2f",
                self.target_gate_world[2],
                predicted_z,
                z_delta,
            )
            return None
        if (now - self.last_replan).to_sec() < self.min_replan_interval:
            return None
        if (
            self.height_replan_min_interval_s > 0.0
            and (now - self.last_height_replan_stamp).to_sec()
            < self.height_replan_min_interval_s
        ):
            return None

        visual_error = self.fresh_visual_error(now)
        visual_y = visual_error[1] if visual_error is not None else 0.0
        if abs(visual_y) < self.height_replan_visual_threshold:
            return None
        if abs(z_delta) < self.height_replan_z_threshold_m:
            return None

        # Replan only when the predicted height and image-space vertical error
        # agree on the direction. This avoids chasing one-frame PNP jumps.
        if z_delta * visual_y <= 0.0:
            return None

        max_step = self.height_replan_max_step_m
        if max_step > 0.0 and abs(z_delta) > max_step:
            requested_z = predicted_z
            predicted_z = self.target_gate_world[2] + clamp(z_delta, max_step)
            z_delta = predicted_z - self.target_gate_world[2]
            rospy.logwarn_throttle(
                0.5,
                "limit height replan step: requested_z=%.2f limited_z=%.2f dz=%.2f max_step=%.2f",
                requested_z,
                predicted_z,
                z_delta,
                max_step,
            )

        rospy.loginfo(
            "height replan requested: old_z=%.2f predicted_z=%.2f dz=%.2f visual_y=%.2f rel_x=%.2f xy_error=%s keep_active_xy=true",
            self.target_gate_world[2],
            predicted_z,
            z_delta,
            visual_y,
            gate_rel[0],
            "none" if xy_error is None else "%.2f" % xy_error,
        )
        return predicted_z

    def replan_active_gate_height_z_only(self, new_z, now, is_next_gate=False):
        if self.target_front_world is None or self.target_gate_world is None:
            return False

        front_world = (
            self.target_front_world[0],
            self.target_front_world[1],
            new_z,
        )
        gate_world = (
            self.target_gate_world[0],
            self.target_gate_world[1],
            new_z,
        )
        exit_world = None
        if self.target_exit_world is not None:
            exit_world = (
                self.target_exit_world[0],
                self.target_exit_world[1],
                new_z,
            )

        current_forward_dir = self.current_planning_forward_dir()
        front_progress = dot(
            (
                front_world[0] - self.pos_world[0],
                front_world[1] - self.pos_world[1],
                front_world[2] - self.pos_world[2],
            ),
            current_forward_dir,
        )
        if front_progress < -0.2:
            rospy.logwarn(
                "reject replan behind current position forward_progress=%.2f min=-0.20 front=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
                front_progress,
                front_world[0],
                front_world[1],
                front_world[2],
                self.pos_world[0],
                self.pos_world[1],
                self.pos_world[2],
            )
            return False

        if not self.is_gate_plan_candidate_valid(
            front_world,
            gate_world,
            exit_world,
            current_forward_dir,
            context="replan",
        ):
            return False

        old_z = self.target_gate_world[2]
        previous = self.snapshot_active_plan()
        if not self.build_plan_from_waypoints(
            front_world,
            gate_world,
            exit_world,
            now,
            is_next_gate=is_next_gate,
        ):
            self.restore_active_plan(previous)
            return False

        self.height_ref_world = new_z
        self.height_ref_stamp = now
        self.last_height_replan_stamp = now
        rospy.loginfo(
            "height replan z-only: old_z=%.2f new_z=%.2f keep_xy=true",
            old_z,
            new_z,
        )
        return True

    def snapshot_active_plan(self):
        return {
            "plan": list(self.plan),
            "plan_waypoints": list(self.plan_waypoints),
            "active_waypoint_index": self.active_waypoint_index,
            "active_waypoint_start": self.active_waypoint_start,
            "gate_clear_extensions": self.gate_clear_extensions,
            "trajectory": self.trajectory,
            "plan_start": self.plan_start,
            "plan_total_time": self.plan_total_time,
            "hold_position_world": self.hold_position_world,
            "last_replan": self.last_replan,
            "target_front_world": self.target_front_world,
            "target_gate_world": self.target_gate_world,
            "active_gate_world": self.active_gate_world,
            "aim_gate_world": self.aim_gate_world,
            "target_exit_world": self.target_exit_world,
            "target_forward_dir_world": self.target_forward_dir_world,
            "state": self.state,
        }

    def restore_active_plan(self, snapshot):
        self.plan = snapshot["plan"]
        self.plan_waypoints = snapshot["plan_waypoints"]
        self.active_waypoint_index = snapshot["active_waypoint_index"]
        self.active_waypoint_start = snapshot["active_waypoint_start"]
        self.gate_clear_extensions = snapshot["gate_clear_extensions"]
        self.trajectory = snapshot["trajectory"]
        self.plan_start = snapshot["plan_start"]
        self.plan_total_time = snapshot["plan_total_time"]
        self.hold_position_world = snapshot["hold_position_world"]
        self.last_replan = snapshot["last_replan"]
        self.target_front_world = snapshot["target_front_world"]
        self.target_gate_world = snapshot["target_gate_world"]
        self.active_gate_world = snapshot["active_gate_world"]
        self.aim_gate_world = snapshot["aim_gate_world"]
        self.target_exit_world = snapshot["target_exit_world"]
        self.target_forward_dir_world = snapshot["target_forward_dir_world"]
        self.state = snapshot["state"]
        rospy.logwarn("restore previous active trajectory after rejected height replan")

    def gate_clear_lateral_limit_m(self):
        if self.gate_trigger_width_m <= 0.0:
            return float("inf")
        return max(0.0, 0.5 * self.gate_trigger_width_m - self.gate_clear_lateral_margin_m)

    def gate_clear_vertical_limit_m(self):
        if self.gate_trigger_height_m <= 0.0:
            return float("inf")
        return max(0.0, 0.5 * self.gate_trigger_height_m - self.gate_clear_vertical_margin_m)

    def current_gate_clear_metrics(self):
        active_gate = self.active_gate_world or self.target_gate_world
        if active_gate is None or self.target_forward_dir_world is None:
            return None
        axes = self.build_path_frame_axes(self.current_planning_forward_dir())
        gate_error = world_to_frame(self.pos_world, active_gate, axes)
        center_miss = math.sqrt(
            gate_error[1] * gate_error[1] + gate_error[2] * gate_error[2]
        )
        return {
            "forward": gate_error[0],
            "lateral": gate_error[1],
            "vertical": gate_error[2],
            "center_miss": center_miss,
        }

    def is_near_gate_zone(self, metrics=None):
        if not self.near_gate_zone_enabled:
            return False
        if metrics is None:
            metrics = self.current_gate_clear_metrics()
        if metrics is None:
            return False
        return metrics["forward"] >= -max(0.0, self.near_gate_zone_forward_m)

    def should_prefetch_next_gate(self, metrics=None):
        if not self.near_gate_zone_enabled:
            return False
        if metrics is None:
            metrics = self.current_gate_clear_metrics()
        if metrics is None:
            return False
        return metrics["forward"] >= -max(
            self.near_gate_zone_forward_m,
            self.near_gate_prefetch_forward_m,
        )

    def current_aim_gate_world(self):
        if self.aim_gate_world is not None:
            if (
                self.near_gate_forbid_current_gate_yaw
                and self.is_near_gate_zone()
                and self.active_gate_world is not None
                and distance(self.aim_gate_world, self.active_gate_world)
                <= self.effective_cleared_gate_ignore_radius_m()
            ):
                return None
            return self.aim_gate_world
        if self.near_gate_forbid_current_gate_yaw and self.is_near_gate_zone():
            return None
        return self.target_gate_world

    def current_gate_passed_distance(self):
        metrics = self.current_gate_clear_metrics()
        if metrics is None:
            return None
        return metrics["forward"]

    def can_clear_active_gate(
        self,
        forward,
        lateral,
        vertical,
        center_miss,
        recovery_active,
        tilt_deg,
        yaw_error_deg,
    ):
        clear_distance = self.effective_clear_distance_m()
        if forward < clear_distance:
            return False, "forward %.2f < %.2f" % (forward, clear_distance)

        if self.gate_clear_block_during_recovery and recovery_active:
            if not self.gate_clear_recovery_strict_enabled:
                return False, "heading recovery active"
            if abs(lateral) > self.gate_clear_recovery_lateral_limit_m:
                return False, "recovery_not_centered lateral %.2f > %.2f" % (
                    abs(lateral),
                    self.gate_clear_recovery_lateral_limit_m,
                )
            if abs(vertical) > self.gate_clear_recovery_vertical_limit_m:
                return False, "recovery_not_centered vertical %.2f > %.2f" % (
                    abs(vertical),
                    self.gate_clear_recovery_vertical_limit_m,
                )
            if center_miss > self.gate_clear_recovery_center_miss_m:
                return False, "recovery_not_centered center_miss %.2f > %.2f" % (
                    center_miss,
                    self.gate_clear_recovery_center_miss_m,
                )
            return True, "strict recovery clear"

        if (
            self.gate_clear_block_on_large_attitude
            and self.max_vision_tilt_deg > 0.0
            and tilt_deg > self.max_vision_tilt_deg
        ):
            return False, "tilt %.1f > %.1f" % (tilt_deg, self.max_vision_tilt_deg)

        if (
            self.gate_clear_block_on_yaw_mismatch
            and self.max_vision_yaw_error_deg > 0.0
            and abs(yaw_error_deg) > self.max_vision_yaw_error_deg
        ):
            return False, "yaw error %.1f > %.1f" % (
                abs(yaw_error_deg),
                self.max_vision_yaw_error_deg,
            )

        if not self.gate_clear_require_center_check:
            return True, "ok"

        lateral_limit = self.gate_clear_lateral_limit_m()
        if abs(lateral) > lateral_limit:
            return False, "lateral %.2f > %.2f" % (abs(lateral), lateral_limit)

        vertical_limit = self.gate_clear_vertical_limit_m()
        if abs(vertical) > vertical_limit:
            return False, "vertical %.2f > %.2f" % (abs(vertical), vertical_limit)

        if self.gate_clear_center_miss_m > 0.0 and center_miss > self.gate_clear_center_miss_m:
            return False, "center_miss %.2f > %.2f" % (
                center_miss,
                self.gate_clear_center_miss_m,
            )

        return True, "ok"

    def current_gate_clear_decision(self):
        metrics = self.current_gate_clear_metrics()
        if metrics is None:
            return {
                "allowed": False,
                "forward_passed": False,
                "reason": "no active gate",
                "forward": float("nan"),
                "lateral": float("nan"),
                "vertical": float("nan"),
                "center_miss": float("nan"),
                "clear_distance": self.effective_clear_distance_m(),
            }

        expected_yaw = self.heading_recovery_expected_yaw()
        yaw_error = wrap_angle(self.yaw - expected_yaw)
        tilt_deg = max(abs(math.degrees(self.roll)), abs(math.degrees(self.pitch)))
        recovery_for_clear = self.heading_recovery_active
        if not recovery_for_clear and self.enable_heading_recovery:
            recovery_needed, _expected_yaw, _yaw_error, _roll_deg, _pitch_deg = (
                self.heading_recovery_status()
            )
            recovery_for_clear = recovery_needed
        allowed, reason = self.can_clear_active_gate(
            metrics["forward"],
            metrics["lateral"],
            metrics["vertical"],
            metrics["center_miss"],
            recovery_for_clear,
            tilt_deg,
            math.degrees(yaw_error),
        )
        decision = dict(metrics)
        decision.update(
            {
                "allowed": allowed,
                "forward_passed": metrics["forward"] >= self.effective_clear_distance_m(),
                "reason": reason,
                "clear_distance": self.effective_clear_distance_m(),
                "recovery_active": recovery_for_clear,
                "tilt_deg": tilt_deg,
                "yaw_error_deg": math.degrees(yaw_error),
            }
        )
        self.last_clear_decision = decision
        return decision

    def has_cleared_current_gate(self):
        return self.current_gate_clear_decision()["allowed"]

    def log_gate_center_tracking_diagnostics(self, now, desired_pos):
        if self.gate_center_diag_interval_s <= 0.0 or self.target_gate_world is None:
            return

        axes = self.build_path_frame_axes(self.current_planning_forward_dir())
        cmd_error = world_to_frame(self.pos_world, desired_pos, axes)
        gate_error = world_to_frame(self.pos_world, self.target_gate_world, axes)
        visual_error = self.fresh_visual_error(now)
        if visual_error is None:
            visual_x = float("nan")
            visual_y = float("nan")
            visual_area = float("nan")
        else:
            visual_x, visual_y, visual_area = visual_error

        warn_tracking = (
            abs(cmd_error[0]) > self.track_error_warn_forward_m
            or abs(cmd_error[1]) > self.track_error_warn_lateral_m
            or abs(cmd_error[2]) > self.track_error_warn_vertical_m
        )
        log_fn = rospy.logwarn_throttle if warn_tracking else rospy.loginfo_throttle
        log_fn(
            self.gate_center_diag_interval_s,
            "track center diag: state=%s cmd_err=(forward=%.2f lateral=%.2f vertical=%.2f) gate_err=(forward=%.2f lateral=%.2f vertical=%.2f) visual=(%.2f, %.2f area=%.3f) pos=(%.2f, %.2f, %.2f) cmd=(%.2f, %.2f, %.2f) gate=(%.2f, %.2f, %.2f)",
            self.state,
            cmd_error[0],
            cmd_error[1],
            cmd_error[2],
            gate_error[0],
            gate_error[1],
            gate_error[2],
            visual_x,
            visual_y,
            visual_area,
            self.pos_world[0],
            self.pos_world[1],
            self.pos_world[2],
            desired_pos[0],
            desired_pos[1],
            desired_pos[2],
            self.target_gate_world[0],
            self.target_gate_world[1],
            self.target_gate_world[2],
        )

    def log_gate_center_pass_diagnostics(self):
        if self.target_gate_world is None:
            return
        metrics = self.current_gate_clear_metrics()
        if metrics is None:
            return
        center_miss = metrics["center_miss"]
        log_fn = rospy.logwarn if center_miss > 1.0 else rospy.loginfo
        log_fn(
            "gate center pass: forward=%.2f lateral=%.2f vertical=%.2f center_miss=%.2f pos=(%.2f, %.2f, %.2f) gate=(%.2f, %.2f, %.2f)",
            metrics["forward"],
            metrics["lateral"],
            metrics["vertical"],
            center_miss,
            self.pos_world[0],
            self.pos_world[1],
            self.pos_world[2],
            self.target_gate_world[0],
            self.target_gate_world[1],
            self.target_gate_world[2],
        )

    def replan_to_active_gate_from_current(self, now, reason):
        if self.command_mode != "trajectory" or self.target_gate_world is None:
            return False

        forward_dir = self.current_planning_forward_dir()
        gate_vec = (
            self.target_gate_world[0] - self.pos_world[0],
            self.target_gate_world[1] - self.pos_world[1],
            0.0,
        )
        horizontal_dist = math.sqrt(gate_vec[0] * gate_vec[0] + gate_vec[1] * gate_vec[1])
        approach_dir = normalize_vector(gate_vec, forward_dir)
        front_distance = min(max(0.5, self.front_offset_m), max(0.5, 0.5 * horizontal_dist))
        front_ratio = (
            min(1.0, front_distance / horizontal_dist)
            if horizontal_dist > 1.0e-6
            else 0.0
        )
        front_world = (
            self.pos_world[0] + approach_dir[0] * front_distance,
            self.pos_world[1] + approach_dir[1] * front_distance,
            self.pos_world[2]
            + front_ratio * (self.target_gate_world[2] - self.pos_world[2]),
        )
        exit_world = self.target_exit_world
        if exit_world is None:
            exit_offset = self.effective_exit_offset_m()
            exit_world = (
                self.target_gate_world[0] + forward_dir[0] * exit_offset,
                self.target_gate_world[1] + forward_dir[1] * exit_offset,
                self.target_gate_world[2],
            )

        replanned = self.build_plan_from_waypoints(
            front_world,
            self.target_gate_world,
            exit_world,
            now,
            is_next_gate=(self.state == TRACK_NEXT_GATE),
        )
        rospy.logwarn(
            "continue same gate: replan_to_active_gate=%s reason=%s front=(%.2f, %.2f, %.2f) active_gate=(%.2f, %.2f, %.2f) active_gate_z_kept_after_clear_reject=%.2f",
            str(bool(replanned)),
            reason,
            front_world[0],
            front_world[1],
            front_world[2],
            self.target_gate_world[0],
            self.target_gate_world[1],
            self.target_gate_world[2],
            self.target_gate_world[2],
        )
        return bool(replanned)

    def build_plan_to_gate_world(self, gate_world, forward_dir, now, reason):
        if self.command_mode != "trajectory" or gate_world is None:
            return False

        forward_dir = self.horizontal_forward_dir(forward_dir)
        gate_vec = (
            gate_world[0] - self.pos_world[0],
            gate_world[1] - self.pos_world[1],
            0.0,
        )
        horizontal_dist = math.sqrt(gate_vec[0] * gate_vec[0] + gate_vec[1] * gate_vec[1])
        approach_dir = normalize_vector(gate_vec, forward_dir)
        front_distance = min(max(0.5, self.front_offset_m), max(0.5, 0.5 * horizontal_dist))
        front_ratio = (
            min(1.0, front_distance / horizontal_dist)
            if horizontal_dist > 1.0e-6
            else 0.0
        )
        front_world = (
            self.pos_world[0] + approach_dir[0] * front_distance,
            self.pos_world[1] + approach_dir[1] * front_distance,
            self.pos_world[2] + front_ratio * (gate_world[2] - self.pos_world[2]),
        )
        exit_offset = self.effective_exit_offset_m()
        exit_world = (
            gate_world[0] + forward_dir[0] * exit_offset,
            gate_world[1] + forward_dir[1] * exit_offset,
            gate_world[2],
        )
        if not self.is_gate_plan_candidate_valid(
            front_world,
            gate_world,
            exit_world,
            forward_dir,
            context=reason,
        ):
            return False

        planned = self.build_plan_from_waypoints(
            front_world,
            gate_world,
            exit_world,
            now,
            is_next_gate=True,
        )
        rospy.logwarn(
            "plan to prefetched next gate: planned=%s reason=%s front=(%.2f, %.2f, %.2f) gate=(%.2f, %.2f, %.2f) exit=(%.2f, %.2f, %.2f)",
            str(bool(planned)),
            reason,
            front_world[0],
            front_world[1],
            front_world[2],
            gate_world[0],
            gate_world[1],
            gate_world[2],
            exit_world[0],
            exit_world[1],
            exit_world[2],
        )
        return bool(planned)

    def mark_active_gate_missed(self, now, reason, decision=None):
        missed_gate = self.active_gate_world or self.target_gate_world
        if missed_gate is None:
            return False

        forward_dir = self.horizontal_forward_dir(self.target_forward_dir_world)
        prefetched_next_gate = (
            self.next_gate_candidate_world
            if self.next_gate_candidate_count >= self.new_gate_stable_frames
            else None
        )
        self.missed_gate_world_positions.append(missed_gate)
        self.last_missed_gate_world = missed_gate
        self.last_cleared_gate_world = missed_gate
        # Treat a missed gate as processed for search/height staging, but keep it
        # separate from cleared gates for diagnostics.
        self.gate_sequence_index += 1

        self.clear_active_plan_after_finish(clear_target=True)
        self.reset_heading_recovery()
        self.reset_new_gate_candidate()
        self.search_forward_dir_world = forward_dir
        self.search_start_world = self.pos_world
        self.search_next_altitude_world = None
        self.update_search_next_altitude_target(now)
        self.search_next_target_world = None
        self.search_next_target_stamp = None
        self.search_next_enter_time = now
        self.search_next_settle_target_world = None
        self.last_search_steer_stamp = None
        self.yaw_reference = math.atan2(forward_dir[1], forward_dir[0])
        self.last_yaw_time = now
        self.set_state(ADVANCE_SEARCH_NEXT_GATE)

        if decision is None:
            decision = {}
        rospy.logwarn(
            "mark active gate missed: reason=%s missed_gate=(%.2f, %.2f, %.2f) forward=%s lateral=%s vertical=%s center_miss=%s next_state=%s gate_sequence_index=%d",
            reason,
            missed_gate[0],
            missed_gate[1],
            missed_gate[2],
            "none" if "forward" not in decision else "%.2f" % decision["forward"],
            "none" if "lateral" not in decision else "%.2f" % decision["lateral"],
            "none" if "vertical" not in decision else "%.2f" % decision["vertical"],
            "none" if "center_miss" not in decision else "%.2f" % decision["center_miss"],
            self.state,
            self.gate_sequence_index,
        )
        if prefetched_next_gate is not None:
            if self.build_plan_to_gate_world(
                prefetched_next_gate,
                forward_dir,
                now,
                "missed active gate, use prefetched next",
            ):
                return True
        return True

    def handle_rejected_gate_clear(self, now, decision):
        rospy.logwarn_throttle(
            0.3,
            "gate clear rejected: reason=%s forward=%.2f/%.2f lateral=%.2f limit=%.2f vertical=%.2f limit=%.2f center_miss=%.2f limit=%.2f recovery=%s tilt=%.1f yaw_error=%.1f",
            decision["reason"],
            decision["forward"],
            decision["clear_distance"],
            decision["lateral"],
            self.gate_clear_lateral_limit_m(),
            decision["vertical"],
            self.gate_clear_vertical_limit_m(),
            decision["center_miss"],
            self.gate_clear_center_miss_m,
            str(decision.get("recovery_active", False)),
            decision.get("tilt_deg", float("nan")),
            decision.get("yaw_error_deg", float("nan")),
        )
        if (
            decision["forward"]
            >= decision["clear_distance"] + self.gate_clear_recovery_behind_margin_m
        ):
            missed_reason = (
                "MISS_ACTIVE_GATE_RECOVERY: "
                if decision.get("recovery_active", False)
                else "MISS_ACTIVE_GATE: "
            ) + decision["reason"]
            rospy.logwarn(
                "gate behind without valid clear, abandon active gate instead of chasing it: forward=%.2f clear=%.2f center_miss=%.2f recovery=%s reason=%s",
                decision["forward"],
                decision["clear_distance"],
                decision["center_miss"],
                str(decision.get("recovery_active", False)),
                decision["reason"],
            )
            return self.mark_active_gate_missed(now, missed_reason, decision)

        if self.heading_recovery_active:
            return False
        elapsed = (now - self.last_gate_clear_reject_stamp).to_sec()
        if (
            self.gate_clear_reject_replan_interval_s > 0.0
            and elapsed < self.gate_clear_reject_replan_interval_s
        ):
            return False
        self.last_gate_clear_reject_stamp = now
        return self.replan_to_active_gate_from_current(now, decision["reason"])

    def finish_current_gate(self, now, hold_position=None):
        decision = self.current_gate_clear_decision()
        if not decision["allowed"]:
            if decision["forward_passed"]:
                self.handle_rejected_gate_clear(now, decision)
            return False

        passed_distance = self.current_gate_passed_distance()
        cleared_gate_world = self.active_gate_world or self.target_gate_world
        search_forward_dir = self.horizontal_forward_dir(self.target_forward_dir_world)
        prefetched_next_gate = (
            self.next_gate_candidate_world
            if self.next_gate_candidate_count >= self.new_gate_stable_frames
            else None
        )

        if cleared_gate_world is not None:
            self.cleared_gate_world_positions.append(cleared_gate_world)
            self.gate_sequence_index += 1
        search_altitude = self.search_next_altitude_target()

        self.log_gate_center_pass_diagnostics()
        if decision["reason"] == "strict recovery clear":
            rospy.logwarn(
                "strict recovery clear: allowed forward=%.2f lateral=%.2f vertical=%.2f center_miss=%.2f",
                decision["forward"],
                decision["lateral"],
                decision["vertical"],
                decision["center_miss"],
            )
        rospy.loginfo(
            "gate cleared by world projection: passed_distance=%.2f clear_distance=%.2f reason=%s",
            passed_distance if passed_distance is not None else 0.0,
            self.effective_clear_distance_m(),
            decision["reason"],
        )
        self.clear_active_plan_after_finish(clear_target=True)
        self.reset_new_gate_candidate()

        if self.stop_after_first_gate:
            self.gate_completed = True
            self.set_state(CLEAR_GATE)
            self.publish_stop_or_hold(now, hold_position)
            rospy.loginfo("gate completed, holding position")
            return

        self.last_cleared_gate_world = cleared_gate_world
        self.search_forward_dir_world = search_forward_dir
        self.search_start_world = self.pos_world
        self.search_next_altitude_world = search_altitude
        self.search_next_target_world = None
        self.search_next_target_stamp = None
        self.search_next_enter_time = now
        self.last_search_steer_stamp = None
        self.search_next_settle_target_world = (
            self.pos_world[0],
            self.pos_world[1],
            search_altitude,
        )
        self.yaw_reference = math.atan2(search_forward_dir[1], search_forward_dir[0])
        self.last_yaw_time = now
        if prefetched_next_gate is not None:
            if self.build_plan_to_gate_world(
                prefetched_next_gate,
                search_forward_dir,
                now,
                "cleared active gate, use prefetched next",
            ):
                rospy.loginfo("planned prefetched next gate immediately after clear")
                return True
        self.set_state(CLEAR_GATE)
        self.set_state(CRUISE_SEARCH_NEXT_GATE)
        rospy.loginfo("gate cleared, cruising to search next gate")
        self.publish_search_next_gate(now)
        return True

    def control_loop(self, _event):
        if not self.got_pose:
            rospy.logwarn_throttle(1.0, "waiting for current pose")
            return

        now = rospy.Time.now()
        if self.gate_completed:
            self.publish_stop_or_hold(now)
            rospy.loginfo_throttle(1.0, "gate completed, holding position")
            return

        if self.should_wait_for_initial_gate(now):
            self.set_state(TAKEOFF)
            rospy.loginfo_throttle(1.0, "waiting for initial gate before se3 takeoff")
            return

        if self.should_hold_takeoff_position():
            self.set_state(TAKEOFF)
            self.publish_takeoff_hold(now)
            return

        if self.state == TAKEOFF:
            self.set_state(SEARCH_GATE)

        if (
            self.state
            in (CRUISE_SEARCH_NEXT_GATE, ADVANCE_SEARCH_NEXT_GATE, CONFIRM_NEXT_GATE)
            and not self.plan
        ):
            if self.state == ADVANCE_SEARCH_NEXT_GATE:
                self.publish_search_next_gate(now)
                return
            if self.should_publish_heading_recovery(now):
                self.publish_heading_recovery(now)
                return
            self.publish_search_next_gate(now)
            return

        if not self.plan:
            if self.should_publish_heading_recovery(now):
                self.publish_heading_recovery(now)
                return
            self.set_state(SEARCH_GATE)
            self.publish_search_or_hover(now)
            return

        heading_recovery_needed = self.should_publish_heading_recovery(now)
        clear_decision = self.current_gate_clear_decision()
        if self.is_near_gate_zone(clear_decision):
            aim_gate = self.current_aim_gate_world()
            rospy.loginfo_throttle(
                0.5,
                "near gate zone: active gate is clear/miss only forward=%.2f lateral=%.2f vertical=%.2f center_miss=%.2f aim=%s next_count=%d/%d",
                clear_decision["forward"],
                clear_decision["lateral"],
                clear_decision["vertical"],
                clear_decision["center_miss"],
                "forward"
                if aim_gate is None
                else "(%.2f, %.2f, %.2f)" % (aim_gate[0], aim_gate[1], aim_gate[2]),
                self.next_gate_candidate_count,
                self.new_gate_stable_frames,
            )
        if clear_decision["forward_passed"]:
            if clear_decision["allowed"]:
                self.finish_current_gate(now)
                return
            replanned_after_reject = self.handle_rejected_gate_clear(now, clear_decision)
            if self.state == ADVANCE_SEARCH_NEXT_GATE:
                self.publish_search_next_gate(now)
                return
            if heading_recovery_needed and not replanned_after_reject:
                self.publish_heading_recovery(now)
            return

        if heading_recovery_needed:
            self.publish_heading_recovery(now)
            return

        if (
            self.command_mode == "trajectory"
            and self.trajectory_waypoint_hold
            and not self.use_min_snap
        ):
            self.track_waypoint_plan(now)
            return

        elapsed = (now - self.plan_start).to_sec()
        if elapsed >= self.plan_total_time:
            clear_decision = self.current_gate_clear_decision()
            if self.require_gate_clear_before_finish and not clear_decision["allowed"]:
                if clear_decision["forward_passed"]:
                    self.handle_rejected_gate_clear(now, clear_decision)
                    return
                self.publish_clear_gate_continuation(now)
                return
            final_hold = self.plan[-1][1] if self.plan else self.pos_world
            self.hold_position_world = final_hold
            if self.finish_current_gate(now, final_hold):
                rospy.loginfo_throttle(1.0, "gate trajectory finished")
            return

        desired_pos, desired_vel, desired_acc = self.sample_plan(elapsed)
        yaw_rate = self.compute_yaw_rate(now) if self.enable_yaw_tracking else 0.0
        if self.command_mode == "trajectory":
            traj_source = self.last_trajectory_sample_source
            if self.trajectory_use_feedforward:
                desired_vel_cmd = self.limit_vector_norm(desired_vel, self.max_speed_mps)
                desired_acc_cmd = self.limit_vector_norm(desired_acc, self.max_acc_mps2)
            else:
                desired_vel_cmd = (0.0, 0.0, 0.0)
                desired_acc_cmd = (0.0, 0.0, 0.0)
            yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate)
            self.log_gate_center_tracking_diagnostics(now, desired_pos)
            self.publish_trajectory_cmd(
                desired_pos, desired_vel_cmd, desired_acc_cmd, yaw, yaw_rate_rad
            )
            horizontal_speed = math.sqrt(
                desired_vel_cmd[0] * desired_vel_cmd[0]
                + desired_vel_cmd[1] * desired_vel_cmd[1]
            )
            if horizontal_speed > 1.0e-3:
                velocity_yaw = math.atan2(desired_vel_cmd[1], desired_vel_cmd[0])
            else:
                velocity_yaw = float("nan")
            gate_bearing_yaw, _gate_bearing_dist = self.active_gate_bearing_yaw()
            if gate_bearing_yaw is None:
                gate_bearing_yaw = float("nan")
                velocity_to_gate_error = float("nan")
            elif math.isnan(velocity_yaw):
                velocity_to_gate_error = float("nan")
            else:
                velocity_to_gate_error = wrap_angle(velocity_yaw - gate_bearing_yaw)
            rospy.loginfo_throttle(
                0.5,
                "track se3 traj: source=%s pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f) acc=(%.2f, %.2f, %.2f) yawRate=%.2f velocity_yaw=%.1f gate_bearing_yaw=%.1f velocity_to_gate_error=%.1f",
                traj_source,
                desired_pos[0],
                desired_pos[1],
                desired_pos[2],
                desired_vel_cmd[0],
                desired_vel_cmd[1],
                desired_vel_cmd[2],
                desired_acc_cmd[0],
                desired_acc_cmd[1],
                desired_acc_cmd[2],
                yaw_rate,
                math.degrees(velocity_yaw),
                math.degrees(gate_bearing_yaw),
                math.degrees(velocity_to_gate_error),
            )
            return

        err = (
            desired_pos[0] - self.pos_world[0],
            desired_pos[1] - self.pos_world[1],
            desired_pos[2] - self.pos_world[2],
        )
        cmd_world = (
            desired_vel[0] + self.kp_x * err[0],
            desired_vel[1] + self.kp_pos * err[1],
            desired_vel[2] + self.kp_z * err[2],
        )
        cmd_body = self.world_vector_to_body(cmd_world)

        vx = clamp(cmd_body[0], self.max_vx)
        vy = clamp(cmd_body[1], self.max_vy)
        vz = clamp(cmd_body[2], self.max_vz)
        vx = self.filter_forward_cmd(vx, now)
        vz = self.filter_vertical_cmd(vz)

        self.publish_cmd(vx, vy, vz, yaw_rate)
        rospy.loginfo_throttle(
            0.5,
            "track gate traj: vx=%.2f vy=%.2f vz=%.2f yawRate=%.2f err=(%.2f, %.2f, %.2f)",
            vx,
            vy,
            vz,
            yaw_rate,
            err[0],
            err[1],
            err[2],
        )

    def diagnostic_loop(self, _event):
        if self.got_pose:
            return

        topics = dict(rospy.get_published_topics())
        if self.pose_topic not in topics:
            rospy.logwarn(
                "pose topic is not published: %s. Check simulator, ROS_MASTER_URI, or launch pose_topic.",
                self.pose_topic,
            )
            return

        topic_type = topics[self.pose_topic]
        if topic_type != "geometry_msgs/PoseStamped":
            rospy.logwarn(
                "pose topic type mismatch: %s is %s, expected geometry_msgs/PoseStamped.",
                self.pose_topic,
                topic_type,
            )

    def sample_plan(self, elapsed):
        with self.plan_lock:
            plan_snapshot = list(self.plan)
            trajectory = self.trajectory

        fallback_pos, fallback_vel, fallback_acc = self.sample_quintic_plan(
            elapsed, plan_snapshot
        )
        self.last_trajectory_sample_source = "quintic"

        if trajectory is not None:
            desired_pos = trajectory.eval(elapsed, 0)
            desired_vel = trajectory.eval(elapsed, 1)
            desired_acc = trajectory.eval(elapsed, 2)
            if self.is_min_snap_sample_safe(elapsed, desired_pos, fallback_pos):
                self.last_trajectory_sample_source = "min-snap"
                return desired_pos, desired_vel, desired_acc

            self.last_trajectory_sample_source = "emergency-quintic"
            rospy.logerr_throttle(
                0.5,
                "EMERGENCY min-snap sample out of bounds, using quintic fallback: min_snap=(%.2f, %.2f, %.2f) quintic=(%.2f, %.2f, %.2f) deviation=%.2f",
                desired_pos[0],
                desired_pos[1],
                desired_pos[2],
                fallback_pos[0],
                fallback_pos[1],
                fallback_pos[2],
                distance(desired_pos, fallback_pos),
            )
            return fallback_pos, fallback_vel, fallback_acc

        if self.command_mode == "trajectory" and self.use_min_snap:
            self.last_trajectory_sample_source = "emergency-quintic"
            rospy.logerr_throttle(
                0.5,
                "EMERGENCY min-snap trajectory missing while use_min_snap=True, using quintic fallback",
            )

        return fallback_pos, fallback_vel, fallback_acc

    def sample_quintic_plan(self, elapsed, plan=None):
        active_plan = plan if plan is not None else self.plan
        if not active_plan:
            return self.pos_world, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        t = elapsed
        for p0, p1, duration in active_plan:
            if t <= duration:
                return eval_quintic(p0, p1, duration, t)
            t -= duration

        p0, p1, duration = active_plan[-1]
        return eval_quintic(p0, p1, duration, duration)

    def is_min_snap_sample_safe(self, elapsed, desired_pos, fallback_pos):
        if not self.trajectory_sample_guard_enabled:
            return True

        if (
            self.trajectory_sample_guard_max_deviation_m > 0.0
            and distance(desired_pos, fallback_pos)
            > self.trajectory_sample_guard_max_deviation_m
        ):
            return False

        if self.trajectory_sample_guard_corridor_m <= 0.0:
            return True

        segment = self.locate_plan_segment(elapsed)
        if segment is None:
            return True

        p0, p1, _duration = segment
        segment_vec = (
            p1[0] - p0[0],
            p1[1] - p0[1],
            p1[2] - p0[2],
        )
        segment_len = norm(segment_vec)
        if segment_len < 1.0e-6:
            return True

        unit = (
            segment_vec[0] / segment_len,
            segment_vec[1] / segment_len,
            segment_vec[2] / segment_len,
        )
        rel = (
            desired_pos[0] - p0[0],
            desired_pos[1] - p0[1],
            desired_pos[2] - p0[2],
        )
        progress = dot(rel, unit)
        lateral = (
            rel[0] - progress * unit[0],
            rel[1] - progress * unit[1],
            rel[2] - progress * unit[2],
        )
        corridor = self.trajectory_sample_guard_corridor_m
        return (
            progress >= -corridor
            and progress <= segment_len + corridor
            and norm(lateral) <= corridor
        )

    def locate_plan_segment(self, elapsed):
        t = elapsed
        for p0, p1, duration in self.plan:
            if t <= duration:
                return p0, p1, duration
            t -= duration
        if not self.plan:
            return None
        p0, p1, duration = self.plan[-1]
        return p0, p1, duration

    def is_gate_measurement_plausible(self, rel):
        if rel[0] < self.min_front_x_m:
            return False
        if self.max_gate_lateral_m > 0.0 and abs(rel[1]) > self.max_gate_lateral_m:
            return False
        if self.max_gate_vertical_m > 0.0 and abs(rel[2]) > self.max_gate_vertical_m:
            return False
        return True

    def gate_measurement_reject_reason(self, rel):
        if rel[0] < self.min_front_x_m:
            return "front_x %.2f < %.2f" % (rel[0], self.min_front_x_m)
        if self.max_gate_lateral_m > 0.0 and abs(rel[1]) > self.max_gate_lateral_m:
            return "abs(y) %.2f > %.2f" % (
                abs(rel[1]),
                self.max_gate_lateral_m,
            )
        if self.max_gate_vertical_m > 0.0 and abs(rel[2]) > self.max_gate_vertical_m:
            return "abs(z) %.2f > %.2f" % (
                abs(rel[2]),
                self.max_gate_vertical_m,
            )
        return "unknown"

    def limit_trajectory_velocity(self, velocity):
        return (
            clamp(velocity[0], self.trajectory_max_vx),
            clamp(velocity[1], self.trajectory_max_vy),
            clamp(velocity[2], self.trajectory_max_vz),
        )

    def limit_vector_norm(self, vector, max_norm):
        if max_norm <= 0.0:
            return vector
        norm = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        if norm <= max_norm or norm < 1.0e-9:
            return vector
        scale = max_norm / norm
        return (vector[0] * scale, vector[1] * scale, vector[2] * scale)

    def should_hold_takeoff_position(self):
        if self.command_mode != "trajectory" or not self.enable_takeoff_hold:
            return False

        if self.takeoff_reached or self.takeoff_target_world is None:
            return False

        if distance(self.pos_world, self.takeoff_target_world) <= self.takeoff_reached_tolerance_m:
            self.takeoff_reached = True
            self.clear_stale_plan()
            self.set_state(SEARCH_GATE)
            rospy.loginfo("se3 takeoff hold reached, start accepting gate plans")
            self.plan_pending_gate_if_fresh(rospy.Time.now())
            return False

        return True

    def is_takeoff_pending(self):
        if self.command_mode != "trajectory" or not self.enable_takeoff_hold:
            return False
        if self.takeoff_reached or self.takeoff_target_world is None:
            return False
        return True

    def should_wait_for_initial_gate(self, now):
        if not self.wait_for_gate_before_takeoff:
            return False
        if not self.is_takeoff_pending():
            return False
        if self.initial_gate_wait_done or self.pending_gate_world is not None:
            return False

        if self.initial_gate_wait_start is None:
            self.initial_gate_wait_start = now

        elapsed = (now - self.initial_gate_wait_start).to_sec()
        if self.initial_gate_wait_timeout <= 0.0:
            return True
        if elapsed < self.initial_gate_wait_timeout:
            return True

        self.initial_gate_wait_done = True
        rospy.logwarn(
            "initial gate wait timed out after %.2fs, start se3 takeoff anyway",
            elapsed,
        )
        return False

    def clear_stale_plan(self):
        self.plan = []
        self.plan_waypoints = []
        self.active_waypoint_index = 0
        self.active_waypoint_start = None
        self.gate_clear_extensions = 0
        self.trajectory = None
        self.plan_start = None
        self.plan_total_time = 0.0
        self.hold_position_world = None
        self.target_front_world = None
        self.target_gate_world = None
        self.active_gate_world = None
        self.aim_gate_world = None
        self.target_exit_world = None
        self.target_forward_dir_world = None
        self.filtered_vx_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.last_cmd_time = None
        self.search_next_target_world = None
        self.search_next_target_stamp = None
        self.search_forward_dir_world = None
        self.search_start_world = None
        self.search_next_altitude_world = None
        self.search_next_enter_time = None
        self.search_next_settle_target_world = None
        self.last_search_steer_stamp = None
        self.next_gate_candidate_world = None
        self.next_gate_candidate_count = 0
        self.next_gate_candidate_stamp = None

    def plan_pending_gate_if_fresh(self, now):
        if (
            self.pending_gate_rel is None
            and (self.pending_gate_world is None or self.pending_front_world is None)
        ):
            return

        if self.pending_gate_stamp is not None:
            age = (now - self.pending_gate_stamp).to_sec()
            if self.pending_gate_max_age > 0.0 and age > self.pending_gate_max_age:
                rospy.logwarn(
                    "drop stale pending gate after takeoff: age=%.2fs", age
                )
                return

        if self.pending_gate_rel is not None:
            front_world = None
            gate_world, front_world, exit_world = self.compute_gate_waypoints(
                self.pending_gate_rel,
                now,
                update_height=True,
            )
        else:
            front_world = self.pending_front_world
            gate_world = self.pending_gate_world
            exit_world = self.pending_exit_world

        if not self.is_gate_plan_candidate_valid(
            front_world,
            gate_world,
            exit_world,
            self.current_planning_forward_dir(),
            context="gate candidate",
        ):
            self.pending_front_world = None
            self.pending_gate_world = None
            self.pending_exit_world = None
            self.pending_gate_rel = None
            self.pending_gate_stamp = None
            return

        self.build_plan_from_waypoints(
            front_world,
            gate_world,
            exit_world,
            now,
        )
        rospy.loginfo("planned cached gate after se3 takeoff hold")
        self.pending_front_world = None
        self.pending_gate_world = None
        self.pending_exit_world = None
        self.pending_gate_rel = None
        self.pending_gate_stamp = None

    def track_waypoint_plan(self, now):
        if not self.plan_waypoints:
            self.publish_stop_or_hold(now)
            return

        if self.active_waypoint_index >= len(self.plan_waypoints):
            final_hold = self.plan_waypoints[-1]
            if self.extend_exit_if_gate_still_ahead(now, final_hold):
                return
            if self.finish_current_gate(now, final_hold):
                rospy.loginfo_throttle(1.0, "gate waypoint plan finished")
            return

        target = self.plan_waypoints[self.active_waypoint_index]
        if self.active_waypoint_index >= len(self.plan_waypoints) - 1:
            self.set_state(CLEAR_GATE)
        else:
            self.set_state(TRACK_GATE)
        error = distance(self.pos_world, target)
        timed_out = False
        if self.active_waypoint_start is None:
            self.active_waypoint_start = now
        elif self.trajectory_waypoint_timeout > 0.0:
            timed_out = (
                now - self.active_waypoint_start
            ).to_sec() > self.trajectory_waypoint_timeout

        if error <= self.trajectory_waypoint_tolerance_m or timed_out:
            if timed_out:
                rospy.logwarn(
                    "se3 waypoint %d timed out: target=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f) err=%.2f",
                    self.active_waypoint_index + 1,
                    target[0],
                    target[1],
                    target[2],
                    self.pos_world[0],
                    self.pos_world[1],
                    self.pos_world[2],
                    error,
                )

            self.active_waypoint_index += 1
            self.active_waypoint_start = now
            if self.active_waypoint_index >= len(self.plan_waypoints):
                final_hold = target
                if self.extend_exit_if_gate_still_ahead(now, final_hold):
                    return
                if self.finish_current_gate(now, final_hold):
                    rospy.loginfo("gate waypoint plan finished")
                return

            target = self.plan_waypoints[self.active_waypoint_index]
            if self.active_waypoint_index >= len(self.plan_waypoints) - 1:
                self.set_state(CLEAR_GATE)
            else:
                self.set_state(TRACK_GATE)
            error = distance(self.pos_world, target)
            rospy.loginfo(
                "advance to se3 waypoint %d/%d: target=(%.2f, %.2f, %.2f)",
                self.active_waypoint_index + 1,
                len(self.plan_waypoints),
                target[0],
                target[1],
                target[2],
            )

        command_target = self.limited_trajectory_tracking_target(target)
        yaw_rate = self.compute_yaw_rate(now) if self.enable_yaw_tracking else 0.0
        yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate)
        self.publish_trajectory_cmd(
            command_target,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )
        rospy.loginfo_throttle(
            0.5,
            "track se3 waypoint %d/%d: target=(%.2f, %.2f, %.2f) cmd=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f) err=%.2f",
            self.active_waypoint_index + 1,
            len(self.plan_waypoints),
            target[0],
            target[1],
            target[2],
            command_target[0],
            command_target[1],
            command_target[2],
            self.pos_world[0],
            self.pos_world[1],
            self.pos_world[2],
            error,
        )

    def limited_trajectory_tracking_target(self, target):
        lookahead = self.trajectory_track_lookahead_m
        if lookahead <= 0.0:
            return target

        error = distance(self.pos_world, target)
        if error <= lookahead or error < 1.0e-6:
            return target

        scale = lookahead / error
        return (
            self.pos_world[0] + (target[0] - self.pos_world[0]) * scale,
            self.pos_world[1] + (target[1] - self.pos_world[1]) * scale,
            self.pos_world[2] + (target[2] - self.pos_world[2]) * scale,
        )

    def clear_active_plan_after_finish(self, clear_target=False):
        self.plan = []
        self.plan_waypoints = []
        self.active_waypoint_index = 0
        self.active_waypoint_start = None
        self.trajectory = None
        self.plan_start = None
        self.plan_total_time = 0.0
        self.filtered_vx_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.last_cmd_time = None
        self.next_gate_candidate_world = None
        self.next_gate_candidate_count = 0
        self.next_gate_candidate_stamp = None
        if clear_target:
            self.target_front_world = None
            self.target_gate_world = None
            self.active_gate_world = None
            self.aim_gate_world = None
            self.target_exit_world = None
            self.target_forward_dir_world = None

    def extend_exit_if_gate_still_ahead(self, now, final_hold):
        if not self.require_gate_clear_before_finish:
            return False
        if self.gate_clear_extensions >= self.gate_clear_max_extensions:
            rospy.logwarn(
                "finish gate plan after max clear extensions: passed_distance=%.2f",
                self.current_gate_passed_distance()
                if self.current_gate_passed_distance() is not None
                else 0.0,
            )
            return False
        if not self.is_gate_still_ahead(now):
            return False

        forward_dir = self.target_forward_dir_world
        if forward_dir is None:
            forward_dir = (math.cos(self.yaw), math.sin(self.yaw), 0.0)
        extra_target = (
            self.pos_world[0] + forward_dir[0] * self.gate_clear_extra_m,
            self.pos_world[1] + forward_dir[1] * self.gate_clear_extra_m,
            final_hold[2],
        )
        self.plan_waypoints.append(extra_target)
        self.active_waypoint_index = len(self.plan_waypoints) - 1
        self.active_waypoint_start = now
        self.gate_clear_extensions += 1
        rospy.logwarn(
            "gate not cleared by world projection, extend se3 waypoint %d/%d to (%.2f, %.2f, %.2f), passed_distance=%.2f",
            self.gate_clear_extensions,
            self.gate_clear_max_extensions,
            extra_target[0],
            extra_target[1],
            extra_target[2],
            self.current_gate_passed_distance()
            if self.current_gate_passed_distance() is not None
            else 0.0,
        )
        return True

    def is_gate_still_ahead(self, now):
        _ = now
        passed_distance = self.current_gate_passed_distance()
        if passed_distance is None:
            return False
        return passed_distance <= self.effective_clear_distance_m()

    def publish_takeoff_hold(self, now):
        target = self.takeoff_target_world
        yaw_rate = 0.0
        yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate)
        self.publish_trajectory_cmd(
            target,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )
        rospy.loginfo_throttle(
            1.0,
            "se3 takeoff hold: target=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f)",
            target[0],
            target[1],
            target[2],
            self.pos_world[0],
            self.pos_world[1],
            self.pos_world[2],
        )

    def publish_search_or_hover(self, now):
        self.filtered_vx_cmd = 0.0
        self.filtered_vz_cmd = 0.0
        self.last_cmd_time = None
        if self.last_gate_stamp is None:
            stale = True
        else:
            stale = (now - self.last_gate_stamp).to_sec() > self.gate_timeout

        if stale and self.enable_search:
            if self.command_mode == "trajectory":
                self.publish_hold_trajectory(now, self.search_yaw_rate)
            else:
                self.publish_cmd(0.0, 0.0, 0.0, self.search_yaw_rate)
            rospy.loginfo_throttle(1.0, "waiting for gate, searching")
        else:
            self.publish_stop_or_hold(now)
            rospy.loginfo_throttle(1.0, "waiting for gate")

    def publish_search_next_gate(self, now):
        yaw_rate = self.search_yaw_rate if self.enable_search else 0.0
        forward_step = self.search_next_forward_step_m
        search_speed = self.search_next_speed_mps
        if self.state == ADVANCE_SEARCH_NEXT_GATE:
            forward_step = max(forward_step, self.advance_search_lookahead_m)
            search_speed = max(search_speed, self.advance_search_speed_mps)

        if self.command_mode != "trajectory":
            vx = max(0.0, min(self.max_vx, search_speed))
            self.publish_cmd(vx, 0.0, 0.0, yaw_rate)
            rospy.loginfo_throttle(
                1.0,
                "searching next gate: vx=%.2f yawRate=%.2f",
                vx,
                yaw_rate,
            )
            return

        if self.is_search_next_settling(now):
            self.publish_search_next_settle(now)
            return

        forward_dir = self.search_forward_dir_world
        if forward_dir is None:
            forward_dir = self.horizontal_forward_dir(self.target_forward_dir_world)
            self.search_forward_dir_world = forward_dir
        forward_dir = self.steer_search_forward_dir_from_visual(now, forward_dir)
        yaw_rate = (
            self.compute_search_yaw_rate(now, self.trajectory_search_yaw_rate)
            if self.enable_search
            else 0.0
        )
        if self.search_start_world is None:
            self.search_start_world = self.pos_world
        self.update_search_next_altitude_target(now)

        search_delta = (
            self.pos_world[0] - self.search_start_world[0],
            self.pos_world[1] - self.search_start_world[1],
            0.0,
        )
        search_distance = max(0.0, dot(search_delta, forward_dir))
        if (
            self.search_next_max_distance_m > 0.0
            and search_distance >= self.search_next_max_distance_m
        ):
            hold_target = (
                self.pos_world[0],
                self.pos_world[1],
                self.search_next_altitude_world,
            )
            yaw, yaw_rate_rad = self.update_yaw_reference(now, 0.0)
            self.publish_trajectory_cmd(
                hold_target,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                yaw,
                yaw_rate_rad,
            )
            rospy.logwarn_throttle(
                1.0,
                "search next max distance reached, holding: distance=%.2f limit=%.2f",
                search_distance,
                self.search_next_max_distance_m,
            )
            return

        target_stale = False
        if self.search_next_target_stamp is None:
            target_stale = True
        elif self.gate_reacquire_timeout > 0.0:
            target_stale = (
                now - self.search_next_target_stamp
            ).to_sec() > self.gate_reacquire_timeout

        target_passed = False
        if self.search_next_target_world is not None:
            target_delta = (
                self.pos_world[0] - self.search_next_target_world[0],
                self.pos_world[1] - self.search_next_target_world[1],
                0.0,
            )
            target_passed = dot(target_delta, forward_dir) > -0.1

        if (
            self.search_next_target_world is None
            or distance(self.pos_world, self.search_next_target_world) < 0.25
            or target_passed
            or target_stale
        ):
            target_distance = search_distance + forward_step
            if self.search_next_max_distance_m > 0.0:
                target_distance = min(target_distance, self.search_next_max_distance_m)
            self.search_next_target_world = (
                self.search_start_world[0] + forward_dir[0] * target_distance,
                self.search_start_world[1] + forward_dir[1] * target_distance,
                self.search_next_altitude_world,
            )
            self.search_next_target_stamp = now
            rospy.loginfo(
                "search next gate target=(%.2f, %.2f, %.2f) forward_dir=(%.2f, %.2f) distance=%.2f",
                self.search_next_target_world[0],
                self.search_next_target_world[1],
                self.search_next_target_world[2],
                forward_dir[0],
                forward_dir[1],
                target_distance,
            )

        if self.trajectory_use_feedforward:
            velocity = (
                forward_dir[0] * search_speed,
                forward_dir[1] * search_speed,
                0.0,
            )
            velocity = self.limit_vector_norm(velocity, self.max_speed_mps)
        else:
            velocity = (0.0, 0.0, 0.0)

        target_world = self.search_next_target_world
        if target_world is None:
            rospy.logwarn_throttle(
                0.5,
                "skip search next command because target is not ready",
            )
            return

        yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate)
        self.publish_trajectory_cmd(
            target_world,
            velocity,
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )
        rospy.loginfo_throttle(
            1.0,
            "state: %s target=(%.2f, %.2f, %.2f) search_distance=%.2f speed=%.2f step=%.2f",
            self.state,
            target_world[0],
            target_world[1],
            target_world[2],
            search_distance,
            search_speed,
            forward_step,
        )

    def active_recovery_altitude(self, now):
        raw_planned_z = None
        if self.plan and self.plan_start is not None and self.plan_total_time > 0.0:
            elapsed = (now - self.plan_start).to_sec()
            elapsed = max(0.0, min(self.plan_total_time, elapsed))
            if self.trajectory is not None:
                raw_planned_z = self.trajectory.eval(elapsed, 0)[2]
            else:
                target = self.plan[-1][1]
                raw_planned_z = target[2]
        elif self.search_next_altitude_world is not None:
            raw_planned_z = self.search_next_altitude_world
        elif self.target_gate_world is not None:
            raw_planned_z = self.pos_world[2]
        else:
            raw_planned_z = self.pos_world[2]

        return raw_planned_z

    def replan_after_heading_recovery_if_needed(self, now):
        if self.command_mode != "trajectory" or not self.plan:
            rospy.loginfo(
                "heading recovery exit: replan_after_recovery=False reason=no active trajectory"
            )
            return False
        if self.target_gate_world is None:
            rospy.loginfo(
                "heading recovery exit: replan_after_recovery=False reason=no active gate"
            )
            return False
        if self.plan_start is None or self.plan_total_time <= 0.0:
            rospy.loginfo(
                "heading recovery exit: replan_after_recovery=False reason=no trajectory time"
            )
            return False

        elapsed = (now - self.plan_start).to_sec()
        elapsed = max(0.0, min(self.plan_total_time, elapsed))
        if self.trajectory is not None:
            sampled_pos = self.trajectory.eval(elapsed, 0)
        else:
            sampled_pos, _sampled_vel, _sampled_acc = self.sample_quintic_plan(elapsed)

        axes = self.build_path_frame_axes(self.current_planning_forward_dir())
        error = world_to_frame(self.pos_world, sampled_pos, axes)
        forward_error = abs(error[0])
        lateral_error = abs(error[1])
        vertical_error = abs(error[2])
        expected_yaw = self.heading_recovery_expected_yaw()
        yaw_error_deg = abs(math.degrees(wrap_angle(self.yaw - expected_yaw)))
        should_replan = (
            forward_error > 1.5
            or lateral_error > 0.8
            or vertical_error > 0.5
            or yaw_error_deg > 20.0
        )
        if not should_replan:
            rospy.loginfo(
                "heading recovery exit: current_vs_traj=(forward=%.2f lateral=%.2f vertical=%.2f yaw=%.1f) replan_after_recovery=False",
                forward_error,
                lateral_error,
                vertical_error,
                yaw_error_deg,
            )
            return False

        forward_dir = self.current_planning_forward_dir()
        gate_progress = dot(
            (
                self.target_gate_world[0] - self.pos_world[0],
                self.target_gate_world[1] - self.pos_world[1],
                self.target_gate_world[2] - self.pos_world[2],
            ),
            forward_dir,
        )
        front_world = self.target_front_world
        if front_world is None:
            front_progress = -1.0
        else:
            front_progress = dot(
                (
                    front_world[0] - self.pos_world[0],
                    front_world[1] - self.pos_world[1],
                    front_world[2] - self.pos_world[2],
                ),
                forward_dir,
            )
        if front_progress < 0.5 and gate_progress > 0.5:
            recovery_front_progress = min(
                max(0.5, self.front_offset_m),
                max(0.5, gate_progress - 0.5),
            )
            ratio = min(1.0, recovery_front_progress / max(1.0e-6, gate_progress))
            front_world = (
                self.pos_world[0] + forward_dir[0] * recovery_front_progress,
                self.pos_world[1] + forward_dir[1] * recovery_front_progress,
                self.pos_world[2]
                + ratio * (self.target_gate_world[2] - self.pos_world[2]),
            )
        if front_world is None:
            rospy.logwarn(
                "heading recovery exit: current_vs_traj=(forward=%.2f lateral=%.2f vertical=%.2f) replan_after_recovery=False reason=no valid front waypoint",
                forward_error,
                lateral_error,
                vertical_error,
            )
            return False

        if not self.is_gate_plan_candidate_valid(
            front_world,
            self.target_gate_world,
            self.target_exit_world,
            forward_dir,
            context="replan",
        ):
            rospy.logwarn(
                "heading recovery exit: current_vs_traj=(forward=%.2f lateral=%.2f vertical=%.2f) replan_after_recovery=False reason=invalid active gate",
                forward_error,
                lateral_error,
                vertical_error,
            )
            return False

        replanned = self.build_plan_from_waypoints(
            front_world,
            self.target_gate_world,
            self.target_exit_world,
            now,
            is_next_gate=(self.state == TRACK_NEXT_GATE),
        )
        rospy.logwarn(
            "heading recovery exit: current_vs_traj=(forward=%.2f lateral=%.2f vertical=%.2f yaw=%.1f) sampled=(%.2f, %.2f, %.2f) current=(%.2f, %.2f, %.2f) recovery_entry_z=%s replan_after_recovery=%s",
            forward_error,
            lateral_error,
            vertical_error,
            yaw_error_deg,
            sampled_pos[0],
            sampled_pos[1],
            sampled_pos[2],
            self.pos_world[0],
            self.pos_world[1],
            self.pos_world[2],
            "none" if self.heading_recovery_entry_z is None else "%.2f" % self.heading_recovery_entry_z,
            str(bool(replanned)),
        )
        return bool(replanned)

    def publish_heading_recovery(self, now):
        needed, expected_yaw, yaw_error, roll_deg, pitch_deg = (
            self.heading_recovery_status()
        )
        if not needed:
            return

        if not self.heading_recovery_active:
            self.yaw_reference = self.yaw
            self.last_yaw_time = now
            self.heading_recovery_active = True
            self.heading_recovery_last_stamp = now
            self.heading_recovery_entry_z = self.pos_world[2]
            self.heading_recovery_altitude_z = self.pos_world[2]

        recovery_dt = self.pause_active_plan_time_for_recovery(now)
        self.heading_recovery_last_stamp = now

        forward_dir = self.current_planning_forward_dir()
        target_dir, recovery_target_source, recovery_gate_yaw, recovery_path_yaw = (
            self.heading_recovery_target_direction()
        )
        abs_tilt_deg = max(abs(roll_deg), abs(pitch_deg))
        yaw_enabled, yaw_rate_limit_deg, yaw_rate_scale = (
            self.heading_recovery_yaw_ready(now, abs_tilt_deg)
        )
        raw_planned_z = self.active_recovery_altitude(now)
        target_z, target_vz, altitude_mode = self.heading_recovery_altitude_target(
            raw_planned_z
        )
        gate_z_ignored = (
            raw_planned_z is not None
            and target_z > self.clamp_height(raw_planned_z) + 1.0e-3
        )
        near_gate_zone = self.is_near_gate_zone()
        hold_xy_for_leveling = (
            self.heading_recovery_hold_xy_when_yaw_disabled
            and not yaw_enabled
            and near_gate_zone
        )
        target = (
            self.pos_world[0]
            if hold_xy_for_leveling
            else self.pos_world[0] + target_dir[0] * self.heading_recovery_forward_step_m,
            self.pos_world[1]
            if hold_xy_for_leveling
            else self.pos_world[1] + target_dir[1] * self.heading_recovery_forward_step_m,
            target_z,
        )

        recovery_speed = (
            self.heading_recovery_speed_mps
            if yaw_enabled
            else self.heading_recovery_level_speed_mps
        )
        if not near_gate_zone:
            recovery_speed = max(
                recovery_speed,
                self.heading_recovery_far_level_min_speed_mps,
            )
        if self.trajectory_use_feedforward:
            velocity = (
                0.0 if hold_xy_for_leveling else target_dir[0] * recovery_speed,
                0.0 if hold_xy_for_leveling else target_dir[1] * recovery_speed,
                target_vz,
            )
            velocity = self.limit_vector_norm(velocity, self.max_speed_mps)
        else:
            velocity = (0.0, 0.0, 0.0)

        if yaw_enabled:
            if self.yaw_reference is None:
                self.yaw_reference = self.yaw
            yaw_ref_error = wrap_angle(expected_yaw - self.yaw_reference)
            yaw_rate_deg = clamp(
                self.heading_recovery_yaw_gain * math.degrees(yaw_ref_error),
                yaw_rate_limit_deg,
            )
            yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate_deg)
        else:
            yaw_rate_deg = 0.0
            yaw = self.yaw
            yaw_rate_rad = 0.0
            self.yaw_reference = self.yaw
            self.last_yaw_time = now

        self.publish_trajectory_cmd(
            target,
            velocity,
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )
        if (
            recovery_gate_yaw is not None
            and abs(wrap_angle(expected_yaw - recovery_gate_yaw)) < 1.0e-3
        ):
            recovery_yaw_source = recovery_target_source
        else:
            recovery_yaw_source = "path"

        rospy.logwarn_throttle(
            0.3,
            "heading recovery: mode=%s altitude=%s target=(%.2f, %.2f, %.2f) recovery_entry_z=%s recovery_target_z=%.2f raw_planned_z=%.2f gate_z_ignored_in_recovery=%s replan_after_recovery=False vz=%.2f near_gate=%s hold_xy=%s forward_dir=(%.2f, %.2f) target_dir=(%.2f, %.2f) recovery_target_dir=%s recovery_gate_yaw=%s recovery_path_yaw=%.1f recovery_yaw_source=%s yaw=%.1f cmd_yaw=%.1f expected=%.1f yaw_error=%.1f yawRate=%.1f yawLimit=%.1f yawScale=%.2f roll=%.1f pitch=%.1f speed=%.2f",
            "yaw" if yaw_enabled else "level",
            altitude_mode,
            target[0],
            target[1],
            target[2],
            "none" if self.heading_recovery_entry_z is None else "%.2f" % self.heading_recovery_entry_z,
            target_z,
            raw_planned_z,
            str(gate_z_ignored),
            velocity[2],
            str(near_gate_zone),
            str(hold_xy_for_leveling),
            forward_dir[0],
            forward_dir[1],
            target_dir[0],
            target_dir[1],
            recovery_target_source,
            "none" if recovery_gate_yaw is None else "%.1f" % math.degrees(recovery_gate_yaw),
            math.degrees(recovery_path_yaw),
            recovery_yaw_source,
            math.degrees(self.yaw),
            math.degrees(yaw),
            math.degrees(expected_yaw),
            math.degrees(yaw_error),
            yaw_rate_deg,
            yaw_rate_limit_deg,
            yaw_rate_scale,
            roll_deg,
            pitch_deg,
            velocity[0] * target_dir[0] + velocity[1] * target_dir[1],
        )

    def publish_clear_gate_continuation(self, now):
        self.set_state(CLEAR_GATE)
        forward_dir = self.horizontal_forward_dir(self.target_forward_dir_world)
        target_z = (
            self.height_ref_world if self.command_mode == "trajectory" else self.pos_world[2]
        )
        target = (
            self.pos_world[0] + forward_dir[0] * self.search_next_forward_step_m,
            self.pos_world[1] + forward_dir[1] * self.search_next_forward_step_m,
            target_z,
        )
        yaw_rate = self.compute_yaw_rate(now) if self.enable_yaw_tracking else 0.0

        if self.command_mode == "trajectory":
            if self.trajectory_use_feedforward:
                velocity = (
                    forward_dir[0] * self.pass_speed_mps,
                    forward_dir[1] * self.pass_speed_mps,
                    0.0,
                )
                velocity = self.limit_vector_norm(velocity, self.max_speed_mps)
            else:
                velocity = (0.0, 0.0, 0.0)
            yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate)
            self.publish_trajectory_cmd(target, velocity, (0.0, 0.0, 0.0), yaw, yaw_rate_rad)
        else:
            vx = max(0.0, min(self.max_vx, self.pass_speed_mps))
            self.publish_cmd(vx, 0.0, 0.0, yaw_rate)

        passed_distance = self.current_gate_passed_distance()
        rospy.loginfo_throttle(
            0.5,
            "state: CLEAR_GATE continuing through gate target=(%.2f, %.2f, %.2f) passed_distance=%.2f clear_distance=%.2f",
            target[0],
            target[1],
            target[2],
            passed_distance if passed_distance is not None else 0.0,
            self.effective_clear_distance_m(),
        )

    def is_search_next_settling(self, now):
        if self.post_gate_settle_time <= 0.0 or self.search_next_enter_time is None:
            return False
        return (now - self.search_next_enter_time).to_sec() < self.post_gate_settle_time

    def publish_search_next_settle(self, now):
        if self.search_next_settle_target_world is None:
            self.search_next_settle_target_world = self.pos_world
        yaw, yaw_rate_rad = self.update_yaw_reference(now, 0.0)
        self.publish_trajectory_cmd(
            self.search_next_settle_target_world,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )
        remaining = self.post_gate_settle_time - (
            now - self.search_next_enter_time
        ).to_sec()
        rospy.loginfo_throttle(
            0.5,
            "state: %s optional settling target=(%.2f, %.2f, %.2f) remaining=%.2f",
            self.state,
            self.search_next_settle_target_world[0],
            self.search_next_settle_target_world[1],
            self.search_next_settle_target_world[2],
            max(0.0, remaining),
        )

    def compute_yaw_rate(self, now):
        if self.command_mode == "trajectory" and self.trajectory_lock_yaw_to_path:
            (
                target_yaw,
                gate_center_yaw,
                path_yaw,
                yaw_source,
                center_weight,
            ) = self.tracking_target_yaw()
            reference_yaw = self.yaw_reference if self.yaw_reference is not None else self.yaw
            yaw_error = wrap_angle(target_yaw - reference_yaw)
            yaw_rate = clamp(math.degrees(self.k_yaw * yaw_error), self.max_yaw_rate)
            if abs(yaw_rate) > 0.5:
                rospy.loginfo_throttle(
                    0.5,
                    "path yaw: gate_center_yaw=%.1f path_yaw=%.1f blended_target_yaw=%.1f ref=%.1f error=%.1f rate=%.1f yaw_source=%s center_weight=%.2f",
                    math.degrees(gate_center_yaw),
                    math.degrees(path_yaw),
                    math.degrees(target_yaw),
                    math.degrees(reference_yaw),
                    math.degrees(yaw_error),
                    yaw_rate,
                    yaw_source,
                    center_weight,
                )
            return yaw_rate

        if self.last_gate_rel is None or self.last_gate_stamp is None:
            return 0.0

        if (now - self.last_gate_stamp).to_sec() > self.gate_timeout:
            return 0.0

        yaw_error = math.atan2(self.last_gate_rel[1], max(0.2, self.last_gate_rel[0]))
        raw_yaw_rate = math.degrees(self.yaw_tracking_sign * self.k_yaw * yaw_error)
        yaw_rate = clamp(raw_yaw_rate, self.max_yaw_rate)
        if abs(yaw_rate) > 0.5:
            rospy.loginfo_throttle(
                0.5,
                "track yaw: rel=(%.2f, %.2f) yaw_error=%.1f sign=%.1f rate=%.1f",
                self.last_gate_rel[0],
                self.last_gate_rel[1],
                math.degrees(yaw_error),
                self.yaw_tracking_sign,
                yaw_rate,
            )
        return yaw_rate

    def update_yaw_reference(self, now, yaw_rate_deg):
        if self.yaw_reference is None:
            self.yaw_reference = self.yaw

        if self.last_yaw_time is None:
            dt = 1.0 / max(1.0, self.control_rate)
        else:
            dt = max(0.0, min(0.2, (now - self.last_yaw_time).to_sec()))

        yaw_rate_rad = math.radians(yaw_rate_deg)
        self.yaw_reference += yaw_rate_rad * dt
        self.last_yaw_time = now
        return self.yaw_reference, yaw_rate_rad

    def filter_forward_cmd(self, vx, now):
        if abs(vx) < self.vx_deadband:
            vx = 0.0

        alpha = max(0.0, min(1.0, self.vx_filter_alpha))
        target_vx = alpha * vx + (1.0 - alpha) * self.filtered_vx_cmd

        if self.last_cmd_time is None:
            dt = 1.0 / max(1.0, self.control_rate)
        else:
            dt = max(0.0, (now - self.last_cmd_time).to_sec())

        if self.max_vx_step_per_sec > 0.0 and dt > 0.0:
            max_step = self.max_vx_step_per_sec * dt
            target_vx = clamp(
                target_vx,
                self.filtered_vx_cmd - max_step,
                self.filtered_vx_cmd + max_step,
            )

        self.filtered_vx_cmd = target_vx
        self.last_cmd_time = now

        if vx == 0.0 and abs(self.filtered_vx_cmd) < self.vx_deadband:
            self.filtered_vx_cmd = 0.0

        return self.filtered_vx_cmd

    def filter_vertical_cmd(self, vz):
        if abs(vz) < self.vz_deadband:
            vz = 0.0

        alpha = max(0.0, min(1.0, self.vz_filter_alpha))
        self.filtered_vz_cmd = alpha * vz + (1.0 - alpha) * self.filtered_vz_cmd

        if vz == 0.0 and abs(self.filtered_vz_cmd) < self.vz_deadband:
            self.filtered_vz_cmd = 0.0

        return self.filtered_vz_cmd

    def relative_to_world(self, rel):
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return (
            self.pos_world[0] + c * rel[0] - s * rel[1],
            self.pos_world[1] + s * rel[0] + c * rel[1],
            self.pos_world[2] + self.gate_relative_z_sign * rel[2],
        )

    def world_vector_to_body(self, vec):
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return (
            c * vec[0] + s * vec[1],
            -s * vec[0] + c * vec[1],
            vec[2],
        )

    def publish_cmd(self, vx, vy, vz, yaw_rate):
        cmd = VelCmd()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.child_frame_id
        cmd.vx = vx
        cmd.vy = vy
        cmd.vz = vz
        cmd.yawRate = yaw_rate
        cmd.va = self.accel_cmd
        cmd.stop = 0
        self.cmd_pub.publish(cmd)
        self.last_command_position = None
        self.last_command_velocity = None
        self.last_command_acceleration = None
        self.last_command_yaw = self.yaw
        self.last_command_yaw_rate = yaw_rate
        self.publish_planner_debug(cmd.header.stamp)

    def debug_value(self, value):
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    def optional_degrees(self, value):
        if value is None:
            return float("nan")
        return math.degrees(value)

    def yaw_diagnostics_snapshot(
        self,
        now=None,
        position=None,
        velocity=None,
        command_yaw=None,
        command_yaw_rate=None,
    ):
        if now is None:
            now = rospy.Time.now()
        if position is None:
            position = self.last_command_position
        if velocity is None:
            velocity = self.last_command_velocity
        if command_yaw is None:
            command_yaw = self.last_command_yaw
        if command_yaw_rate is None:
            command_yaw_rate = self.last_command_yaw_rate

        (
            target_yaw,
            gate_center_yaw,
            path_yaw,
            yaw_source,
            center_weight,
        ) = self.tracking_target_yaw()
        gate_bearing_yaw, gate_bearing_dist = self.active_gate_bearing_yaw()
        near_gate_zone = self.is_near_gate_zone()

        cmd_vs_odom = (
            None if command_yaw is None else wrap_angle(command_yaw - self.yaw)
        )
        target_vs_cmd = (
            None if command_yaw is None else wrap_angle(target_yaw - command_yaw)
        )
        target_vs_odom = wrap_angle(target_yaw - self.yaw)
        path_vs_cmd = (
            None if command_yaw is None else wrap_angle(path_yaw - command_yaw)
        )
        gate_vs_cmd = (
            None
            if command_yaw is None or gate_bearing_yaw is None
            else wrap_angle(gate_bearing_yaw - command_yaw)
        )

        command_bearing_yaw = None
        command_bearing_dist = None
        if position is not None:
            dx = position[0] - self.pos_world[0]
            dy = position[1] - self.pos_world[1]
            command_bearing_dist = math.sqrt(dx * dx + dy * dy)
            if command_bearing_dist > 1.0e-3:
                command_bearing_yaw = math.atan2(dy, dx)

        velocity_yaw = None
        horizontal_speed = None
        velocity_vs_cmd = None
        velocity_vs_gate = None
        if velocity is not None:
            horizontal_speed = math.sqrt(
                velocity[0] * velocity[0] + velocity[1] * velocity[1]
            )
            if horizontal_speed > 1.0e-3:
                velocity_yaw = math.atan2(velocity[1], velocity[0])
                if command_yaw is not None:
                    velocity_vs_cmd = wrap_angle(velocity_yaw - command_yaw)
                if gate_bearing_yaw is not None:
                    velocity_vs_gate = wrap_angle(velocity_yaw - gate_bearing_yaw)

        rel_age = None
        if self.last_gate_stamp is not None:
            rel_age = max(0.0, (now - self.last_gate_stamp).to_sec())

        if command_yaw_rate is None:
            cmd_rate_deg_s = None
        elif self.command_mode == "trajectory":
            cmd_rate_deg_s = math.degrees(command_yaw_rate)
        else:
            cmd_rate_deg_s = command_yaw_rate
        gate_rel = self.last_gate_rel
        return {
            "yaw_source": yaw_source,
            "near_gate_zone": near_gate_zone,
            "odom_yaw": self.yaw,
            "cmd_yaw": command_yaw,
            "target_yaw": target_yaw,
            "path_yaw": path_yaw,
            "gate_center_yaw": gate_center_yaw,
            "gate_bearing_yaw": gate_bearing_yaw,
            "gate_bearing_dist": gate_bearing_dist,
            "center_weight": center_weight,
            "cmd_vs_odom_yaw_error": cmd_vs_odom,
            "target_vs_cmd_yaw_error": target_vs_cmd,
            "target_vs_odom_yaw_error": target_vs_odom,
            "path_vs_cmd_yaw_error": path_vs_cmd,
            "gate_vs_cmd_yaw_error": gate_vs_cmd,
            "cmd_yaw_rate_deg_s": cmd_rate_deg_s,
            "command_bearing_yaw": command_bearing_yaw,
            "command_bearing_dist": command_bearing_dist,
            "velocity_yaw": velocity_yaw,
            "horizontal_speed": horizontal_speed,
            "velocity_vs_cmd_yaw_error": velocity_vs_cmd,
            "velocity_vs_gate_yaw_error": velocity_vs_gate,
            "last_gate_rel_x": None if gate_rel is None else gate_rel[0],
            "last_gate_rel_y": None if gate_rel is None else gate_rel[1],
            "last_gate_rel_z": None if gate_rel is None else gate_rel[2],
            "last_gate_rel_age": rel_age,
        }

    def log_yaw_diagnostics(
        self,
        now,
        position=None,
        velocity=None,
        command_yaw=None,
        command_yaw_rate=None,
    ):
        if not self.yaw_debug_log_enabled:
            return
        diag = self.yaw_diagnostics_snapshot(
            now, position, velocity, command_yaw, command_yaw_rate
        )
        rospy.loginfo_throttle(
            max(0.05, self.yaw_debug_log_period),
            "yaw diag: state=%s source=%s near=%s odom=%.1f cmd=%.1f target=%.1f path=%.1f gate_center=%.1f gate_bearing=%.1f err_cmd_odom=%.1f err_target_cmd=%.1f err_target_odom=%.1f yaw_rate=%.1f vel_yaw=%.1f speed=%.2f vel_gate_err=%.1f cmd_bearing=%.1f cmd_dist=%.2f gate_dist=%.2f rel=(%s,%s,%s) rel_age=%s",
            self.state,
            diag["yaw_source"],
            str(diag["near_gate_zone"]),
            self.optional_degrees(diag["odom_yaw"]),
            self.optional_degrees(diag["cmd_yaw"]),
            self.optional_degrees(diag["target_yaw"]),
            self.optional_degrees(diag["path_yaw"]),
            self.optional_degrees(diag["gate_center_yaw"]),
            self.optional_degrees(diag["gate_bearing_yaw"]),
            self.optional_degrees(diag["cmd_vs_odom_yaw_error"]),
            self.optional_degrees(diag["target_vs_cmd_yaw_error"]),
            self.optional_degrees(diag["target_vs_odom_yaw_error"]),
            float("nan")
            if diag["cmd_yaw_rate_deg_s"] is None
            else diag["cmd_yaw_rate_deg_s"],
            self.optional_degrees(diag["velocity_yaw"]),
            float("nan") if diag["horizontal_speed"] is None else diag["horizontal_speed"],
            self.optional_degrees(diag["velocity_vs_gate_yaw_error"]),
            self.optional_degrees(diag["command_bearing_yaw"]),
            float("nan")
            if diag["command_bearing_dist"] is None
            else diag["command_bearing_dist"],
            float("nan")
            if diag["gate_bearing_dist"] is None
            else diag["gate_bearing_dist"],
            "nan"
            if diag["last_gate_rel_x"] is None
            else "%.2f" % diag["last_gate_rel_x"],
            "nan"
            if diag["last_gate_rel_y"] is None
            else "%.2f" % diag["last_gate_rel_y"],
            "nan"
            if diag["last_gate_rel_z"] is None
            else "%.2f" % diag["last_gate_rel_z"],
            "nan"
            if diag["last_gate_rel_age"] is None
            else "%.2f" % diag["last_gate_rel_age"],
        )

    def publish_planner_debug(self, now=None, clear_decision=None):
        if self.planner_debug_pub.get_num_connections() <= 0:
            return
        if now is None:
            now = rospy.Time.now()
        if clear_decision is None:
            clear_decision = self.last_clear_decision
            if self.target_gate_world is not None:
                clear_decision = self.current_gate_clear_decision()

        expected_yaw = self.heading_recovery_expected_yaw()
        yaw_error = wrap_angle(self.yaw - expected_yaw)
        command = self.last_command_position
        active_gate = self.active_gate_world
        aim_gate = self.current_aim_gate_world()
        near_gate_zone = self.is_near_gate_zone()
        yaw_diag = self.yaw_diagnostics_snapshot(now)
        data = {
            "stamp": now.to_sec(),
            "state": self.state,
            "active_gate_index": self.gate_sequence_index,
            "forward": None,
            "lateral": None,
            "vertical": None,
            "center_miss": None,
            "clear_allowed": False,
            "clear_reject_reason": "no active gate",
            "heading_recovery_active": self.heading_recovery_active,
            "near_gate_zone": near_gate_zone,
            "yaw": self.yaw,
            "expected_yaw": expected_yaw,
            "yaw_error": yaw_error,
            "target_yaw": self.last_command_yaw,
            "yaw_rate": self.last_command_yaw_rate,
            "yaw_source": yaw_diag["yaw_source"],
            "yaw_target_deg": self.optional_degrees(yaw_diag["target_yaw"]),
            "yaw_cmd_deg": self.optional_degrees(yaw_diag["cmd_yaw"]),
            "yaw_odom_deg": self.optional_degrees(yaw_diag["odom_yaw"]),
            "yaw_path_deg": self.optional_degrees(yaw_diag["path_yaw"]),
            "yaw_gate_center_deg": self.optional_degrees(
                yaw_diag["gate_center_yaw"]
            ),
            "yaw_gate_bearing_deg": self.optional_degrees(
                yaw_diag["gate_bearing_yaw"]
            ),
            "yaw_cmd_vs_odom_error_deg": self.optional_degrees(
                yaw_diag["cmd_vs_odom_yaw_error"]
            ),
            "yaw_target_vs_cmd_error_deg": self.optional_degrees(
                yaw_diag["target_vs_cmd_yaw_error"]
            ),
            "yaw_target_vs_odom_error_deg": self.optional_degrees(
                yaw_diag["target_vs_odom_yaw_error"]
            ),
            "yaw_path_vs_cmd_error_deg": self.optional_degrees(
                yaw_diag["path_vs_cmd_yaw_error"]
            ),
            "yaw_gate_vs_cmd_error_deg": self.optional_degrees(
                yaw_diag["gate_vs_cmd_yaw_error"]
            ),
            "yaw_rate_cmd_deg_s": yaw_diag["cmd_yaw_rate_deg_s"],
            "yaw_center_weight": yaw_diag["center_weight"],
            "yaw_command_bearing_deg": self.optional_degrees(
                yaw_diag["command_bearing_yaw"]
            ),
            "yaw_command_bearing_dist": yaw_diag["command_bearing_dist"],
            "yaw_velocity_deg": self.optional_degrees(yaw_diag["velocity_yaw"]),
            "yaw_velocity_vs_cmd_error_deg": self.optional_degrees(
                yaw_diag["velocity_vs_cmd_yaw_error"]
            ),
            "yaw_velocity_vs_gate_error_deg": self.optional_degrees(
                yaw_diag["velocity_vs_gate_yaw_error"]
            ),
            "yaw_horizontal_speed": yaw_diag["horizontal_speed"],
            "last_gate_rel_x": yaw_diag["last_gate_rel_x"],
            "last_gate_rel_y": yaw_diag["last_gate_rel_y"],
            "last_gate_rel_z": yaw_diag["last_gate_rel_z"],
            "last_gate_rel_age": yaw_diag["last_gate_rel_age"],
            "cmd_x": None if command is None else command[0],
            "cmd_y": None if command is None else command[1],
            "cmd_z": None if command is None else command[2],
            "odom_x": self.pos_world[0],
            "odom_y": self.pos_world[1],
            "odom_z": self.pos_world[2],
            "active_gate_x": None if active_gate is None else active_gate[0],
            "active_gate_y": None if active_gate is None else active_gate[1],
            "active_gate_z": None if active_gate is None else active_gate[2],
            "aim_gate_x": None if aim_gate is None else aim_gate[0],
            "aim_gate_y": None if aim_gate is None else aim_gate[1],
            "aim_gate_z": None if aim_gate is None else aim_gate[2],
            "next_gate_candidate_x": None if self.next_gate_candidate_world is None else self.next_gate_candidate_world[0],
            "next_gate_candidate_y": None if self.next_gate_candidate_world is None else self.next_gate_candidate_world[1],
            "next_gate_candidate_z": None if self.next_gate_candidate_world is None else self.next_gate_candidate_world[2],
            "next_gate_candidate_count": self.next_gate_candidate_count,
            "missed_gate_count": len(self.missed_gate_world_positions),
        }
        if clear_decision is not None:
            data.update(
                {
                    "forward": clear_decision.get("forward"),
                    "lateral": clear_decision.get("lateral"),
                    "vertical": clear_decision.get("vertical"),
                    "center_miss": clear_decision.get("center_miss"),
                    "clear_allowed": clear_decision.get("allowed", False),
                    "clear_reject_reason": clear_decision.get("reason", "unknown"),
                }
            )
        data = {key: self.debug_value(value) for key, value in data.items()}
        msg = String()
        msg.data = json.dumps(data, sort_keys=True, separators=(",", ":"))
        self.planner_debug_pub.publish(msg)

    def publish_stop_or_hold(self, now, hold_position=None):
        if self.command_mode == "trajectory":
            self.publish_hold_trajectory(now, 0.0, hold_position)
        else:
            self.publish_cmd(0.0, 0.0, 0.0, 0.0)

    def publish_hold_trajectory(self, now, yaw_rate_deg, hold_position=None):
        yaw, yaw_rate_rad = self.update_yaw_reference(now, yaw_rate_deg)
        if hold_position is None:
            hold_position = (
                self.hold_position_world
                if self.hold_position_world is not None
                else self.pos_world
            )
        self.publish_trajectory_cmd(
            hold_position,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            yaw,
            yaw_rate_rad,
        )

    def publish_trajectory_cmd(self, position, velocity, acceleration, yaw, yaw_rate):
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.joint_names = [self.child_frame_id]

        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = position[0]
        transform.translation.y = position[1]
        transform.translation.z = position[2]
        fill_quaternion_from_yaw(transform.rotation, yaw)

        vel = Twist()
        vel.linear.x = velocity[0]
        vel.linear.y = velocity[1]
        vel.linear.z = velocity[2]
        vel.angular.z = yaw_rate

        acc = Twist()
        acc.linear.x = acceleration[0]
        acc.linear.y = acceleration[1]
        acc.linear.z = acceleration[2]

        point.transforms.append(transform)
        point.velocities.append(vel)
        point.accelerations.append(acc)
        point.time_from_start = rospy.Duration(0.0)
        msg.points.append(point)
        self.cmd_pub.publish(msg)
        self.last_command_position = position
        self.last_command_velocity = velocity
        self.last_command_acceleration = acceleration
        self.last_command_yaw = yaw
        self.last_command_yaw_rate = yaw_rate
        self.log_yaw_diagnostics(msg.header.stamp, position, velocity, yaw, yaw_rate)
        self.publish_planner_debug(msg.header.stamp)


if __name__ == "__main__":
    node = GatePolyPlanner()
    rospy.spin()
