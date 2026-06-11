#!/usr/bin/env python3
import math
import os

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import Image
from ultralytics import YOLO


def default_model_path():
    candidates = [
        os.environ.get("GATE_MODEL_PATH"),
        "/basic_dev/models/gate_v1_full.pt",
        os.path.abspath(os.path.join(os.getcwd(), "models/gate_v1_full.pt")),
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../models/gate_v1_full.pt")
        ),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return candidates[1]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def select_inference_device():
    if not torch.cuda.is_available():
        rospy.logwarn("CUDA unavailable, using CPU")
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        gpu_arch = f"sm_{major}{minor}"
        supported_arches = torch.cuda.get_arch_list()

        rospy.loginfo(
            "CUDA device=%s capability=%s supported_arches=%s",
            torch.cuda.get_device_name(0),
            gpu_arch,
            supported_arches,
        )

        if gpu_arch not in supported_arches:
            rospy.logwarn(
                "Current PyTorch does not support %s; falling back to CPU",
                gpu_arch,
            )
            return "cpu"

        rospy.loginfo("YOLO using CUDA")
        return 0

    except Exception as exc:
        rospy.logwarn("CUDA compatibility check failed: %s; using CPU", exc)
        return "cpu"


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def camera_to_body(tvec, body_y_sign=1.0, body_z_sign=1.0):
    cam_x = float(tvec[0][0])  # right
    cam_y = float(tvec[1][0])  # down
    cam_z = float(tvec[2][0])  # forward

    rel_x = cam_z
    rel_y = body_y_sign * cam_x
    rel_z = body_z_sign * (-cam_y)
    return rel_x, rel_y, rel_z


def build_camera_matrix(width, height, camera_hfov_deg):
    hfov = math.radians(camera_hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def extract_gate_corners_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def reorder_gate_keypoints(points):
    if points is None or np.asarray(points).shape != (4, 2):
        return None

    pts = np.asarray(points, dtype=np.float32)
    order_y = np.argsort(pts[:, 1])
    top = pts[order_y[:2]]
    bottom = pts[order_y[2:]]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    return np.array(
        [
            top[0],
            top[1],
            bottom[1],
            bottom[0],
        ],
        dtype=np.float32,
    )


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def segment_orientation(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(a, b, c, d):
    o1 = segment_orientation(a, b, c)
    o2 = segment_orientation(a, b, d)
    o3 = segment_orientation(c, d, a)
    o4 = segment_orientation(c, d, b)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0


def quadrilateral_self_intersects(points):
    pts = np.asarray(points, dtype=np.float32)
    return segments_intersect(pts[0], pts[1], pts[2], pts[3]) or segments_intersect(
        pts[1], pts[2], pts[3], pts[0]
    )


class YoloGatePoseKeypointPnp:
    def __init__(self):
        rospy.init_node("yolo_gate_pose_keypoint_pnp")

        model_path_param = rospy.get_param("~model_path", "")
        self.model_path = model_path_param if model_path_param else default_model_path()
        self.image_topic = rospy.get_param(
            "~image_topic", "/airsim_node/drone_1/front_left/Scene"
        )
        self.pose_topic = rospy.get_param("~pose_topic", "/gate/relative_pose")
        self.visual_error_topic = rospy.get_param(
            "~visual_error_topic", "/gate/visual_error"
        )
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", "/gate/keypoint_pnp_debug"
        )
        self.frame_id = rospy.get_param("~frame_id", "drone_1/body")

        self.conf = rospy.get_param("~conf", 0.35)
        self.keypoint_conf = rospy.get_param("~keypoint_conf", 0.45)
        self.imgsz = rospy.get_param("~imgsz", 640)
        self.process_every_n_frames = rospy.get_param("~process_every_n_frames", 2)

        self.camera_hfov_deg = rospy.get_param("~camera_hfov_deg", 60.0)
        self.gate_width_m = rospy.get_param("~gate_width_m", 10.0)
        self.gate_height_m = rospy.get_param("~gate_height_m", 5.0)
        self.min_area_ratio = rospy.get_param("~min_area_ratio", 0.02)
        self.max_area_ratio = rospy.get_param("~max_area_ratio", 0.55)
        self.max_center_error = rospy.get_param("~max_center_error", 0.75)
        self.visual_error_min_area_ratio = rospy.get_param(
            "~visual_error_min_area_ratio", 0.005
        )
        self.max_reproj_error_px = rospy.get_param("~max_reproj_error_px", 8.0)
        self.min_depth_m = rospy.get_param("~min_depth_m", 1.0)
        self.max_depth_m = rospy.get_param("~max_depth_m", 45.0)
        self.max_lateral_m = rospy.get_param("~max_lateral_m", 4.8)
        self.max_vertical_m = rospy.get_param("~max_vertical_m", 8.0)
        self.max_position_jump_m = rospy.get_param("~max_position_jump_m", 2.0)
        self.filter_alpha = clamp(rospy.get_param("~filter_alpha", 0.35), 0.0, 1.0)
        self.fallback_to_bbox_pnp = rospy.get_param("~fallback_to_bbox_pnp", True)
        self.publish_debug_image = rospy.get_param("~publish_debug_image", True)
        self.hold_last_valid_pose = rospy.get_param("~hold_last_valid_pose", True)
        self.hold_last_valid_timeout_s = rospy.get_param(
            "~hold_last_valid_timeout_s", 1.0
        )
        self.hold_visual_error_zero = rospy.get_param("~hold_visual_error_zero", True)
        self.publish_invalid_debug = rospy.get_param("~publish_invalid_debug", True)
        self.debug_reject_reasons = rospy.get_param("~debug_reject_reasons", True)
        self.candidate_reproj_weight = rospy.get_param(
            "~candidate_reproj_weight", 2.0
        )
        self.candidate_center_weight = rospy.get_param(
            "~candidate_center_weight", 1.0
        )
        self.candidate_depth_jump_weight = rospy.get_param(
            "~candidate_depth_jump_weight", 1.5
        )
        self.candidate_pose_jump_weight = rospy.get_param(
            "~candidate_pose_jump_weight", 2.0
        )
        self.require_all_keypoints = rospy.get_param("~require_all_keypoints", True)
        self.selected_debug_only = rospy.get_param("~selected_debug_only", True)
        self.edge_margin_px = rospy.get_param("~edge_margin_px", 0.0)
        self.body_y_sign = rospy.get_param("~body_y_sign", 1.0)
        self.body_z_sign = rospy.get_param("~body_z_sign", 1.0)

        self.object_points = np.array(
            [
                [-self.gate_width_m / 2.0, -self.gate_height_m / 2.0, 0.0],
                [self.gate_width_m / 2.0, -self.gate_height_m / 2.0, 0.0],
                [self.gate_width_m / 2.0, self.gate_height_m / 2.0, 0.0],
                [-self.gate_width_m / 2.0, self.gate_height_m / 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.frame_count = 0
        self.last_raw_position = None
        self.last_valid_pose = None
        self.last_valid_pose_time = None
        self.last_area_ratio = 0.0
        self.filtered_position = None

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.inference_device = select_inference_device()

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.visual_error_pub = rospy.Publisher(
            self.visual_error_topic, Vector3Stamped, queue_size=1
        )
        self.debug_image_pub = None
        if self.publish_debug_image:
            self.debug_image_pub = rospy.Publisher(
                self.debug_image_topic, Image, queue_size=1
            )
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("yolo_gate_pose_keypoint_pnp started.")
        rospy.loginfo("image_topic: %s", self.image_topic)
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("visual_error_topic: %s", self.visual_error_topic)
        rospy.loginfo("debug_image_topic: %s", self.debug_image_topic)
        rospy.loginfo("model: %s", self.model_path)
        rospy.loginfo("camera_hfov_deg: %.1f", self.camera_hfov_deg)
        rospy.loginfo(
            "gate_width_m=%.2f gate_height_m=%.2f area_ratio=[%.3f, %.3f] max_center_error=%.2f keypoint_conf=%.2f max_reproj_error_px=%.1f",
            self.gate_width_m,
            self.gate_height_m,
            self.min_area_ratio,
            self.max_area_ratio,
            self.max_center_error,
            self.keypoint_conf,
            self.max_reproj_error_px,
        )
        rospy.loginfo(
            "depth_range=[%.2f, %.2f] rel_plausibility lateral<=%.2f vertical<=%.2f max_position_jump_m=%.2f filter_alpha=%.2f fallback_to_bbox_pnp=%s",
            self.min_depth_m,
            self.max_depth_m,
            self.max_lateral_m,
            self.max_vertical_m,
            self.max_position_jump_m,
            self.filter_alpha,
            "true" if self.fallback_to_bbox_pnp else "false",
        )
        rospy.loginfo(
            "candidate weights: reproj=%.2f center=%.2f depth_jump=%.2f pose_jump=%.2f require_all_keypoints=%s selected_debug_only=%s",
            self.candidate_reproj_weight,
            self.candidate_center_weight,
            self.candidate_depth_jump_weight,
            self.candidate_pose_jump_weight,
            "true" if self.require_all_keypoints else "false",
            "true" if self.selected_debug_only else "false",
        )
        rospy.loginfo(
            "hold last valid: enabled=%s timeout=%.2f visual_error_zero=%s publish_invalid_debug=%s debug_reject_reasons=%s",
            "true" if self.hold_last_valid_pose else "false",
            self.hold_last_valid_timeout_s,
            "true" if self.hold_visual_error_zero else "false",
            "true" if self.publish_invalid_debug else "false",
            "true" if self.debug_reject_reasons else "false",
        )

    def image_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "image convert failed: %s", str(exc))
            return

        img_height, img_width = img.shape[:2]
        width = msg.width if msg.width > 0 else img_width
        height = msg.height if msg.height > 0 else img_height
        debug_img = img.copy() if self.publish_debug_image else None

        results = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            device=self.inference_device,
        )
        result = results[0]
        now = rospy.Time.now()

        detections = self.extract_detections(result)
        if not detections:
            rospy.loginfo_throttle(1.0, "no gate detected")
            if self.publish_hold_last_valid(
                msg,
                debug_img,
                candidates=[],
                status="no valid candidate",
                now=now,
            ):
                return
            if self.publish_invalid_debug and debug_img is not None:
                self.publish_candidate_debug(
                    msg,
                    debug_img,
                    candidates=[],
                    selected=None,
                    status="no valid candidate",
                )
            return

        camera_matrix = build_camera_matrix(width, height, self.camera_hfov_deg)
        candidates = [
            self.evaluate_candidate(detection, result, width, height, camera_matrix)
            for detection in detections
        ]
        valid_candidates = [candidate for candidate in candidates if candidate["valid"]]

        if not valid_candidates:
            if self.debug_reject_reasons:
                rospy.logwarn_throttle(
                    0.5,
                    "no valid gate candidate: %s",
                    " | ".join(
                        self.format_candidate_warn(candidate)
                        for candidate in candidates
                    ),
                )
            else:
                rospy.logwarn_throttle(
                    0.5,
                    "no valid gate candidate: %s",
                    self.reject_summary(candidates),
                )
            if self.publish_hold_last_valid(
                msg,
                debug_img,
                candidates=candidates,
                status="no valid candidate",
                now=now,
            ):
                return
            if self.publish_invalid_debug:
                self.publish_candidate_debug(
                    msg,
                    debug_img,
                    candidates,
                    selected=None,
                    status="no valid candidate",
                )
            return

        selected = min(valid_candidates, key=lambda candidate: candidate["score"])
        estimate = selected["estimate"]
        raw_position = selected["position"]
        bbox = selected["bbox"]
        x1, y1, x2, y2 = bbox
        bbox_conf = selected["conf"]
        area_ratio = selected["area_ratio"]

        filtered_position = self.apply_position_filter(raw_position)
        self.last_raw_position = raw_position
        self.last_valid_pose = filtered_position
        self.last_valid_pose_time = now
        self.last_area_ratio = area_ratio

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        if pose.header.stamp == rospy.Time(0):
            pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = filtered_position[0]
        pose.pose.position.y = filtered_position[1]
        pose.pose.position.z = filtered_position[2]
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        visual_error = self.publish_visual_error_from_bbox(
            msg.header,
            width,
            height,
            bbox,
            area_ratio,
        )
        if visual_error is None:
            visual_error = (0.0, 0.0, area_ratio)

        rospy.loginfo_throttle(
            0.5,
            "gate selected %s score=%.2f rel xyz=(%.2f, %.2f, %.2f) raw=(%.2f, %.2f, %.2f) reproj=%.2f center=%.2f area=%.3f visual_err=(%.2f, %.2f) bbox=(%.0f,%.0f,%.0f,%.0f) conf=%.2f candidates=%d",
            estimate["method"],
            selected["score"],
            filtered_position[0],
            filtered_position[1],
            filtered_position[2],
            raw_position[0],
            raw_position[1],
            raw_position[2],
            estimate["reproj_error"],
            selected["center_error"],
            area_ratio,
            visual_error[0],
            visual_error[1],
            x1,
            y1,
            x2,
            y2,
            bbox_conf,
            len(valid_candidates),
        )

        self.publish_candidate_debug(
            msg,
            debug_img,
            candidates,
            selected=selected,
            status="published",
        )

    def publish_visual_error_from_bbox(self, image_header, width, height, bbox, area_ratio):
        if area_ratio < self.visual_error_min_area_ratio:
            return None

        x1, y1, x2, y2 = bbox
        bbox_cx = 0.5 * (x1 + x2)
        bbox_cy = 0.5 * (y1 + y2)
        visual_error = Vector3Stamped()
        visual_error.header.stamp = image_header.stamp
        if visual_error.header.stamp == rospy.Time(0):
            visual_error.header.stamp = rospy.Time.now()
        visual_error.header.frame_id = image_header.frame_id
        visual_error.vector.x = clamp(
            (bbox_cx - 0.5 * width) / max(1.0, 0.5 * width), -1.0, 1.0
        )
        visual_error.vector.y = clamp(
            (bbox_cy - 0.5 * height) / max(1.0, 0.5 * height), -1.0, 1.0
        )
        visual_error.vector.z = clamp(area_ratio, 0.0, 1.0)
        self.visual_error_pub.publish(visual_error)
        return (
            visual_error.vector.x,
            visual_error.vector.y,
            visual_error.vector.z,
        )

    def publish_hold_last_valid(self, msg, debug_img, candidates, status, now):
        if not self.can_hold_last_valid(now):
            return False

        hold_age = (now - self.last_valid_pose_time).to_sec()
        rel_x, rel_y, rel_z = self.last_valid_pose

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = rel_x
        pose.pose.position.y = rel_y
        pose.pose.position.z = rel_z
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        if self.hold_visual_error_zero:
            visual_error = Vector3Stamped()
            visual_error.header.stamp = now
            visual_error.header.frame_id = msg.header.frame_id
            visual_error.vector.x = 0.0
            visual_error.vector.y = 0.0
            visual_error.vector.z = clamp(self.last_area_ratio, 0.0, 1.0)
            self.visual_error_pub.publish(visual_error)

        rospy.loginfo_throttle(
            0.5,
            "hold last valid gate pose age=%.2f rel xyz=(%.2f, %.2f, %.2f) area=%.3f reason=%s",
            hold_age,
            rel_x,
            rel_y,
            rel_z,
            self.last_area_ratio,
            status,
        )
        self.publish_hold_debug(
            msg,
            debug_img,
            candidates,
            hold_age,
            status,
        )
        return True

    def can_hold_last_valid(self, now):
        if not self.hold_last_valid_pose:
            return False
        if self.last_valid_pose is None or self.last_valid_pose_time is None:
            return False
        if not self.position_plausible(self.last_valid_pose):
            return False
        hold_age = (now - self.last_valid_pose_time).to_sec()
        return 0.0 <= hold_age <= self.hold_last_valid_timeout_s

    def publish_hold_debug(self, msg, debug_img, candidates, hold_age, status):
        if not self.publish_debug_image or self.debug_image_pub is None:
            return
        if debug_img is None:
            return

        for candidate in candidates:
            x1, y1, x2, y2 = candidate["bbox"]
            cv2.rectangle(
                debug_img,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (150, 150, 150),
                1,
            )
            if self.debug_reject_reasons:
                self.draw_candidate_label(debug_img, candidate, (150, 150, 150))

        last_depth = self.last_valid_pose[0] if self.last_valid_pose is not None else 0.0
        lines = [
            "HOLD LAST VALID",
            "hold_age=%.2f timeout=%.2f" % (hold_age, self.hold_last_valid_timeout_s),
            "last_depth=%.2f area=%.3f" % (last_depth, self.last_area_ratio),
            status,
        ]
        if candidates and self.debug_reject_reasons:
            lines.append("reject summary: %s" % self.reject_summary(candidates))
        self.draw_debug_lines(debug_img, lines)
        self.publish_debug_image_msg(msg, debug_img)

    def reject_summary(self, candidates):
        if not candidates:
            return "none"
        counts = {}
        for candidate in candidates:
            reason = candidate["reject_reason"] if not candidate["valid"] else "ok"
            if not reason:
                reason = "ok"
            counts[reason] = counts.get(reason, 0) + 1
        return ", ".join(
            "%s=%d" % (reason, count) for reason, count in sorted(counts.items())
        )

    def format_candidate_warn(self, candidate):
        reason = candidate["reject_reason"] if not candidate["valid"] else "ok"
        if candidate["reject_detail"]:
            reason = "%s (%s)" % (reason, candidate["reject_detail"])
        parts = [
            "idx=%d" % candidate["index"],
            "conf=%.2f" % candidate["conf"],
            "area=%.3f" % candidate["area_ratio"],
            "center=%.2f" % candidate["center_error"],
            "has_kp=%s" % ("true" if candidate["has_keypoints"] else "false"),
            "kp_count=%d" % candidate["keypoint_count"],
            "kp=%s" % self.format_keypoint_conf_list(candidate["keypoint_conf_list"]),
            "kp_min=%s" % self.format_optional(candidate["keypoint_conf_min"], "%.2f"),
            "reason=%s" % reason,
        ]
        if candidate["depth"] is not None:
            parts.append("depth=%.2f" % candidate["depth"])
        if candidate["position"] is not None:
            position = candidate["position"]
            parts.append(
                "rel=(%.2f,%.2f,%.2f)" % (position[0], position[1], position[2])
            )
        if candidate["reproj_error"] is not None:
            parts.append("reproj=%.2f" % candidate["reproj_error"])
        if candidate["score"] is not None:
            parts.append("score=%.2f" % candidate["score"])
        return " ".join(parts)

    def candidate_label_lines(self, candidate):
        reason = candidate["reject_reason"] if not candidate["valid"] else "ok"
        if candidate["reject_detail"]:
            reason = "%s %s" % (reason, candidate["reject_detail"])
        if candidate["score"] is not None and candidate["valid"]:
            reason = "score=%.2f" % candidate["score"]
        lines = [
            "idx=%d conf=%.2f area=%.3f"
            % (
                candidate["index"],
                candidate["conf"],
                candidate["area_ratio"],
            ),
            "center=%.2f has_kp=%s kp_count=%d"
            % (
                candidate["center_error"],
                "true" if candidate["has_keypoints"] else "false",
                candidate["keypoint_count"],
            ),
            "kp_min=%s kp=%s"
            % (
                self.format_optional(candidate["keypoint_conf_min"], "%.2f"),
                self.format_keypoint_conf_list(candidate["keypoint_conf_list"]),
            ),
            reason,
        ]
        if candidate["depth"] is not None or candidate["reproj_error"] is not None:
            lines.append(
                "d=%s r=%s"
                % (
                    self.format_optional(candidate["depth"], "%.2f"),
                    self.format_optional(candidate["reproj_error"], "%.2f"),
                )
            )
        if candidate["position"] is not None:
            position = candidate["position"]
            lines.append(
                "rel=(%.1f,%.1f,%.1f)"
                % (position[0], position[1], position[2])
            )
        return lines

    def format_optional(self, value, fmt):
        if value is None:
            return "n/a"
        return fmt % value

    def format_keypoint_conf_list(self, values):
        if not values:
            return "[]"
        return "[" + ",".join("%.2f" % value for value in values) + "]"

    def extract_detections(self, result):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = to_numpy(boxes.xyxy)
        confs = to_numpy(boxes.conf)
        if xyxy is None or len(xyxy) == 0:
            return []
        if confs is None:
            confs = np.ones((len(xyxy),), dtype=np.float32)

        detections = []
        for index, coords in enumerate(xyxy):
            x1, y1, x2, y2 = coords[:4]
            detections.append(
                {
                    "index": index,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "conf": float(confs[index]),
                }
            )
        return detections

    def evaluate_candidate(self, detection, result, width, height, camera_matrix):
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        bbox_w = max(0.0, x2 - x1)
        bbox_h = max(0.0, y2 - y1)
        area_ratio = (bbox_w * bbox_h) / float(width * height)
        norm_cx, norm_cy, center_error = self.compute_center_error(
            bbox,
            width,
            height,
        )
        keypoint_debug = self.inspect_detection_keypoints(result, detection["index"])
        keypoints = keypoint_debug["keypoints"]
        keypoints_complete = self.has_complete_keypoints(keypoints)
        keypoint_conf_min = keypoint_debug["keypoint_conf_min"]
        keypoint_conf_list = keypoint_debug["keypoint_conf_list"]
        ordered_keypoints = None
        if keypoints_complete:
            ordered_keypoints = reorder_gate_keypoints(keypoints["points"])
        quad_area, quad_aspect_ratio = self.quad_debug_metrics(ordered_keypoints)

        candidate = {
            "index": detection["index"],
            "bbox": bbox,
            "conf": detection["conf"],
            "area_ratio": area_ratio,
            "center_error": center_error,
            "norm_center": (norm_cx, norm_cy),
            "keypoints": ordered_keypoints,
            "has_keypoints": keypoint_debug["has_keypoints"],
            "keypoint_count": keypoint_debug["keypoint_count"],
            "keypoint_conf_min": keypoint_conf_min,
            "keypoint_conf_list": keypoint_conf_list,
            "keypoint_status": keypoint_debug["status"],
            "quad_area": quad_area,
            "quad_aspect_ratio": quad_aspect_ratio,
            "depth": None,
            "reproj_error": None,
            "estimate": None,
            "position": None,
            "score": None,
            "valid": False,
            "reason": "ok",
            "reject_reason": "",
            "reject_detail": "",
            "depth_jump": 0.0,
            "pose_jump": 0.0,
        }

        if area_ratio < self.min_area_ratio:
            return self.reject_candidate(
                candidate,
                "reject: area too small",
                "%.3f < %.3f" % (area_ratio, self.min_area_ratio),
            )
        if area_ratio > self.max_area_ratio:
            return self.reject_candidate(
                candidate,
                "reject: area too large",
                "%.3f > %.3f" % (area_ratio, self.max_area_ratio),
            )
        if center_error > self.max_center_error:
            return self.reject_candidate(
                candidate,
                "reject: center error too large",
                "%.2f > %.2f" % (center_error, self.max_center_error),
            )
        if self.bbox_near_edge(bbox, width, height) and not keypoints_complete:
            return self.reject_candidate(
                candidate,
                self.incomplete_keypoint_reject_reason(candidate),
                "edge bbox with incomplete keypoints",
            )

        if not keypoints_complete:
            if self.require_all_keypoints:
                return self.reject_candidate(
                    candidate,
                    self.incomplete_keypoint_reject_reason(candidate),
                )
            if not self.fallback_to_bbox_pnp:
                return self.reject_candidate(
                    candidate,
                    self.incomplete_keypoint_reject_reason(candidate),
                    "fallback disabled",
                )
            estimate, reason = self.estimate_bbox_pose(bbox, width, height, camera_matrix)
            if estimate is None:
                return self.reject_candidate(candidate, "reject: pnp failed", reason)
            return self.finalize_candidate(candidate, estimate)

        if keypoint_conf_min is not None and keypoint_conf_min < self.keypoint_conf:
            return self.reject_candidate(
                candidate,
                "reject: low kp conf",
                "%s min=%.2f < %.2f"
                % (
                    self.format_keypoint_conf_list(keypoint_conf_list),
                    keypoint_conf_min,
                    self.keypoint_conf,
                ),
            )

        valid_quad, quad_reason = self.valid_gate_quadrilateral(
            ordered_keypoints,
            width,
            height,
        )
        if not valid_quad:
            return self.reject_candidate(candidate, quad_reason)

        estimate, reason = self.solve_pose(
            ordered_keypoints,
            camera_matrix,
            method="keypoint_pnp",
            prefer_ippe=True,
        )
        if estimate is None:
            if self.fallback_to_bbox_pnp:
                estimate, reason = self.estimate_bbox_pose(
                    bbox,
                    width,
                    height,
                    camera_matrix,
                )
            if estimate is None:
                return self.reject_candidate(candidate, "reject: pnp failed", reason)

        self.attach_estimate_debug(candidate, estimate)
        if estimate["reproj_error"] > self.max_reproj_error_px:
            return self.reject_candidate(
                candidate,
                "reject: reproj error",
                "%.2f > %.2f"
                % (estimate["reproj_error"], self.max_reproj_error_px),
            )

        return self.finalize_candidate(candidate, estimate)

    def reject_candidate(self, candidate, reason, detail=""):
        candidate["valid"] = False
        candidate["reason"] = reason
        candidate["reject_reason"] = reason
        candidate["reject_detail"] = detail
        return candidate

    def finalize_candidate(self, candidate, estimate):
        self.attach_estimate_debug(candidate, estimate)
        position = estimate["position"]
        if estimate["reproj_error"] > self.max_reproj_error_px:
            return self.reject_candidate(
                candidate,
                "reject: reproj error",
                "%.2f > %.2f"
                % (estimate["reproj_error"], self.max_reproj_error_px),
            )
        if not self.valid_depth(position[0]):
            return self.reject_candidate(
                candidate,
                "reject: depth range",
                "%.2f outside [%.2f, %.2f]"
                % (position[0], self.min_depth_m, self.max_depth_m),
            )
        if not self.position_plausible(position):
            return self.reject_candidate(
                candidate,
                "reject: rel plausibility",
                self.position_plausibility_reason(position),
            )

        depth_jump, pose_jump = self.compute_pose_jumps(position)
        candidate["estimate"] = estimate
        candidate["position"] = position
        candidate["depth_jump"] = depth_jump
        candidate["pose_jump"] = pose_jump
        candidate["score"] = self.compute_candidate_score(candidate)
        candidate["valid"] = True
        candidate["reason"] = "ok"
        candidate["reject_reason"] = ""
        candidate["reject_detail"] = ""
        return candidate

    def attach_estimate_debug(self, candidate, estimate):
        candidate["estimate"] = estimate
        candidate["position"] = estimate["position"]
        candidate["depth"] = estimate["position"][0]
        candidate["reproj_error"] = estimate["reproj_error"]

    def compute_center_error(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        bbox_cx = 0.5 * (x1 + x2)
        bbox_cy = 0.5 * (y1 + y2)
        norm_cx = clamp(
            (bbox_cx - 0.5 * width) / max(1.0, 0.5 * width),
            -1.0,
            1.0,
        )
        norm_cy = clamp(
            (bbox_cy - 0.5 * height) / max(1.0, 0.5 * height),
            -1.0,
            1.0,
        )
        return norm_cx, norm_cy, math.sqrt(norm_cx * norm_cx + norm_cy * norm_cy)

    def compute_pose_jumps(self, position):
        if self.last_raw_position is None or self.max_position_jump_m <= 0.0:
            return 0.0, 0.0
        current = np.asarray(position, dtype=np.float64)
        last = np.asarray(self.last_raw_position, dtype=np.float64)
        delta = current - last
        depth_jump = abs(float(delta[0]))
        pose_jump = float(np.linalg.norm(delta))
        return depth_jump, pose_jump

    def compute_candidate_score(self, candidate):
        estimate = candidate["estimate"]
        reproj_norm = estimate["reproj_error"] / max(1.0e-6, self.max_reproj_error_px)
        center_norm = candidate["center_error"] / max(1.0e-6, self.max_center_error)
        jump_scale = max(1.0e-6, self.max_position_jump_m)
        depth_jump_norm = candidate["depth_jump"] / jump_scale
        pose_jump_norm = candidate["pose_jump"] / jump_scale
        return (
            self.candidate_reproj_weight * reproj_norm
            + self.candidate_center_weight * center_norm
            + self.candidate_depth_jump_weight * depth_jump_norm
            + self.candidate_pose_jump_weight * pose_jump_norm
            - 0.5 * candidate["conf"]
        )

    def bbox_near_edge(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        margin = max(0.0, float(self.edge_margin_px))
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= width - margin
            or y2 >= height - margin
        )

    def has_complete_keypoints(self, keypoints):
        if keypoints is None:
            return False
        points = keypoints.get("points")
        if points is None or np.asarray(points).shape != (4, 2):
            return False
        if not np.isfinite(points).all():
            return False
        conf = keypoints.get("conf")
        if conf is None or np.asarray(conf).shape[0] < 4:
            return False
        if not np.isfinite(conf[:4]).all():
            return False
        return True

    def inspect_detection_keypoints(self, result, index):
        info = {
            "keypoints": None,
            "has_keypoints": False,
            "keypoint_count": 0,
            "keypoint_conf_min": None,
            "keypoint_conf_list": [],
            "status": "no keypoints",
        }

        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return info

        points = to_numpy(getattr(keypoints, "xy", None))
        if points is None or index >= len(points):
            return info

        point_array = points[index]
        info["keypoint_count"] = int(point_array.shape[0])
        info["has_keypoints"] = info["keypoint_count"] > 0

        confs = to_numpy(getattr(keypoints, "conf", None))
        if confs is None:
            data = to_numpy(getattr(keypoints, "data", None))
            if data is not None and data.ndim == 3 and data.shape[2] >= 3:
                confs = data[:, :, 2]
        if confs is not None and index < len(confs):
            conf = np.asarray(confs[index], dtype=np.float32)
            info["keypoint_conf_list"] = [
                float(value) for value in conf[: min(4, conf.shape[0])]
            ]

        if info["keypoint_count"] == 0:
            info["status"] = "no keypoints"
            return info
        if info["keypoint_count"] < 4:
            info["status"] = "insufficient keypoints"
            return info

        if len(info["keypoint_conf_list"]) >= 4:
            point_confs = np.asarray(info["keypoint_conf_list"][:4], dtype=np.float32)
        else:
            point_confs = np.ones((4,), dtype=np.float32)
            info["keypoint_conf_list"] = [float(value) for value in point_confs]

        info["keypoints"] = {
            "points": point_array[:4].astype(np.float32),
            "conf": point_confs,
        }
        info["keypoint_conf_min"] = float(np.min(point_confs))
        info["status"] = "ok"
        return info

    def incomplete_keypoint_reject_reason(self, candidate):
        if not candidate["has_keypoints"]:
            return "reject: no keypoints"
        if candidate["keypoint_count"] < 4:
            return "reject: insufficient keypoints"
        return "reject: missing keypoints"

    def min_keypoint_conf(self, keypoints):
        if keypoints is None or keypoints.get("conf") is None:
            return None
        conf = np.asarray(keypoints["conf"], dtype=np.float32)
        if conf.shape[0] < 4:
            return None
        return float(np.min(conf[:4]))

    def keypoint_conf_list(self, keypoints):
        if keypoints is None or keypoints.get("conf") is None:
            return []
        conf = np.asarray(keypoints["conf"], dtype=np.float32)
        return [float(value) for value in conf[:4]]

    def quad_debug_metrics(self, corners):
        if corners is None or np.asarray(corners).shape != (4, 2):
            return None, None
        if not np.isfinite(corners).all():
            return None, None

        area = polygon_area(corners)
        tl, tr, br, bl = np.asarray(corners, dtype=np.float32)
        top_w = float(np.linalg.norm(tr - tl))
        bottom_w = float(np.linalg.norm(br - bl))
        left_h = float(np.linalg.norm(bl - tl))
        right_h = float(np.linalg.norm(br - tr))
        avg_w = 0.5 * (top_w + bottom_w)
        avg_h = 0.5 * (left_h + right_h)
        if avg_h < 1.0e-6:
            return area, None
        return area, avg_w / avg_h

    def valid_gate_quadrilateral(self, corners, width, height):
        if not self.valid_corners(corners, width, height):
            return False, "reject: invalid quad"

        area = polygon_area(corners)
        min_area_px = max(16.0, width * height * self.min_area_ratio * 0.25)
        if area < min_area_px:
            return False, "reject: invalid quad"

        if quadrilateral_self_intersects(corners):
            return False, "reject: quad self intersect"

        contour = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour):
            return False, "reject: invalid quad"

        tl, tr, br, bl = np.asarray(corners, dtype=np.float32)
        top_w = float(np.linalg.norm(tr - tl))
        bottom_w = float(np.linalg.norm(br - bl))
        left_h = float(np.linalg.norm(bl - tl))
        right_h = float(np.linalg.norm(br - tr))
        avg_w = 0.5 * (top_w + bottom_w)
        avg_h = 0.5 * (left_h + right_h)
        if avg_w < 2.0 or avg_h < 2.0:
            return False, "reject: invalid quad"

        aspect = avg_w / max(1.0e-6, avg_h)
        if aspect < 0.25 or aspect > 8.0:
            return False, "reject: aspect ratio"

        return True, "ok"

    def extract_detection_keypoints(self, result, index):
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None

        points = to_numpy(getattr(keypoints, "xy", None))
        if points is None or index >= len(points) or points[index].shape[0] < 4:
            return None

        confs = to_numpy(getattr(keypoints, "conf", None))
        if confs is None:
            data = to_numpy(getattr(keypoints, "data", None))
            if data is not None and data.ndim == 3 and data.shape[2] >= 3:
                confs = data[:, :, 2]
        if confs is not None and index < len(confs):
            point_confs = confs[index][:4].astype(np.float32)
        else:
            point_confs = np.ones((4,), dtype=np.float32)

        return {
            "points": points[index][:4].astype(np.float32),
            "conf": point_confs,
        }

    def estimate_keypoint_pose(self, keypoints, width, height, camera_matrix):
        if keypoints is None:
            return None, "missing keypoints"

        point_confs = keypoints["conf"]
        if np.any(point_confs < self.keypoint_conf):
            return None, "low keypoint confidence"

        image_points = reorder_gate_keypoints(keypoints["points"])
        if not self.valid_corners(image_points, width, height):
            return None, "invalid keypoint corners"

        estimate, reason = self.solve_pose(
            image_points,
            camera_matrix,
            method="keypoint_pnp",
            prefer_ippe=True,
        )
        if estimate is None:
            return None, reason
        if estimate["reproj_error"] > self.max_reproj_error_px:
            return (
                None,
                "reproj %.2f > %.2f"
                % (estimate["reproj_error"], self.max_reproj_error_px),
            )
        return estimate, ""

    def estimate_bbox_pose(self, bbox, width, height, camera_matrix):
        image_points = extract_gate_corners_from_bbox(bbox)
        if not self.valid_corners(image_points, width, height):
            return None, "invalid bbox corners"
        return self.solve_pose(
            image_points,
            camera_matrix,
            method="bbox_pnp_fallback",
            prefer_ippe=False,
        )

    def solve_pose(self, image_points, camera_matrix, method, prefer_ippe):
        flags = [cv2.SOLVEPNP_ITERATIVE]
        if prefer_ippe and hasattr(cv2, "SOLVEPNP_IPPE"):
            flags.insert(0, cv2.SOLVEPNP_IPPE)

        last_error = ""
        for flag in flags:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    image_points,
                    camera_matrix,
                    self.dist_coeffs,
                    flags=flag,
                )
            except cv2.error as exc:
                last_error = str(exc)
                continue
            if not success:
                last_error = "solvePnP returned false"
                continue

            reproj_error = self.compute_reprojection_error(
                image_points,
                rvec,
                tvec,
                camera_matrix,
            )
            rel_x, rel_y, rel_z = camera_to_body(
                tvec,
                body_y_sign=self.body_y_sign,
                body_z_sign=self.body_z_sign,
            )
            return (
                {
                    "method": method,
                    "image_points": image_points,
                    "rvec": rvec,
                    "tvec": tvec,
                    "position": (rel_x, rel_y, rel_z),
                    "reproj_error": reproj_error,
                },
                "",
            )

        return None, "solvePnP failed: %s" % last_error

    def compute_reprojection_error(self, image_points, rvec, tvec, camera_matrix):
        projected, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        return float(np.mean(errors))

    def valid_corners(self, corners, width, height):
        if corners is None or corners.shape != (4, 2):
            return False
        if not np.isfinite(corners).all():
            return False

        x_values = corners[:, 0]
        y_values = corners[:, 1]
        if np.any(x_values < 0.0) or np.any(x_values > width):
            return False
        if np.any(y_values < 0.0) or np.any(y_values > height):
            return False

        box_w = float(np.max(x_values) - np.min(x_values))
        box_h = float(np.max(y_values) - np.min(y_values))
        return box_w >= 2.0 and box_h >= 2.0

    def valid_depth(self, depth):
        return self.min_depth_m <= depth <= self.max_depth_m

    def position_plausible(self, position):
        if position is None:
            return False
        rel_y = abs(float(position[1]))
        rel_z = abs(float(position[2]))
        if self.max_lateral_m > 0.0 and rel_y > self.max_lateral_m:
            return False
        if self.max_vertical_m > 0.0 and rel_z > self.max_vertical_m:
            return False
        return True

    def position_plausibility_reason(self, position):
        if position is None:
            return "missing position"
        rel_y = abs(float(position[1]))
        rel_z = abs(float(position[2]))
        if self.max_lateral_m > 0.0 and rel_y > self.max_lateral_m:
            return "abs(y) %.2f > %.2f" % (rel_y, self.max_lateral_m)
        if self.max_vertical_m > 0.0 and rel_z > self.max_vertical_m:
            return "abs(z) %.2f > %.2f" % (rel_z, self.max_vertical_m)
        return "ok"

    def position_jump(self, position):
        if self.last_raw_position is None:
            return 0.0
        delta = np.asarray(position, dtype=np.float64) - np.asarray(
            self.last_raw_position, dtype=np.float64
        )
        return float(np.linalg.norm(delta))

    def exceeds_position_jump(self, position):
        if self.last_raw_position is None or self.max_position_jump_m <= 0.0:
            return False
        return self.position_jump(position) > self.max_position_jump_m

    def apply_position_filter(self, position):
        current = np.asarray(position, dtype=np.float64)
        if self.filtered_position is None:
            filtered = current
        else:
            filtered = (
                self.filter_alpha * current
                + (1.0 - self.filter_alpha) * np.asarray(self.filtered_position)
            )
        self.filtered_position = (float(filtered[0]), float(filtered[1]), float(filtered[2]))
        return self.filtered_position

    def reject_pose(self, msg, debug_img, debug_info, estimate, reason):
        rospy.logwarn_throttle(0.5, "reject gate keypoint pnp: %s", reason)
        debug_info["status"] = "reject: %s" % reason
        if estimate is not None:
            debug_info["method"] = estimate["method"]
            if estimate["method"] == "keypoint_pnp" or debug_info["keypoints"] is None:
                debug_info["keypoints"] = estimate["image_points"]
            debug_info["reproj_error"] = estimate["reproj_error"]
            debug_info["position"] = estimate["position"]
        self.publish_debug(msg, debug_img, **debug_info)

    def publish_candidate_debug(
        self,
        msg,
        debug_img,
        candidates,
        selected=None,
        status="",
    ):
        if not self.publish_debug_image or self.debug_image_pub is None:
            return
        if debug_img is None:
            return

        selected_index = selected["index"] if selected is not None else None
        for candidate in candidates:
            is_selected = candidate["index"] == selected_index
            color = (0, 255, 255) if is_selected else (150, 150, 150)
            thickness = 3 if is_selected else 1
            x1, y1, x2, y2 = candidate["bbox"]
            cv2.rectangle(
                debug_img,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                thickness,
            )

            if is_selected:
                self.draw_keypoints(debug_img, candidate.get("keypoints"))
            if self.debug_reject_reasons or (
                not is_selected and not self.selected_debug_only
            ):
                self.draw_candidate_label(debug_img, candidate, color)

        lines = [status]
        if selected is not None:
            estimate = selected["estimate"]
            keypoint_conf_min = selected["keypoint_conf_min"]
            keypoint_conf_text = (
                "%.2f" % keypoint_conf_min
                if keypoint_conf_min is not None
                else "n/a"
            )
            lines = [
                "%s selected score=%.2f" % (status, selected["score"]),
                "conf=%.2f reproj=%.2f depth=%.2f"
                % (
                    selected["conf"],
                    estimate["reproj_error"],
                    selected["position"][0],
                ),
                "area=%.3f center=%.2f kpt_min=%s"
                % (
                    selected["area_ratio"],
                    selected["center_error"],
                    keypoint_conf_text,
                ),
                "jump depth=%.2f pose=%.2f method=%s"
                % (
                    selected["depth_jump"],
                    selected["pose_jump"],
                    estimate["method"],
                ),
            ]
        elif candidates:
            valid_count = len([candidate for candidate in candidates if candidate["valid"]])
            lines.append("valid=%d total=%d" % (valid_count, len(candidates)))
            if self.debug_reject_reasons:
                lines.append("reject summary: %s" % self.reject_summary(candidates))

        self.draw_debug_lines(debug_img, lines)
        self.publish_debug_image_msg(msg, debug_img)

    def draw_candidate_label(self, debug_img, candidate, color):
        x1, y1, _x2, y2 = candidate["bbox"]
        start_x = int(round(x1))
        start_y = max(14, int(round(y1)) - 36)
        if start_y < 18:
            start_y = min(debug_img.shape[0] - 6, int(round(y2)) + 14)
        for line_index, line in enumerate(self.candidate_label_lines(candidate)):
            cv2.putText(
                debug_img,
                line[:60],
                (start_x, start_y + 14 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_keypoints(self, debug_img, keypoints):
        if keypoints is None or np.asarray(keypoints).shape != (4, 2):
            return
        labels = ("TL", "TR", "BR", "BL")
        for label, point in zip(labels, keypoints):
            px, py = int(round(point[0])), int(round(point[1]))
            cv2.circle(debug_img, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(
                debug_img,
                label,
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def draw_debug_lines(self, debug_img, lines):
        for line_index, line in enumerate(lines):
            if not line:
                continue
            cv2.putText(
                debug_img,
                line,
                (10, 24 + 22 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    def publish_debug_image_msg(self, msg, debug_img):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "debug image publish failed: %s", str(exc))

    def publish_debug(
        self,
        msg,
        debug_img,
        bbox=None,
        keypoints=None,
        method="none",
        reproj_error=None,
        position=None,
        status="",
    ):
        if not self.publish_debug_image or self.debug_image_pub is None:
            return
        if debug_img is None:
            return

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                debug_img,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 255),
                2,
            )

        if keypoints is not None and np.asarray(keypoints).shape == (4, 2):
            labels = ("TL", "TR", "BR", "BL")
            for label, point in zip(labels, keypoints):
                px, py = int(round(point[0])), int(round(point[1]))
                cv2.circle(debug_img, (px, py), 4, (0, 255, 0), -1)
                cv2.putText(
                    debug_img,
                    label,
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        lines = [status]
        if position is not None:
            rel_x, rel_y, rel_z = position
            lines.append(
                "%s depth=%.2f rel=(%.2f, %.2f, %.2f)"
                % (method, rel_x, rel_x, rel_y, rel_z)
            )
        elif method:
            lines.append(method)
        if reproj_error is not None:
            lines.append("reproj=%.2f px" % reproj_error)

        for line_index, line in enumerate(lines):
            if not line:
                continue
            cv2.putText(
                debug_img,
                line,
                (10, 24 + 22 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "debug image publish failed: %s", str(exc))


if __name__ == "__main__":
    node = YoloGatePoseKeypointPnp()
    rospy.spin()
