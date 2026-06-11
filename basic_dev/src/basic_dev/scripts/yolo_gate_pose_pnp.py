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


def extract_gate_corners_from_bbox(image, bbox):
    _ = image
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
    fx = width / (2.0 * math.tan(0.5 * hfov))
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


class YoloGatePosePnp:
    def __init__(self):
        rospy.init_node("yolo_gate_pose_pnp")

        model_path_param = rospy.get_param("~model_path", "")
        self.model_path = model_path_param if model_path_param else default_model_path()
        self.image_topic = rospy.get_param(
            "~image_topic", "/airsim_node/drone_1/front_left/Scene"
        )
        self.pose_topic = rospy.get_param("~pose_topic", "/gate/relative_pose")
        self.visual_error_topic = rospy.get_param(
            "~visual_error_topic", "/gate/visual_error"
        )
        self.frame_id = rospy.get_param("~frame_id", "drone_1/body")

        self.conf = rospy.get_param("~conf", 0.25)
        self.imgsz = rospy.get_param("~imgsz", 640)
        self.process_every_n_frames = rospy.get_param("~process_every_n_frames", 2)

        self.camera_hfov_deg = rospy.get_param("~camera_hfov_deg", 60.0)
        self.gate_width_m = rospy.get_param("~gate_width_m", 10.0)
        self.gate_height_m = rospy.get_param("~gate_height_m", 5.0)
        self.min_depth_m = rospy.get_param("~min_depth_m", 0.25)
        self.max_depth_m = rospy.get_param("~max_depth_m", 80.0)
        self.min_area_ratio = rospy.get_param("~min_area_ratio", 0.0)
        self.visual_error_min_area_ratio = rospy.get_param(
            "~visual_error_min_area_ratio", 0.005
        )
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

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.inference_device = select_inference_device()

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.visual_error_pub = rospy.Publisher(
            self.visual_error_topic, Vector3Stamped, queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("yolo_gate_pose_pnp started.")
        rospy.loginfo("image_topic: %s", self.image_topic)
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("visual_error_topic: %s", self.visual_error_topic)
        rospy.loginfo("model: %s", self.model_path)
        rospy.loginfo("camera_hfov_deg: %.1f", self.camera_hfov_deg)
        rospy.loginfo(
            "gate_width_m=%.2f gate_height_m=%.2f min_area_ratio=%.3f visual_error_min_area_ratio=%.3f edge_margin_px=%.1f",
            self.gate_width_m,
            self.gate_height_m,
            self.min_area_ratio,
            self.visual_error_min_area_ratio,
            self.edge_margin_px,
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

    def image_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "image convert failed: %s", str(exc))
            return

        h, w = img.shape[:2]
        results = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            device=self.inference_device,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            rospy.loginfo_throttle(1.0, "no gate detected")
            return

        best = self.select_best_box(boxes)
        if best is None:
            return

        x1, y1, x2, y2, conf = best
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_ratio = area / float(w * h)
        visual_error = self.publish_visual_error_from_bbox(
            msg.header, w, h, (x1, y1, x2, y2), area_ratio
        )
        if area_ratio < self.min_area_ratio:
            rospy.loginfo_throttle(
                0.5,
                "reject gate detection: area %.3f < %.3f conf=%.2f",
                area_ratio,
                self.min_area_ratio,
                conf,
            )
            return

        if self.edge_margin_px > 0.0 and (
            x1 < self.edge_margin_px
            or y1 < self.edge_margin_px
            or x2 > w - self.edge_margin_px
            or y2 > h - self.edge_margin_px
            ):
            rospy.loginfo_throttle(
                0.5,
                "reject gate detection near image edge: bbox=(%.0f,%.0f,%.0f,%.0f) conf=%.2f visual_hint=%s",
                x1,
                y1,
                x2,
                y2,
                conf,
                "true" if visual_error is not None else "false",
            )
            return

        image_points = extract_gate_corners_from_bbox(img, (x1, y1, x2, y2))
        if not self.valid_corners(image_points, w, h):
            rospy.logwarn_throttle(0.5, "invalid gate corners")
            return

        camera_matrix = build_camera_matrix(w, h, self.camera_hfov_deg)
        success, _rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            rospy.logwarn_throttle(0.5, "solvePnP failed")
            return

        rel_x, rel_y, rel_z = camera_to_body(
            tvec,
            body_y_sign=self.body_y_sign,
            body_z_sign=self.body_z_sign,
        )
        if rel_x < self.min_depth_m or rel_x > self.max_depth_m:
            rospy.logwarn_throttle(
                0.5,
                "reject pnp depth %.2f outside [%.2f, %.2f]",
                rel_x,
                self.min_depth_m,
                self.max_depth_m,
            )
            return

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        if pose.header.stamp == rospy.Time(0):
            pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = rel_x
        pose.pose.position.y = rel_y
        pose.pose.position.z = rel_z
        pose.pose.orientation.w = 1.0

        self.pose_pub.publish(pose)

        if visual_error is None:
            visual_error = (0.0, 0.0, area_ratio)

        rospy.loginfo_throttle(
            0.5,
            "gate pnp rel xyz=(%.2f, %.2f, %.2f) visual_err=(%.2f, %.2f) bbox=(%.0f,%.0f,%.0f,%.0f) conf=%.2f",
            rel_x,
            rel_y,
            rel_z,
            visual_error[0],
            visual_error[1],
            x1,
            y1,
            x2,
            y2,
            conf,
        )

    def select_best_box(self, boxes):
        best = None
        best_area = 0.0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (float(x1), float(y1), float(x2), float(y2), conf)
        return best

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


if __name__ == "__main__":
    node = YoloGatePosePnp()
    rospy.spin()
