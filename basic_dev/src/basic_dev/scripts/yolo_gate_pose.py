#!/usr/bin/env python3
import math
import os

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from ultralytics import YOLO


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


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


class YoloGatePose:
    def __init__(self):
        rospy.init_node("yolo_gate_pose")

        model_path_param = rospy.get_param("~model_path", "")
        self.model_path = model_path_param if model_path_param else default_model_path()
        self.image_topic = rospy.get_param(
            "~image_topic", "/airsim_node/drone_1/front_left/Scene"
        )
        self.pose_topic = rospy.get_param("~pose_topic", "/gate/relative_pose")
        self.frame_id = rospy.get_param("~frame_id", "drone_1/body")

        self.conf = rospy.get_param("~conf", 0.25)
        self.imgsz = rospy.get_param("~imgsz", 640)
        self.process_every_n_frames = rospy.get_param("~process_every_n_frames", 2)

        self.camera_hfov_deg = rospy.get_param("~camera_hfov_deg", 60.0)
        self.gate_width_m = rospy.get_param("~gate_width_m", 10.0)
        self.gate_height_m = rospy.get_param("~gate_height_m", 5.0)
        self.depth_scale = rospy.get_param("~depth_scale", 1.0)
        self.min_depth_m = rospy.get_param("~min_depth_m", 0.25)
        self.max_depth_m = rospy.get_param("~max_depth_m", 30.0)
        self.min_area_ratio = rospy.get_param("~min_area_ratio", 0.0)
        self.edge_margin_px = rospy.get_param("~edge_margin_px", 0.0)

        base_filter_alpha = rospy.get_param("~filter_alpha", 0.35)
        self.filter_alpha_x = rospy.get_param("~filter_alpha_x", base_filter_alpha)
        self.filter_alpha_y = rospy.get_param("~filter_alpha_y", base_filter_alpha)
        self.filter_alpha_z = rospy.get_param("~filter_alpha_z", 0.12)
        self.x_deadband_m = rospy.get_param("~x_deadband_m", 0.05)
        self.max_x_step_m = rospy.get_param("~max_x_step_m", 0.30)
        self.z_deadband_m = rospy.get_param("~z_deadband_m", 0.10)
        self.max_z_step_m = rospy.get_param("~max_z_step_m", 0.25)
        self.filtered_pose = None
        self.frame_count = 0

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("yolo_gate_pose started.")
        rospy.loginfo("image_topic: %s", self.image_topic)
        rospy.loginfo("pose_topic: %s", self.pose_topic)
        rospy.loginfo("model: %s", self.model_path)
        rospy.loginfo("camera_hfov_deg: %.1f", self.camera_hfov_deg)
        rospy.loginfo(
            "bbox depth gate size: width=%.2f height=%.2f",
            self.gate_width_m,
            self.gate_height_m,
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
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            rospy.loginfo_throttle(1.0, "no gate detected")
            return

        best = None
        best_area = 0.0
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2, conf)

        if best is None:
            return

        x1, y1, x2, y2, conf = best
        area_ratio = best_area / float(w * h)
        if area_ratio < self.min_area_ratio:
            rospy.loginfo_throttle(
                0.5,
                "reject gate detection: area %.3f < %.3f conf=%.2f",
                area_ratio,
                self.min_area_ratio,
                conf,
            )
            return

        if self.edge_margin_px > 0.0:
            if (
                x1 < self.edge_margin_px
                or y1 < self.edge_margin_px
                or x2 > w - self.edge_margin_px
                or y2 > h - self.edge_margin_px
            ):
                rospy.loginfo_throttle(
                    0.5,
                    "reject gate detection near image edge: box=(%.0f, %.0f, %.0f, %.0f) conf=%.2f",
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                )
                return

        if best is None:
            return

        rel_x, rel_y, rel_z = self.estimate_relative_pose(w, h, best)
        rel_x, rel_y, rel_z = self.filter_pose(rel_x, rel_y, rel_z)

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

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        rospy.loginfo_throttle(
            0.5,
            "gate rel xyz=(%.2f, %.2f, %.2f) conf=%.2f center=(%.0f, %.0f) area=%.3f",
            rel_x,
            rel_y,
            rel_z,
            conf,
            cx,
            cy,
            area_ratio,
        )

    def estimate_relative_pose(self, width, height, detection):
        x1, y1, x2, y2, _ = detection
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        hfov = math.radians(self.camera_hfov_deg)
        fx = width / (2.0 * math.tan(0.5 * hfov))
        fy = fx

        depth_from_width = self.gate_width_m * fx / box_w
        depth_values = [depth_from_width]
        if self.gate_height_m > 0.0:
            depth_values.append(self.gate_height_m * fy / box_h)

        depth = self.depth_scale * sum(depth_values) / float(len(depth_values))
        depth = clamp(depth, self.min_depth_m, self.max_depth_m)

        rel_x = depth
        rel_y = (cx - 0.5 * width) * depth / fx
        rel_z = -(cy - 0.5 * height) * depth / fy

        return rel_x, rel_y, rel_z

    def filter_pose(self, rel_x, rel_y, rel_z):
        if self.filtered_pose is None:
            if abs(rel_z) < self.z_deadband_m:
                rel_z = 0.0
            self.filtered_pose = (rel_x, rel_y, rel_z)
            return self.filtered_pose

        alpha_x = clamp(self.filter_alpha_x, 0.0, 1.0)
        alpha_y = clamp(self.filter_alpha_y, 0.0, 1.0)
        alpha_z = clamp(self.filter_alpha_z, 0.0, 1.0)
        old_x, old_y, old_z = self.filtered_pose

        if abs(rel_x - old_x) < self.x_deadband_m:
            rel_x = old_x

        if self.max_x_step_m > 0.0:
            rel_x = clamp(rel_x, old_x - self.max_x_step_m, old_x + self.max_x_step_m)

        if abs(rel_z) < self.z_deadband_m:
            rel_z = 0.0

        if self.max_z_step_m > 0.0:
            rel_z = clamp(rel_z, old_z - self.max_z_step_m, old_z + self.max_z_step_m)

        filtered_z = alpha_z * rel_z + (1.0 - alpha_z) * old_z
        if rel_z == 0.0 and abs(filtered_z) < self.z_deadband_m:
            filtered_z = 0.0

        self.filtered_pose = (
            alpha_x * rel_x + (1.0 - alpha_x) * old_x,
            alpha_y * rel_y + (1.0 - alpha_y) * old_y,
            filtered_z,
        )
        return self.filtered_pose


if __name__ == "__main__":
    node = YoloGatePose()
    rospy.spin()
