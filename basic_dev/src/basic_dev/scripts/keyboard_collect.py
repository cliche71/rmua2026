#!/usr/bin/env python3
import os
import sys
import termios
import tty
import select

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from airsim_ros.msg import VelCmd
from airsim_ros.srv import Takeoff


class KeyboardCollect:
    def __init__(self):
        rospy.init_node("keyboard_collect")

        self.pub = rospy.Publisher(
            "/airsim_node/drone_1/vel_body_cmd",
            VelCmd,
            queue_size=1
        )

        self.bridge = CvBridge()
        self.latest_img = None
        self.save_count = 0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

        self.max_vxy = 0.8
        self.max_vz = 0.5
        self.max_yaw =25.0

        self.save_dir = "/basic_dev/debug"
        os.makedirs(self.save_dir, exist_ok=True)

        rospy.Subscriber(
            "/airsim_node/drone_1/front_left/Scene",
            Image,
            self.image_cb,
            queue_size=1
        )

        #self.takeoff()

    def takeoff(self):
        try:
            rospy.wait_for_service("/airsim_node/drone_1/takeoff", timeout=5.0)
            takeoff_client = rospy.ServiceProxy(
                "/airsim_node/drone_1/takeoff",
                Takeoff
            )
            resp = takeoff_client(1)
            rospy.loginfo("takeoff result: %s", resp.success)
        except Exception as e:
            rospy.logwarn("takeoff failed: %s", str(e))

    def image_cb(self, msg):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rospy.loginfo_throttle(
                1.0,
                "image ok: %dx%d",
                self.latest_img.shape[1],
                self.latest_img.shape[0]
            )
        except Exception as e:
            rospy.logwarn_throttle(1.0, "image convert failed: %s", str(e))

    def save_image(self):
        if self.latest_img is None:
            rospy.logwarn("no image yet")
            return

        filename = os.path.join(
            self.save_dir,
            "frame_%05d.jpg" % self.save_count
        )

        cv2.imwrite(filename, self.latest_img)
        rospy.loginfo("saved image: %s", filename)

        self.save_count += 1

    def publish_cmd(self):
        cmd = VelCmd()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "drone_1"

        cmd.vx = self.vx
        cmd.vy = self.vy
        cmd.vz = self.vz
        cmd.yawRate = self.yaw_rate

        cmd.va = 4
        cmd.stop = 0

        self.pub.publish(cmd)

    def stop_motion(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

    def handle_key(self, key):
        self.stop_motion()

        if key == "w":
            self.vx = self.max_vxy
        elif key == "s":
            self.vx = -self.max_vxy
        elif key == "a":
            self.vy = -self.max_vxy
        elif key == "d":
            self.vy = self.max_vxy
        elif key == "q":
            self.yaw_rate = -self.max_yaw
        elif key == "e":
            self.yaw_rate = self.max_yaw
        elif key == "r":
            self.vz = self.max_vz
        elif key == "f":
            self.vz = -self.max_vz
        elif key == "x":
            self.stop_motion()
        elif key == " ":
            self.save_image()
            self.stop_motion()

    def get_key(self, timeout=0.05):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

    def run(self):
        print("")
        print("Keyboard control:")
        print("  W/S: forward/back")
        print("  A/D: left/right")
        print("  R/F: up/down")
        print("  Q/E: yaw left/right")
        print("  SPACE: save image")
        print("  X: hover")
        print("  Ctrl-C: exit")
        print("")

        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())
            rate = rospy.Rate(20)

            while not rospy.is_shutdown():
                key = self.get_key()

                if key is not None:
                    self.handle_key(key)

                self.publish_cmd()
                rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.stop_motion()
            for _ in range(10):
                self.publish_cmd()


if __name__ == "__main__":
    node = KeyboardCollect()
    node.run()