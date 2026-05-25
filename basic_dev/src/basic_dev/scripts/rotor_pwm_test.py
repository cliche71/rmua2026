#!/usr/bin/env python3
import rospy
from airsim_ros.msg import RotorPWM


class RotorPwmTest:
    def __init__(self):
        rospy.init_node("rotor_pwm_test")

        self.motor_topic = rospy.get_param(
            "~motor_topic", "/airsim_node/drone_1/rotor_pwm_cmd"
        )
        self.frame_id = rospy.get_param("~frame_id", "drone_1")
        self.target_pwm = rospy.get_param("~target_pwm", 0.176)
        self.start_pwm = rospy.get_param("~start_pwm", 0.0)
        self.ramp_time = rospy.get_param("~ramp_time", 2.0)
        self.duration = rospy.get_param("~duration", 6.0)
        self.publish_rate = rospy.get_param("~publish_rate", 200.0)

        self.start_time = rospy.Time.now()
        self.pub = rospy.Publisher(self.motor_topic, RotorPWM, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)), self.timer_cb
        )

        rospy.loginfo("rotor_pwm_test started.")
        rospy.loginfo("motor_topic: %s", self.motor_topic)
        rospy.loginfo("target_pwm: %.3f", self.target_pwm)

    def timer_cb(self, _event):
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed > self.duration:
            self.publish_pwm(0.0)
            rospy.signal_shutdown("rotor pwm test finished")
            return

        if self.ramp_time > 0.0:
            ratio = max(0.0, min(1.0, elapsed / self.ramp_time))
        else:
            ratio = 1.0

        pwm = self.start_pwm + ratio * (self.target_pwm - self.start_pwm)
        self.publish_pwm(pwm)
        rospy.loginfo_throttle(0.5, "publish equal rotor pwm: %.3f", pwm)

    def publish_pwm(self, pwm):
        msg = RotorPWM()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.rotorPWM0 = pwm
        msg.rotorPWM1 = pwm
        msg.rotorPWM2 = pwm
        msg.rotorPWM3 = pwm
        self.pub.publish(msg)


if __name__ == "__main__":
    node = RotorPwmTest()
    rospy.spin()
