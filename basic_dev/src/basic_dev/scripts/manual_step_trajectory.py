#!/usr/bin/env python3
import math
import rospy

from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler


latest_odom = None


def odom_cb(msg):
    global latest_odom
    latest_odom = msg


def make_traj(x, y, z, yaw, duration):
    msg = MultiDOFJointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.joint_names = ["base_link"]

    tr = Transform()
    tr.translation.x = x
    tr.translation.y = y
    tr.translation.z = z

    q = quaternion_from_euler(0.0, 0.0, yaw)
    tr.rotation.x = q[0]
    tr.rotation.y = q[1]
    tr.rotation.z = q[2]
    tr.rotation.w = q[3]

    vel = Twist()
    acc = Twist()

    pt = MultiDOFJointTrajectoryPoint()
    pt.transforms.append(tr)
    pt.velocities.append(vel)
    pt.accelerations.append(acc)
    pt.time_from_start = rospy.Duration(duration)

    msg.points.append(pt)
    return msg


def main():
    rospy.init_node("manual_step_trajectory")

    odom_topic = rospy.get_param("~odom_topic", "/eskf_odom")
    cmd_topic = rospy.get_param("~cmd_topic", "/command/trajectory")

    step_x = float(rospy.get_param("~step_x", 0.0))
    step_y = float(rospy.get_param("~step_y", 0.0))
    step_z = float(rospy.get_param("~step_z", 0.0))

    # 直接指定目标 z，最适合排查 z 符号
    target_z_param = rospy.get_param("~target_z", None)

    # AirSim NED 测试默认：takeoff_height=0.3 -> target_z=-0.3
    takeoff_height = float(rospy.get_param("~takeoff_height", 0.0))

    duration = float(rospy.get_param("~duration", 999.0))
    rate_hz = float(rospy.get_param("~rate", 20.0))

    rospy.Subscriber(odom_topic, Odometry, odom_cb, queue_size=1)
    pub = rospy.Publisher(cmd_topic, MultiDOFJointTrajectory, queue_size=1)

    rospy.loginfo("manual_step_trajectory waiting for odom: %s", odom_topic)

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown() and latest_odom is None:
        rate.sleep()

    if rospy.is_shutdown():
        return

    p = latest_odom.pose.pose.position
    q_odom = latest_odom.pose.pose.orientation
    _, _, current_yaw = euler_from_quaternion([
        q_odom.x, q_odom.y, q_odom.z, q_odom.w
    ])

    yaw_param = rospy.get_param("~yaw", None)
    if yaw_param is None:
        yaw = current_yaw
    else:
        yaw = float(yaw_param)

    target_x = p.x + step_x
    target_y = p.y + step_y

    if target_z_param is not None:
        target_z = float(target_z_param)
    elif takeoff_height > 0.0:
        target_z = -abs(takeoff_height)
    else:
        target_z = p.z + step_z

    rospy.loginfo(
        "manual_step_trajectory target=(%.3f, %.3f, %.3f), current=(%.3f, %.3f, %.3f), step=(%.3f, %.3f, %.3f), current_yaw=%.3f rad, target_yaw=%.3f rad",
        target_x, target_y, target_z,
        p.x, p.y, p.z,
        step_x, step_y, step_z,
        current_yaw, yaw
    )

    while not rospy.is_shutdown():
        pub.publish(make_traj(target_x, target_y, target_z, yaw, duration))
        rate.sleep()


if __name__ == "__main__":
    main()
