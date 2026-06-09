cmake_minimum_required(VERSION 3.0.2)
project(basic_dev)

find_package(catkin REQUIRED COMPONENTS
  airsim_ros
  message_generation
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  trajectory_msgs
  visualization_msgs
  geographic_msgs
  geometry_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  image_transport
  cv_bridge
  pcl_ros
  pcl_conversions
)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    airsim_ros
    roscpp
    rospy
    sensor_msgs
    std_msgs
    trajectory_msgs
    visualization_msgs
    nav_msgs
    geographic_msgs
    geometry_msgs
    std_srvs
    tf2
    tf2_ros
    tf2_geometry_msgs
    image_transport
    cv_bridge
    pcl_ros
    pcl_conversions
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(basic_dev src/basic_dev.cpp)

target_link_libraries(basic_dev
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/check_gate_rel_x_consistency.py
  scripts/check_se3_chain.py
  scripts/gate_poly_planner.py
  scripts/live_debug_viz.py
  scripts/manual_step_trajectory.py
  scripts/odom_to_tf.py
  scripts/pose_to_odom.py
  scripts/rotor_pwm_test.py
  scripts/se3_hover_command.py
  scripts/se3_trajectory_step_command.py
  scripts/yolo_gate_pose.py
  scripts/yolo_gate_pose_keypoint_pnp.py
  scripts/yolo_gate_pose_pnp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS basic_dev
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
