cmake_minimum_required(VERSION 2.8.3)
project(rotors_control)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  airsim_ros
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  trajectory_msgs
  cmake_modules
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include ${Eigen3_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
  LIBRARIES lee_position_controller #roll_pitch_yawrate_thrust_controller
  CATKIN_DEPENDS airsim_ros geometry_msgs nav_msgs roscpp sensor_msgs trajectory_msgs
  DEPENDS Eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(lee_position_controller
  src/library/lee_position_controller.cpp
)

# add_library(roll_pitch_yawrate_thrust_controller
#   src/library/roll_pitch_yawrate_thrust_controller.cpp
# )

target_link_libraries(lee_position_controller ${catkin_LIBRARIES})
add_dependencies(lee_position_controller ${catkin_EXPORTED_TARGETS})

# target_link_libraries(roll_pitch_yawrate_thrust_controller ${catkin_LIBRARIES})
# add_dependencies(roll_pitch_yawrate_thrust_controller ${catkin_EXPORTED_TARGETS})

add_executable(lee_position_controller_node src/nodes/lee_position_controller_node.cpp)
add_dependencies(lee_position_controller_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(lee_position_controller_node
  lee_position_controller ${catkin_LIBRARIES})

add_executable(hovering_example src/hovering_example.cpp )
target_link_libraries(hovering_example ${catkin_LIBRARIES})

# add_executable(roll_pitch_yawrate_thrust_controller_node
#   src/nodes/roll_pitch_yawrate_thrust_controller_node.cpp)
# add_dependencies(roll_pitch_yawrate_thrust_controller_node ${catkin_EXPORTED_TARGETS})
# target_link_libraries(roll_pitch_yawrate_thrust_controller_node
#   roll_pitch_yawrate_thrust_controller ${catkin_LIBRARIES})

install(TARGETS lee_position_controller #roll_pitch_yawrate_thrust_controller
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS lee_position_controller_node #roll_pitch_yawrate_thrust_controller_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)
