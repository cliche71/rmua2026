FROM osrf/ros:noetic-desktop-full-focal

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    libgl1 \
    libglib2.0-0 \
    python3-catkin-tools \
    python3-pip \
    python3-rosdep \
    ros-noetic-cmake-modules \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-geographic-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-nav-msgs \
    ros-noetic-roscpp \
    ros-noetic-rospy \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-std-srvs \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-trajectory-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-image-transport \
    ros-noetic-cv-bridge \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /basic_dev

COPY requirements-docker.txt /basic_dev/requirements-docker.txt

RUN python3 -m pip install --no-cache-dir --upgrade "pip<25" && \
    python3 -m pip install --no-cache-dir --ignore-installed \
      -r /basic_dev/requirements-docker.txt

COPY src /basic_dev/src

RUN chmod +x /basic_dev/src/basic_dev/scripts/*.py && \
    source /opt/ros/noetic/setup.bash && \
    catkin_make \
      "-DCATKIN_WHITELIST_PACKAGES=airsim_ros;imu_gps_odometry;rotors_control;basic_dev" \
      -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /basic_dev/build

COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]
