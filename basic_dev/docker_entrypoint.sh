#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
source /basic_dev/devel/setup.bash

MODEL_PATH="/basic_dev/src/basic_dev/models/gate_model.pt"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: model not found: ${MODEL_PATH}"
  exit 1
fi

python3 - <<'PY'
import cv2
import numpy
import torch
import ultralytics

print("Docker Python dependencies OK")
print("opencv:", cv2.__version__)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
PY

exec roslaunch basic_dev rmua_submit.launch
