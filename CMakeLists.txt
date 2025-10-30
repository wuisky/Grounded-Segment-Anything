cmake_minimum_required(VERSION 3.8)
project(Grounded-Segment-Anything)

find_package(ament_cmake REQUIRED)
set(INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/Grounded-Segment-Anything)

# --- コピー処理 ---
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION Grounded-Segment-Anything
  USE_SOURCE_PERMISSIONS
  PATTERN "build" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.egg-info" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.pyo" EXCLUDE
)

# --- post install コマンド ---
install(CODE "
  message(STATUS \"Running uv sync and uv pip install -e . inside ${INSTALL_DIR}\")
  execute_process(
    COMMAND uv sync
    WORKING_DIRECTORY ${INSTALL_DIR}
  )
  execute_process(
    COMMAND CUDA_HOME=/usr/local/cuda-12.8/ uv pip install --no-build-isolation -e ./GroundingDINO
    WORKING_DIRECTORY ${INSTALL_DIR}
  )
")

ament_package()
