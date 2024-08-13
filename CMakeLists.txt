cmake_minimum_required(VERSION 3.5)
project(gate_fusion LANGUAGES CXX CUDA)

## Catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  sensor_msgs
  cv_bridge
  image_transport
  flightgoggles
  drolib
  jsk_recognition_msgs
  tf
  std_msgs
  geometry_msgs
  message_runtime
  message_generation
)

## System dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

## Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES native)

## Set TensorRT path
set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

## Set compiler flags
add_compile_options(
  -std=c++17
  -g3
  -Wall)

add_message_files(
  DIRECTORY msg
  FILES
  YoloPose.msg
  YoloPoseArray.msg
  BoundingBox.msg
  Keypoint.msg
)

generate_messages(DEPENDENCIES geometry_msgs std_msgs)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gate_fusion
  CATKIN_DEPENDS roscpp nav_msgs sensor_msgs cv_bridge image_transport flightgoggles drolib jsk_recognition_msgs tf message_runtime std_msgs geometry_msgs message_generation
  DEPENDS eigen libopencv-dev nvidia-cuda yaml-cpp
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)

list(APPEND thirdparty_libraries
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${CUDAToolkit_LIBRARIES}
  yaml-cpp
  nvinfer
  nvinfer_plugin
)

add_library(${PROJECT_NAME}
  src/eskf/ESKF.cpp
  src/gate_sim/GateSimROS.cpp
  src/gate_yolo/GateYoloROS.cpp
  src/pipeline/Pipeline.cpp
  src/pencil_filter/pencil_filter.cpp
  src/yolo/core/core.cpp
  src/yolo/core/tensor.cpp
  src/yolo/core/types.cpp
  src/yolo/utils/utils.cpp
  src/yolo/vision/cudaWarp.cu
  src/yolo/vision/detection.cpp
  src/viz/yolo_viz.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${thirdparty_libraries}
)

add_executable(gate_sim_node src/gate_sim/gate_sim_node.cpp)

target_link_libraries(gate_sim_node
  ${PROJECT_NAME}
  ${thirdparty_libraries}
)

add_executable(gate_yolo_node src/gate_yolo/gate_yolo_node.cpp)

target_link_libraries(gate_yolo_node
  ${PROJECT_NAME}
  ${thirdparty_libraries}
)

add_executable(gate_map_publisher src/viz/gate_map_publisher.cpp)

target_link_libraries(gate_map_publisher
  ${PROJECT_NAME}
  ${thirdparty_libraries}
)

add_executable(pencil_filter_node src/pencil_filter/pencil_filter_node.cpp)

target_link_libraries(pencil_filter_node
  ${PROJECT_NAME}
  ${thirdparty_libraries}
)

add_executable(yolo_viz_node src/viz/yolo_viz_node.cpp)

target_link_libraries(yolo_viz_node
  ${PROJECT_NAME}
  ${thirdparty_libraries}
)