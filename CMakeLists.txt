cmake_minimum_required(VERSION 3.5)
project(gate_fusion LANGUAGES CXX CUDA)

## Catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

## System dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDAToolkit REQUIRED)

## Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES native)

## Set TensorRT path
set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

## Set compiler flags
add_compile_options(
  -std=c++17
  -g3
  -Wall
  -Wuninitialized
  -Wmaybe-uninitialized)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gate_fusion
  CATKIN_DEPENDS roscpp nav_msgs sensor_msgs cv_bridge image_transport
  DEPENDS EIGEN3 OpenCV CUDAToolkit
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
)

list(APPEND thirdparty_libraries
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${CUDAToolkit_LIBRARIES}
  nvinfer
  nvinfer_plugin
)

add_library(${PROJECT_NAME}
  src/ESKF.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${thirdparty_libraries}
)

# add_executable(${PROJECT_NAME}_node src/gate_fusion_node.cpp)


