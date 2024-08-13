#include <ros/ros.h>

#include "gate_fusion/GateYoloROS.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "gate_yolo_node");

  gate::GateYoloROS gate_yolo_ros;

  ros::spin();

  return 0;
}