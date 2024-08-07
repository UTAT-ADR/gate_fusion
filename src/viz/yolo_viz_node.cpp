#include <ros/ros.h>

#include "gate_fusion/yolo_viz.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "yolo_viz_node");

  gate::YOLOSubscriber yolo_viz_node;

  ros::spin();

  return 0;
}