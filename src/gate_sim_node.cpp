#include <ros/ros.h>

#include "gate_fusion/GateSimROS.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "gate_sim_node");

  gate::GateSimROS gate_sim_ros;

  ros::spin();

  return 0;
}