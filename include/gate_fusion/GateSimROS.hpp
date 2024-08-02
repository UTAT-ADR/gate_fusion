#ifndef GATE_SIM_ROS_H_
#define GATE_SIM_ROS_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <flightgoggles/IRMarkerArray.h>

#include "gate_fusion/Pipeline.hpp"

namespace gate {

class GateSimROS {
public:
  GateSimROS(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  GateSimROS() : GateSimROS(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~GateSimROS() {}

private:
  void odomCallBack(const nav_msgs::OdometryConstPtr& odom_msg);

  void cornerCallBack(const flightgoggles::IRMarkerArrayConstPtr& corner_msg);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber odom_sub_;
  ros::Subscriber corner_sub_;
  ros::Publisher odom_pub_;

  std::string param_path_;
  std::string odom_topic_;
  std::string gate_topic_;

  Pipeline* Pipeline_;

};
} // namespace gate


#endif  // GATE_SIM_ROS_H_