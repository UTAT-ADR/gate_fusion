#ifndef GATE_YOLO_ROS_H_
#define GATE_YOLO_ROS_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <yaml-cpp/yaml.h>

#include "gate_fusion/yolo/vision/detection.h"
#include "gate_fusion/yolo/vision/result.h"
#include "gate_fusion/Pipeline.hpp"
#include "gate_fusion/YoloPoseArray.h"

namespace gate {

class GateYoloROS {
public:
  GateYoloROS(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  GateYoloROS() : GateYoloROS(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~GateYoloROS() {}

private:
  void odomCallBack(const nav_msgs::OdometryConstPtr& odom_msg);

  void imageCallBack(const sensor_msgs::ImageConstPtr& image_msg);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber odom_sub_;
  image_transport::Subscriber image_sub_;
  ros::Publisher odom_pub_;
  ros::Publisher gate_pub_;
  ros::Publisher path_pub_;
  ros::Publisher yolo_pub_;
  tf::TransformBroadcaster br;

  nav_msgs::Path path_msg_;

  std::string param_path_;
  std::string odom_topic_;
  std::string image_topic_;

  std::string engine_file_path_;
  YOLO::DeployDet* model_;

  Pipeline* Pipeline_;

  geometry_msgs::Vector3 dimensions;
  tf::Transform transform;
  bool initialized_ = false;

};
} // namespace gate


#endif  // GATE_YOLO_ROS_H_