#ifndef YOLO_VIZ_HPP
#define YOLO_VIZ_HPP

#include <random>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

#include "gate_fusion/yolo/vision/detection.h"
#include "gate_fusion/yolo/vision/result.h"

namespace gate {

class YOLOSubscriber {
 public:
  YOLOSubscriber(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  YOLOSubscriber()
      : YOLOSubscriber(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~YOLOSubscriber() {}

 private:
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  
  std::vector<std::pair<std::string, cv::Scalar>> generateLabelColorPairs(const std::string& labelFile);
  void visualize(cv::Mat& image,
                 const YOLO::DetectionResult& result, 
                 const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  image_transport::Subscriber imageSub_;
  image_transport::Publisher yoloPub_;

  std::string label_file_path;
  std::string img_topic;
  std::string engine_file_path;

  YOLO::DeployDet* model;
  std::vector<std::pair<std::string, cv::Scalar>> labels;

  cv::Mat K, D;

};
} // namespace gate

#endif