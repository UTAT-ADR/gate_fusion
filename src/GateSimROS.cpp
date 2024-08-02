#include "gate_fusion/GateSimROS.hpp"

using namespace gate;

GateSimROS::GateSimROS(const ros::NodeHandle& nh, const ros::NodeHandle& pnh) 
  : nh_(nh), pnh_(pnh) {

  if (pnh_.getParam("param_path", param_path_) != true) {
    ROS_ERROR("Fail to get param file path!");
    exit(1);
  }

  if (pnh_.getParam("odom_topic", odom_topic_) != true) {
    ROS_ERROR("Fail to get odom topic!");
    exit(1);
  }

  if (pnh_.getParam("gate_topic", gate_topic_) != true) {
    ROS_ERROR("Fail to get gate topic!");
    exit(1);
  }

  odom_sub_ = nh_.subscribe(odom_topic_, 1, &GateSimROS::odomCallBack, this);
  corner_sub_ = nh_.subscribe(gate_topic_, 1, &GateSimROS::cornerCallBack, this);

  odom_pub_ = nh_.advertise<nav_msgs::Odometry>("gate_fusion_odom", 1000);

  Pipeline_ = new Pipeline(param_path_);

  ROS_INFO("Gate fusion pipline has started!");
}

void GateSimROS::odomCallBack(const nav_msgs::OdometryConstPtr& odom_msg) {
  Eigen::Vector3d _p(odom_msg->pose.pose.position.x,
                     odom_msg->pose.pose.position.y,
                     odom_msg->pose.pose.position.z);

  Eigen::Vector3d _v(odom_msg->twist.twist.linear.x,
                     odom_msg->twist.twist.linear.y,
                     odom_msg->twist.twist.linear.z);

  Eigen::Quaterniond _q(odom_msg->pose.pose.orientation.w,
                        odom_msg->pose.pose.orientation.x,
                        odom_msg->pose.pose.orientation.y,
                        odom_msg->pose.pose.orientation.z);

  double _t = odom_msg->header.stamp.toSec();

  Pipeline_->feed_odom(_p, _v, _q, _t);

  Eigen::Matrix<double, 6, 1> state = Pipeline_->get_state(_t);

  nav_msgs::Odometry pub_odom_msg;

  pub_odom_msg.header.frame_id = odom_msg->header.frame_id;
  pub_odom_msg.header.stamp = odom_msg->header.stamp;
  pub_odom_msg.child_frame_id = odom_msg->child_frame_id;

  pub_odom_msg.pose.pose.position.x = state(0);
  pub_odom_msg.pose.pose.position.y = state(1);
  pub_odom_msg.pose.pose.position.z = state(2);

  pub_odom_msg.twist.twist.linear.x = state(3);
  pub_odom_msg.twist.twist.linear.y = state(4);
  pub_odom_msg.twist.twist.linear.z = state(5);

  pub_odom_msg.pose.pose.orientation.x = odom_msg->pose.pose.orientation.x;
  pub_odom_msg.pose.pose.orientation.y = odom_msg->pose.pose.orientation.y;
  pub_odom_msg.pose.pose.orientation.z = odom_msg->pose.pose.orientation.z;
  pub_odom_msg.pose.pose.orientation.w = odom_msg->pose.pose.orientation.w;

  pub_odom_msg.twist.twist.angular.x = odom_msg->twist.twist.angular.x;
  pub_odom_msg.twist.twist.angular.y = odom_msg->twist.twist.angular.y;
  pub_odom_msg.twist.twist.angular.z = odom_msg->twist.twist.angular.z;

  odom_pub_.publish(pub_odom_msg);
}

void GateSimROS::cornerCallBack(const flightgoggles::IRMarkerArrayConstPtr& corner_msg) {
  std::vector<std::vector<cv::Point2d>> _corners;
  std::vector<std::string> _corner_ids;
  double _t = corner_msg->header.stamp.toSec();

  for (auto& marker : corner_msg->markers) {
    bool exist = false;
    for (int i = 0; i < static_cast<int>(_corner_ids.size()); i++) {
      if (_corner_ids[i] == marker.landmarkID.data) {
        _corners[i].push_back(cv::Point2d(marker.x, marker.y));
        exist = true;
        break;
      }
    }

    if (!exist) {
      std::vector<cv::Point2d> new_gate;

      _corner_ids.push_back(marker.landmarkID.data);
      new_gate.push_back(cv::Point2d(marker.x, marker.y));
      _corners.push_back(new_gate);
    }
  }

  Pipeline_->feed_corners(_corners, _t);
}