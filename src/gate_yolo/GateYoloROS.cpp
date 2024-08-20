#include "gate_fusion/GateYoloROS.hpp"

using namespace gate;

GateYoloROS::GateYoloROS(const ros::NodeHandle& nh, const ros::NodeHandle& pnh) 
  : nh_(nh), pnh_(pnh) {

  if (pnh_.getParam("param_path", param_path_) != true) {
    ROS_ERROR("Fail to get param file path!");
    exit(1);
  }

  if (pnh_.getParam("odom_topic", odom_topic_) != true) {
    ROS_ERROR("Fail to get odom topic!");
    exit(1);
  }

  if (pnh_.getParam("image_topic", image_topic_) != true) {
    ROS_ERROR("Fail to get image topic!");
    exit(1);
  }
  YAML::Node params = YAML::LoadFile(param_path_);
  double gate_size = params["gate_size"].as<double>();
  engine_file_path_ = params["engine_file_path"].as<std::string>();

  dimensions.x = 0.01;
  dimensions.y = gate_size;
  dimensions.z = gate_size;

  transform.setOrigin(tf::Vector3(params["start_pos_offset"][0].as<double>(),
                                  params["start_pos_offset"][1].as<double>(),
                                  params["start_pos_offset"][2].as<double>()));

  image_transport::ImageTransport it(nh_);

  odom_sub_ = nh_.subscribe(odom_topic_, 1, &GateYoloROS::odomCallBack, this);
  image_sub_ = it.subscribe(image_topic_, 1, &GateYoloROS::imageCallBack, this);

  odom_pub_ = nh_.advertise<nav_msgs::Odometry>("gate_fusion/odom", 1000);
  gate_pub_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>("gate_fusion/gates", 1000);
  path_pub_ = nh_.advertise<nav_msgs::Path>("gate_fusion/path", 1000);
  yolo_pub_ = nh_.advertise<gate_fusion::YoloPoseArray>("gate_fusion/yolo_res", 1000);

  Pipeline_ = new Pipeline(param_path_);
  model_ = new YOLO::DeployDet(engine_file_path_);

  ROS_INFO("Gate fusion pipline has started!");
}

void GateYoloROS::odomCallBack(const nav_msgs::OdometryConstPtr& odom_msg) {
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

  path_msg_.header.frame_id = "world";

  double _t = odom_msg->header.stamp.toSec();

  if (!initialized_) {
    tf::Quaternion q(odom_msg->pose.pose.orientation.x,
                     odom_msg->pose.pose.orientation.y,
                     odom_msg->pose.pose.orientation.z,
                     odom_msg->pose.pose.orientation.w);
    transform.setRotation(q.inverse());
    
    initialized_ = true;
  }

  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "global"));

  Pipeline_->feed_odom(_p, _v, _q, _t);

  Eigen::Matrix<double, 10, 1> state = Pipeline_->get_state(_t);

  nav_msgs::Odometry pub_odom_msg;

  pub_odom_msg.header.frame_id = "world";
  pub_odom_msg.header.stamp = odom_msg->header.stamp;
  pub_odom_msg.child_frame_id = odom_msg->child_frame_id;

  pub_odom_msg.pose.pose.position.x = state(0);
  pub_odom_msg.pose.pose.position.y = state(1);
  pub_odom_msg.pose.pose.position.z = state(2);

  pub_odom_msg.twist.twist.linear.x = state(3);
  pub_odom_msg.twist.twist.linear.y = state(4);
  pub_odom_msg.twist.twist.linear.z = state(5);

  pub_odom_msg.pose.pose.orientation.x = state(6);
  pub_odom_msg.pose.pose.orientation.y = state(7);
  pub_odom_msg.pose.pose.orientation.z = state(8);
  pub_odom_msg.pose.pose.orientation.w = state(9);

  pub_odom_msg.twist.twist.angular = odom_msg->twist.twist.angular;

  odom_pub_.publish(pub_odom_msg);

  path_msg_.header.stamp = odom_msg->header.stamp;
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = odom_msg->header.stamp;
  pose.header.frame_id = "world";
  pose.pose = pub_odom_msg.pose.pose;
  path_msg_.poses.push_back(pose);

  path_pub_.publish(path_msg_);
}

void GateYoloROS::imageCallBack(const sensor_msgs::ImageConstPtr& image_msg) {
  std::vector<std::vector<cv::Point2d>> _corners;
  double _t = image_msg->header.stamp.toSec();

  cv_bridge::CvImagePtr BridgePtr;
  BridgePtr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);

  YOLO::Image image_in(BridgePtr->image.data, BridgePtr->image.cols, BridgePtr->image.rows);
  auto result = model_->predict(image_in);

  gate_fusion::YoloPoseArray yolo_msgs;

  for (int i = 0; i < result.num; ++i) {
    const auto& box       = result.boxes[i];
    int         cls       = result.classes[i];
    float       score     = result.scores[i];
    const auto& kps       = result.kps[i];

    std::vector<cv::Point2d> gate;
    gate_fusion::YoloPose yolo_msg;
    yolo_msg.box.cls = cls;
    yolo_msg.box.score = score;
    yolo_msg.box.x_center = box.x_center;
    yolo_msg.box.y_center = box.y_center;
    yolo_msg.box.width = box.width;
    yolo_msg.box.height = box.height;

    for (auto& kp : kps) {
      gate_fusion::Keypoint corner;
      corner.point.x = kp.x;
      corner.point.y = kp.y;
      corner.score = kp.score;

      gate.push_back(cv::Point2d(kp.x, kp.y));
      yolo_msg.corners.push_back(corner);
    }
    _corners.push_back(gate);
  }

  yolo_msgs.header.stamp = image_msg->header.stamp;
  yolo_msgs.header.frame_id = image_msg->header.frame_id;
  yolo_pub_.publish(yolo_msgs);

  std::vector<Eigen::Vector3d> p_g_i_vec;
  std::vector<Eigen::Quaterniond> q_i_g_vec;

  Pipeline_->feed_corners(_corners, _t, p_g_i_vec, q_i_g_vec);

  jsk_recognition_msgs::BoundingBoxArray gate_msg;
  gate_msg.header.stamp = image_msg->header.stamp;
  gate_msg.header.frame_id = "world";
  for (int i = 0; i < static_cast<int>(p_g_i_vec.size()); i++) {
    jsk_recognition_msgs::BoundingBox gate;
    gate.header.stamp = image_msg->header.stamp;
    gate.header.frame_id = "world";
    gate.pose.position.x = p_g_i_vec[i](0);
    gate.pose.position.y = p_g_i_vec[i](1);
    gate.pose.position.z = p_g_i_vec[i](2);
    gate.pose.orientation.x = q_i_g_vec[i].x();
    gate.pose.orientation.y = q_i_g_vec[i].y();
    gate.pose.orientation.z = q_i_g_vec[i].z();
    gate.pose.orientation.w = q_i_g_vec[i].w();
    gate.label = i;
    gate.dimensions = dimensions;
    gate_msg.boxes.push_back(gate);
  }
  gate_pub_.publish(gate_msg);
}