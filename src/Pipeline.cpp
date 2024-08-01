#include "gate_fusion/Pipeline.hpp"

using namespace gate;

Pipeline::Pipeline(const std::string& param_file_path) {
  cv::FileStorage params = cv::FileStorage(param_file_path, cv::FileStorage::READ);

  params["acc_noise_density_multiplier"] >> acc_noise_density_multiplier_;

  std::string imu_params_path;
  std::string cam_params_path;
  params["config_path_imu"] >> imu_params_path;
  params["config_path_imucam"] >> cam_params_path;
  cv::FileStorage imu_params = cv::FileStorage(imu_params_path, cv::FileStorage::READ);
  cv::FileStorage cam_params = cv::FileStorage(cam_params_path, cv::FileStorage::READ);

  D_ = cv::Mat(cv::Size(1, 4), CV_64FC1)
  cam_params["cam0"]["distortion_coeffs"] >> D_;
  cv::Mat K_vec;
  cam_params["cam0"]["intrinsics"] >> K_vec;
  K_ = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
  K_.at<double>(0, 0) = K_vec.at<double>(0, 0);
  K_.at<double>(1, 1) = K_vec.at<double>(0, 1);
  K_.at<double>(0, 2) = K_vec.at<double>(0, 2);
  K_.at<double>(1, 2) = K_vec.at<double>(0, 3);
  K_.at<double>(2, 2) = 1.0;

  cv::Mat T_IMUtoCAM_cv = cv::Mat(cv::Size(4, 4), CV_64FC1);
  cam_params["cam0"]["T_imu_cam"] >> T_IMUtoCAM_cv;
  cv::cv2eigen(T_IMUtoCAM_cv, T_IMUtoCAM_);

  double gate_size;
  params["gate_size"] >> gate_size;

  obj_pts_ = std::vector<cv::Point3d>{
    cv::Point3d(-gate_size_ / 2.0, gate_size_ / 2.0, 0.0),
    cv::Point3d(gate_size_ / 2.0, gate_size_ / 2.0, 0.0),
    cv::Point3d(gate_size_ / 2.0, -gate_size_ / 2.0, 0.0),
    cv::Point3d(-gate_size_ / 2.0, -gate_size_ / 2.0, 0.0)};

  std::string gate_map_path;
  params["gate_map_path"] >> gate_map_path;
  Pipeline::load_gatemap(gate_map_path);

  imu_params["imu0"]["accelerometer_noise_density"] >> acc_noise_density;
}

void Pipeline::feed_odom(const Eigen::Vector3d& p,
                         const Eigen::Vector3d& v,
                         const Eigen::Quaterniond& q,
                         const double& t) {
  if (!initialized) {
    ESKF_ = new ESKF(acc_noise_density_, acc_noise_density_multiplier_, p, v, t);
    initialized = true;
    return;
  }

  
}