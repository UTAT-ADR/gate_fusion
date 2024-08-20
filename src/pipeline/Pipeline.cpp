#include "gate_fusion/Pipeline.hpp"
#include <iostream>

using namespace gate;

Pipeline::Pipeline(const std::string& param_file_path) {
  YAML::Node params = YAML::LoadFile(param_file_path);

  acc_noise_density_multiplier_ = params["acc_noise_density_multiplier"].as<double>();
  gate_match_threshold_ = params["gate_match_threshold"].as<double>();
  rpj_error_threshold_ = params["rpj_error_threshold"].as<double>();

  std::string imu_params_path = params["config_path_imu"].as<std::string>();
  std::string cam_params_path = params["config_path_imucam"].as<std::string>();

  YAML::Node imu_params = YAML::LoadFile(imu_params_path);
  YAML::Node cam_params = YAML::LoadFile(cam_params_path);

  Eigen::Matrix<double, 1, 4> D;
  for (int i = 0; i < 4; ++i) {
    D(0, i) = cam_params["cam0"]["distortion_coeffs"][i].as<double>();
  }

  timeshift_cam_imu_ = cam_params["cam0"]["timeshift_cam_imu"].as<double>();

  cv::eigen2cv(D, D_);

  Eigen::Matrix<double, 1, 4> K_vec;
  for (int i = 0; i < 4; i++) {
    K_vec(0, i) = cam_params["cam0"]["intrinsics"][i].as<double>();
  }

  Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
  K(0, 0) = K_vec(0, 0);
  K(1, 1) = K_vec(0, 1);
  K(0, 2) = K_vec(0, 2);
  K(1, 2) = K_vec(0, 3);
  K(2, 2) = 1.0;

  cv::eigen2cv(K, K_);

  T_b_c_ = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      T_b_c_(i, j) = cam_params["cam0"]["T_imu_cam"][i][j].as<double>();
    }
  }

  for (int i = 0; i < 3; i++) {
    start_pos_offset_(i, 0) = params["start_pos_offset"][i].as<double>();
  }

  double gate_size = params["gate_size"].as<double>();

  obj_pts_ = std::vector<cv::Point3d>{
    cv::Point3d(-gate_size / 2.0, gate_size / 2.0, 0.0),
    cv::Point3d(gate_size / 2.0, gate_size / 2.0, 0.0),
    cv::Point3d(gate_size / 2.0, -gate_size / 2.0, 0.0),
    cv::Point3d(-gate_size / 2.0, -gate_size / 2.0, 0.0)};

  std::string gate_map_path = params["gate_map_path"].as<std::string>();
  Pipeline::load_gatemap(gate_map_path);

  acc_noise_density_ = imu_params["imu0"]["accelerometer_noise_density"].as<double>();

  // p_b_i_ = Eigen::Vector3d::Zero();
  // v_b_i_ = Eigen::Vector3d::Zero();
  // R_i_b_ = Eigen::Matrix3d::Identity();

  std::cout << "acc_noise_density_multiplier: " << acc_noise_density_multiplier_ << std::endl;
  std::cout << "gate_match_threshold: " << gate_match_threshold_ << std::endl;
  std::cout << "rpj_error_threshold: " << rpj_error_threshold_ << std::endl;
  std::cout << "acc_noise_density:" << acc_noise_density_ << std::endl;
  std::cout << "gate_size:" << gate_size << std::endl;
  std::cout << "D:\n" << D_ << std::endl;
  std::cout << "K:\n" << K_ << std::endl;
  std::cout << "T_b_c:\n" << T_b_c_ << std::endl;
  std::cout << "start_pos_offset:\n" << start_pos_offset_ << std::endl;
  std::cout << "timeshift_cam_imu:\n" << timeshift_cam_imu_ << std::endl;
}

void Pipeline::feed_odom(const Eigen::Vector3d& p_b_i,
                         const Eigen::Vector3d& v_b_i,
                         const Eigen::Quaterniond& q_i_b,
                         const double& t) {
  if (!initialized_) {
    yaw_offset_ = q_i_b.toRotationMatrix().transpose();
    Eigen::Vector3d _p_b_i = yaw_offset_ * p_b_i + start_pos_offset_;
    Eigen::Vector3d _v_b_i = yaw_offset_ * v_b_i;
    ESKF_ = new ESKF(acc_noise_density_, acc_noise_density_multiplier_, _p_b_i, _v_b_i, t);
    p_b_i_ = _p_b_i;
    v_b_i_ = _v_b_i;
    R_i_b_ = yaw_offset_ * q_i_b.toRotationMatrix();
    initialized_ = true;
    return;
  }

  Eigen::Vector3d _p_b_i = yaw_offset_ * p_b_i + start_pos_offset_;
  Eigen::Vector3d _v_b_i = yaw_offset_ * v_b_i;

  ESKF_->feed_prediction(_p_b_i - p_b_i_, _v_b_i - v_b_i_, t);

  p_b_i_ = _p_b_i;
  v_b_i_ = _v_b_i;
  R_i_b_ = yaw_offset_ * q_i_b.toRotationMatrix();
}

void Pipeline::feed_corners(const std::vector<std::vector<cv::Point2d>>& corners,
                            const double& t,
                            std::vector<Eigen::Vector3d>& p_g_i_vec,
                            std::vector<Eigen::Quaterniond>& q_i_g_vec) {
  if (!initialized_) {
    return;
  }

  std::vector<Eigen::Matrix<double, 4, 4>> T_c_gs;
  if (!Pipeline::solveIPPE(corners, T_c_gs)) {
    return;
  }

  Eigen::Vector3d p_b_i = (ESKF_->get_state(t)).block<3, 1>(0, 0);

  std::vector<Eigen::Vector3d> p_b_i_vec;
  std::vector<Eigen::Matrix3d> R_vec;
  if (!Pipeline::matchGates(T_c_gs, p_b_i, t_imu, p_b_i_vec, R_vec, p_g_i_vec, q_i_g_vec)) {
    return;
  }

  ESKF_->feed_measurement(p_b_i_vec, R_vec, t_imu);
}

bool Pipeline::solveIPPE(const std::vector<std::vector<cv::Point2d>>& gates,
                         std::vector<Eigen::Matrix<double, 4, 4>>& T_c_gs) {
  for (auto& gate : gates) {
    if (gate.size() == 4) {
      std::vector<cv::Point2d> corners_sorted = Pipeline::sort_corners(gate);
      // for (auto& corner : corners_sorted) {
      //   std::cout << corner << std::endl;
      // }
      std::vector<cv::Mat> t_g_c_vec, r_c_g_vec;
      cv::Mat rpj_err = cv::Mat(2,1, CV_64FC1);
      cv::solvePnPGeneric(obj_pts_,
                          corners_sorted,
                          K_,
                          D_,
                          r_c_g_vec, 
                          t_g_c_vec,
                          false,
                          cv::SOLVEPNP_IPPE_SQUARE,
                          cv::noArray(),
                          cv::noArray(),
                          rpj_err);

      // std::cout << "rpj_err:\n" << sqrt(rpj_err.at<double>(0) * rpj_err.at<double>(0) +
      //                 rpj_err.at<double>(1) * rpj_err.at<double>(1)) << std::endl;

      if (t_g_c_vec.empty()) {
        continue;
      } else if (sqrt(rpj_err.at<double>(0) * rpj_err.at<double>(0) +
                      rpj_err.at<double>(1) * rpj_err.at<double>(1)) > rpj_error_threshold_) {
        continue;
      }
      cv::Mat R_c_g;
      cv::Rodrigues(r_c_g_vec[0], R_c_g);
      Eigen::Vector3d t_g_c;
      Eigen::Matrix3d R_c_g_eigen;
      cv::cv2eigen(t_g_c_vec[0], t_g_c);
      cv::cv2eigen(R_c_g, R_c_g_eigen);

      Eigen::Matrix<double, 4, 4> T_c_g = Eigen::Matrix<double, 4, 4>::Identity();
      T_c_g.block<3, 1>(0, 3) = t_g_c;
      T_c_g.block<3, 3>(0, 0) = R_c_g_eigen.transpose();
      // std::cout << "t_g_c:\n" << t_g_c << std::endl;
      // std::cout << "R_c_g:\n" << drolib::rad2deg(drolib::rotationMatrixToEulerAnglesRPY(R_c_g_eigen)) << std::endl;
      // std::cout << "rpj_err:\n" << rpj_err << std::endl;
      T_c_gs.push_back(T_c_g);
    }
  }

  if (T_c_gs.empty()) {
    return false;
  } else {
    return true;
  }
}

bool Pipeline::matchGates(const std::vector<Eigen::Matrix<double, 4, 4>>& T_c_gs,
                          const Eigen::Vector3d& p_b_i,
                          const double& t,
                          std::vector<Eigen::Vector3d>& p_b_i_vec,
                          std::vector<Eigen::Matrix3d>& R_vec,
                          std::vector<Eigen::Vector3d>& p_g_i_vec,
                          std::vector<Eigen::Quaterniond>& q_i_g_vec) {

  // Eigen::Matrix<double, 4, 4> T_i_b_pred = Eigen::Matrix<double, 4, 4>::Identity();
  // T_i_b_pred.block<3, 3>(0, 0) = R_i_b_;
  // T_i_b_pred.block<3, 1>(0, 3) = p_b_i;

  Eigen::Vector3d _p_b_i = (ESKF_->get_state(t)).block<3, 1>(0, 0);
  
  Eigen::Vector3d p_c_b = T_b_c_.block<3, 1>(0, 3);
  Eigen::Matrix3d R_b_c = T_b_c_.block<3, 3>(0, 0); 

  Eigen::Matrix3d R_single = 0.1 * Eigen::Matrix3d::Identity();

  for (auto& T_c_g : T_c_gs) {
    // Eigen::Vector3d t_g_b = R_b_c * T_c_g.block<3, 1>(0, 3) + t_c_b;
    // Eigen::Matrix3d R_b_g = R_b_c * T_c_g.block<3, 3>(0, 0) * R_b_c.transpose();
    // Eigen::Matrix<double, 4, 4> T_b_g = Eigen::Matrix<double, 4, 4>::Identity();
    // T_b_g.block<3, 3>(0, 0) = R_b_g;
    // T_b_g.block<3, 1>(0, 3) = t_g_b;

    Eigen::Vector3d p_g_i = R_i_b_ * R_b_c * T_c_g.block<3, 1>(0, 3) + R_i_b_ * p_c_b + _p_b_i;
    Eigen::Quaterniond q_i_g(R_i_b_.transpose() * R_b_c * T_c_g.block<3, 3>(0, 0) * R_b_c.transpose());
    // std::cout << "R_i_g:\n" << drolib::rad2deg(drolib::rotationMatrixToEulerAnglesRPY(R_i_b_.transpose() * R_b_c * T_c_g.block<3, 3>(0, 0) * R_b_c.transpose())) << std::endl;
    

    p_g_i_vec.push_back(p_g_i);
    q_i_g_vec.push_back(q_i_g.inverse());

    // std::cout << "T_c_g:\n" << T_c_g.inverse() << std::endl;
    // Eigen::Matrix<double, 4, 4> T_i_g_pred = T_i_b_pred * T_b_g * T_i_b_pred.inverse();
    // std::cout << "t_b_i:\n" << p_b_i << std::endl;
    // std::cout << "R_i_b:\n" << R_i_b_ << std::endl;
    // std::cout << "t_g_c:\n" << T_c_g.block<3, 1>(0, 3) << std::endl;
    // std::cout << "R_c_g:\n" << T_c_g.block<3, 3>(0, 0) << std::endl;
    // std::cout << "t_c_b:\n" << t_c_b << std::endl;
    // std::cout << "R_b_c:\n" << R_b_c << std::endl;
    // std::cout << "t_g_i:\n" << t_g_i << std::endl;
    // std::cout << "R_i_g:\n" << drolib::rad2deg(drolib::rotationMatrixToEulerAnglesRPY(R_i_g)) << std::endl;

    for (auto& gate : gate_map_) {
      if ((p_g_i - gate).norm() < gate_match_threshold_) {
        Eigen::Vector3d p_b_i = (gate - p_g_i) + _p_b_i;
        p_b_i_vec.push_back(p_b_i);
        R_vec.push_back(R_single);
        break;
      }
    }
  }

  if (p_b_i_vec.empty()) {
    return false;
  } else {
    return true;
  }
}

cv::Point2d Pipeline::findCentroid(const std::vector<cv::Point2d>& corners) {
    double x = 0, y = 0;
    for (const auto& corner : corners) {
        x += corner.x;
        y += corner.y;
    }
    return cv::Point2d(x / corners.size(), y / corners.size());
}

// Function to sort corners in the specified order
std::vector<cv::Point2d> Pipeline::sort_corners(const std::vector<cv::Point2d>& corners) {
    // Find the centroid of the corners
    cv::Point2d centroid = findCentroid(corners);

    // Rearrange sorted corners to match the required order
    std::vector<cv::Point2d> orderedCorners(4);
    for (const auto& corner : corners) {
        if (corner.x < centroid.x && corner.y < centroid.y) {
            orderedCorners[3] = corner; // top-left
        } else if (corner.x > centroid.x && corner.y < centroid.y) {
            orderedCorners[2] = corner; // top-right
        } else if (corner.x > centroid.x && corner.y > centroid.y) {
            orderedCorners[1] = corner; // bottom-right
        } else if (corner.x < centroid.x && corner.y > centroid.y) {
            orderedCorners[0] = corner; // bottom-left
        }
    }

    return orderedCorners;
}

void Pipeline::load_gatemap(const std::string& filepath) {
  YAML::Node config = YAML::LoadFile(filepath);

  // Check if the YAML file loaded correctly
  if (!config) {
    std::cerr << "Error loading YAML file: " << filepath << std::endl;
    return;
  }

  // Iterate through all nodes in the YAML file
  for (const auto& node : config) {
    // Ensure that the node is a map and contains the "position" key
    if (node.second.IsMap() && node.second["position"]) {
      const YAML::Node& positionNode = node.second["position"];
      if (positionNode.size() == 3) {
        double x = positionNode[0].as<double>();
        double y = positionNode[1].as<double>();
        double z = positionNode[2].as<double>();

        Eigen::Vector3d position(x, y, z);
        gate_map_.push_back(position);
      }
    }
  }
    std::cout <<"Loaded " << gate_map_.size() << " gates." << std::endl;
}

Eigen::Matrix<double, 10, 1> Pipeline::get_state(const double t) {
  Eigen::Matrix<double, 10, 1> cur_state;
  cur_state.block<6, 1>(0, 0) = ESKF_->get_state(t);
  Eigen::Quaterniond q_res(R_i_b_);
  cur_state(6) = q_res.x();
  cur_state(7) = q_res.y();
  cur_state(8) = q_res.z();
  cur_state(9) = q_res.w();
  return cur_state;
}