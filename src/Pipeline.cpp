#include "gate_fusion/Pipeline.hpp"

using namespace gate;

Pipeline::Pipeline(const std::string& param_file_path) {
  YAML::Node params = YAML::LoadFile(param_file_path);

  acc_noise_density_multiplier_ = params["acc_noise_density_multiplier"].as<double>();

  std::string imu_params_path = params["config_path_imu"].as<std::string>();
  std::string cam_params_path = params["config_path_imucam"].as<std::string>();

  YAML::Node imu_params = YAML::LoadFile(imu_params_path);
  YAML::Node cam_params = YAML::LoadFile(cam_params_path);

  D_ = cv::Mat(cv::Size(1, 4), CV_64FC1);
  for (int i = 0; i < 4; ++i) {
    D_.at<double>(i, 0) = cam_params["cam0"]["distortion_coeffs"][i].as<double>();
  }

  cv::Mat K_vec = cv::Mat::zeros(cv::Size(4, 1), CV_64FC1);
  for (int i = 0; i < 4; ++i) {
    K_vec.at<double>(i, 0) = cam_params["cam0"]["intrinsics"][i].as<double>();
  }

  K_ = cv::Mat::zeros(cv::Size(3, 3), CV_64FC1);
  K_.at<double>(0, 0) = K_vec.at<double>(0, 0);
  K_.at<double>(1, 1) = K_vec.at<double>(1, 0);
  K_.at<double>(0, 2) = K_vec.at<double>(2, 0);
  K_.at<double>(1, 2) = K_vec.at<double>(3, 0);
  K_.at<double>(2, 2) = 1.0;

  Eigen::Matrix4d T_b_c_ = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      T_b_c_(i, j) = cam_params["cam0"]["T_imu_cam"][i][j].as<double>();
    }
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

  // std::cout << "acc_noise_density_multiplier: " << acc_noise_density_multiplier_ << std::endl;
  // std::cout << "acc_noise_density:" << acc_noise_density_ << std::endl;
  // std::cout << "gate_size:" << gate_size << std::endl;
  // std::cout << "D:\n" << D_ << std::endl;
  // std::cout << "K:\n" << K_ << std::endl;
  // std::cout << "T_b_c:\n" << T_b_c_ << std::endl;
}

void Pipeline::feed_odom(const Eigen::Vector3d& p_b_i,
                         const Eigen::Vector3d& v_b_i,
                         const Eigen::Quaterniond& q_i_b,
                         const double& t) {
  if (!initialized_) {
    ESKF_ = new ESKF(acc_noise_density_, acc_noise_density_multiplier_, p_b_i, v_b_i, t);
    initialized_ = true;
    return;
  }

  ESKF_->feed_prediction(p_b_i - p_b_i_, v_b_i - v_b_i_, t);

  p_b_i_ = p_b_i;
  v_b_i_ = v_b_i;
  R_i_b_ = fsc::QuaternionToRotationMatrix(q_i_b);
}

void Pipeline::feed_corners(const std::vector<std::vector<cv::Point2d>>& corners,
                            const double& t) {
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
  if (!Pipeline::matchGates(T_c_gs, p_b_i, p_b_i_vec, R_vec)) {
    return;
  }

  ESKF_->feed_measurement(p_b_i_vec, R_vec, t);
}

bool Pipeline::solveIPPE(const std::vector<std::vector<cv::Point2d>>& gates,
                         std::vector<Eigen::Matrix<double, 4, 4>>& T_c_gs) {
  for (auto& gate : gates) {
    if (gate.size() == 4) {
      std::vector<cv::Point2d> corners_sorted = Pipeline::sort_corners(gate);
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

      if (t_g_c_vec.empty()) {
        continue;
      }

      Eigen::Vector3d t_g_c;
      Eigen::Vector3d r_c_g;
      cv::cv2eigen(t_g_c_vec[0], t_g_c);
      cv::cv2eigen(r_c_g_vec[0], r_c_g);

      Eigen::Matrix<double, 4, 4> T_c_g = Eigen::Matrix<double, 4, 4>::Identity();
      T_c_g.block<3, 1>(0, 3) = t_g_c;
      T_c_g.block<3, 3>(0, 0) = fsc::AngleAxisToRotationMatrix(r_c_g);

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
                          std::vector<Eigen::Vector3d>& p_b_i_vec,
                          std::vector<Eigen::Matrix3d>& R_vec) {

  Eigen::Matrix<double, 4, 4> T_i_b_pred = Eigen::Matrix<double, 4, 4>::Identity();
  T_i_b_pred.block<3, 3>(0, 0) = R_i_b_;
  T_i_b_pred.block<3, 1>(0, 3) = p_b_i;

  Eigen::Matrix3d R_single = 0.0001 * Eigen::Matrix3d::Identity();

  for (auto& T_c_g : T_c_gs) {
    Eigen::Matrix<double, 4, 4> T_i_g_pred = T_i_b_pred * T_b_c_ * T_c_g;

    for (auto& gate : gate_map_) {
      if ((T_i_g_pred.block<3, 1>(0, 3) - gate).norm() < 1.0) {
        Eigen::Vector3d p_b_i = (gate - T_i_g_pred.block<3, 1>(0, 3)) + T_i_b_pred.block<3, 1>(0, 3);
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