#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include "gate_fusion/ESKF.hpp"
#include "drolib/rotation/rotation_utils.h"

namespace gate {
class Pipeline {
public:
  Pipeline(const std::string& param_file_path);
  ~Pipeline() = default;

  void feed_odom(const Eigen::Vector3d& p_b_i,
                 const Eigen::Vector3d& v_b_i,
                 const Eigen::Quaterniond& q_i_b,
                 const double& t);
  
  void feed_corners(const std::vector<std::vector<cv::Point2d>>& corners,
                    const double& t,
                    std::vector<Eigen::Vector3d>& p_g_i_vec,
                    std::vector<Eigen::Quaterniond>& q_i_g_vec);

  Eigen::Matrix<double, 6, 1> get_state(const double& t) { return ESKF_->get_state(t); };

private:
  bool solveIPPE(const std::vector<std::vector<cv::Point2d>>& gates,
                 std::vector<Eigen::Matrix<double, 4, 4>>& T_c_gs);

  bool matchGates(const std::vector<Eigen::Matrix<double, 4, 4>>& T_c_gs,
                  const Eigen::Vector3d& p_b_i,
                  std::vector<Eigen::Vector3d>& p_b_i_vec,
                  std::vector<Eigen::Matrix3d>& R_vec,
                  std::vector<Eigen::Vector3d>& p_g_i_vec,
                  std::vector<Eigen::Quaterniond>& q_i_g_vec);

  cv::Point2d findCentroid(const std::vector<cv::Point2d>& corners);
  std::vector<cv::Point2d> sort_corners(const std::vector<cv::Point2d>& corners);

  void load_gatemap(const std::string& filepath);

  cv::Mat K_, D_;

  std::vector<Eigen::Vector3d> gate_map_;

  Eigen::Matrix<double, 4, 4> T_b_c_;
  
  ESKF* ESKF_;
  std::vector<cv::Point3d> obj_pts_;

  bool initialized_ = false;
  Eigen::Matrix3d yaw_offset_;
  Eigen::Vector3d start_pos_offset_;
  Eigen::Vector3d p_b_i_;
  Eigen::Vector3d v_b_i_;
  Eigen::Matrix3d R_i_b_;

  double acc_noise_density_multiplier_;
  double acc_noise_density_;


};
  
} // namespace gate


#endif  // PIPELINE_H_