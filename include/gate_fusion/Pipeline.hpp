#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "gate_fusion/ESKF.hpp"

namespace gate {
class Pipeline {
public:
  Pipeline(const std::string& param_file_path);
  ~Pipeline() = default;

  void feed_odom(const Eigen::Vector3d& p,
                 const Eigen::Vector3d& v,
                 const Eigen::Quaterniond& q,
                 const double& t);
  
  void feed_corners(const std::vector<std::vecor<cv::Point2d>>& corners,
                    const double& t);

  Eigen::Matrix<double, 6, 6> get_state(const double& t) { return ESKF->get_state(t); };

private:
  bool solveIPPE(const std::vector<std::vecor<cv::Point2d>>& gates,
                 std::vector<Eigen::Matrix<double, 4, 4>>& T_GATEtoCAMs,
                 std::vector<std::vector<cv::Point2d>>& res_pt_arr);

  bool matchGates(const std::vector<Eigen::Matrix<double, 4, 4>>& T_GATEtoCAMs,
                  const Eigen::Matrix<double, 6, 1>& state,
                  std::vector<Eigen::Vector3d>& p_vec,
                  std::vector<Eigen::Matrix3d>& R_vec);

  cv::Point2d findCentroid(const std::vector<cv::Point2d>& corners);
  std::vector<cv::Point2d> sort_corners(const std::vector<cv::Point2d>& corners);

  void load_gatemap(const std::string& filepath);

  cv::Mat K_, D_;

  std::vector<Eigen::Vector3d> gate_map_;

  Eigen::Matrix<double, 4, 4> T_IMUtoCAM_;
  
  ESKF* ESKF_;
  std::vector<cv::Point3d> obj_pts_;

  bool initialized_ = false;
  Eigen::Vector3d p_;
  Eigen::Vector3d v_;
  double t_;

  double acc_noise_density_multiplier_;
  double acc_noise_density_;


}
  
} // namespace gate


#endif  // PIPELINE_H_