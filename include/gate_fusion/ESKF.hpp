#ifndef ESKF_H_
#define ESKF_H_

#include <Eigen/Eigen>

namespace gate {

class ESKF {
public:
  ESKF(double acc_noise_density, 
       double acc_noise_density_multiplier,
       Eigen::Vector3d p0,
       Eigen::Vector3d v0,
       double t0);

  ~ESKF() = default;

  void feed_prediction(const Eigen::Vector3d& delta_p,
                       const Eigen::Vector3d& delta_v,
                       const double& t);

  void feed_measurement(const std::vector<Eigen::Vector3d>& p_b_i_vecs,
                        const std::vector<Eigen::Matrix3d>& p_b_i_covs,
                        const double& t);
  
  Eigen::Matrix<double, 6, 1> get_state(const double& t) { return fast_propagate(t); };

private:
  void propagate(const double& t);

  void update(const std::vector<Eigen::Vector3d>& p_b_i_vecs,
              const std::vector<Eigen::Matrix3d>& p_b_i_covs);

  Eigen::Matrix<double, 6, 1> fast_propagate(const double& t);

  void remove_old_predictions(const double& t);

  // state
  Eigen::Vector3d p_;
  Eigen::Vector3d v_;
  double t_;

  // predictions
  std::vector<Eigen::Vector3d> delta_p_vecs_;
  std::vector<Eigen::Vector3d> delta_v_vecs_;
  std::vector<double> t_vec_;

  // covariance
  Eigen::Matrix<double, 6, 6> P_;

  // noise
  double a_w_;
};

} // namespace gate


#endif  // ESKF_H_