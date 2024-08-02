#include "gate_fusion/ESKF.hpp"
#include <iostream>

using namespace gate;

ESKF::ESKF(double acc_noise_density, 
           double acc_noise_density_multiplier,
           Eigen::Vector3d p0,
           Eigen::Vector3d v0,
           double t0) {
  // Initialize state
  p_ = p0;
  v_ = v0;
  t_ = t0;

  // Initialize covariance
  P_ = Eigen::Matrix<double, 6, 6>::Zero();

  // Initialize noise
  // [m * sqrt(s) / (s^2)]
  a_w_ = acc_noise_density * acc_noise_density_multiplier;
}

void ESKF::feed_prediction(const Eigen::Vector3d& delta_p,
                           const Eigen::Vector3d& delta_v,
                           const double& t) {
  delta_p_vecs_.push_back(delta_p);
  delta_v_vecs_.push_back(delta_v);
  t_vec_.push_back(t);
}

void ESKF::feed_measurement(const std::vector<Eigen::Vector3d>& p_b_i_vecs,
                            const std::vector<Eigen::Matrix3d>& p_b_i_covs,
                            const double& t) {
  ESKF::propagate(t);

  ESKF::update(p_b_i_vecs, p_b_i_covs);

  ESKF::remove_old_predictions(t);
}

void ESKF::propagate(const double& t) {
  if (t < t_) {
    throw std::runtime_error("Cannot propagate backwards in time!");
  } else if (t_vec_.size() < 2) {
    throw std::runtime_error("Cannot propagate without atleast 2 predictions!");
  }

  double total_dt = 0.0;

  for (int i = 0; i < static_cast<int>(t_vec_.size()); i++) {
    if (t_vec_.at(i) < t_) {
      throw std::runtime_error("Predictions out of order!");
    } else if (t_vec_.at(i) <= t) {
      double dt = t_vec_.at(i) - t_;

      p_ = p_ + delta_p_vecs_.at(i);
      v_ = v_ + delta_v_vecs_.at(i);

      t_ = t_vec_.at(i);

      total_dt += dt;
    } else {
      double interval_dt = t_vec_.at(i) - t_vec_.at(i - 1);
      double dt = t - t_;

      p_ = p_ + ((delta_p_vecs_.at(i) - delta_p_vecs_.at(i - 1)) / interval_dt) * dt;
      v_ = v_ + ((delta_v_vecs_.at(i) - delta_v_vecs_.at(i - 1)) / interval_dt) * dt;

      t_ = t;

      total_dt += dt;

      break;
    }
  }

  if (t_ != t) {
    throw std::runtime_error("Not enough prediction to propagate to desired time!");
  }

  Eigen::Matrix<double, 6, 6> F_ = Eigen::Matrix<double, 6, 6>::Identity();
  F_.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * total_dt;

  Eigen::Matrix<double, 6, 6> Q_i_ = Eigen::Matrix<double, 6, 6>::Zero();
  Q_i_.block<3, 3>(0, 0) = 0.5 * a_w_ * a_w_ * total_dt * Eigen::Matrix3d::Identity();
  Q_i_.block<3, 3>(3, 3) = a_w_ * a_w_ * Eigen::Matrix3d::Identity();

  P_ = F_ * P_ * F_.transpose() + F_ * Q_i_ * F_.transpose();
}

void ESKF::update(const std::vector<Eigen::Vector3d>& p_b_i_vecs,
                  const std::vector<Eigen::Matrix3d>& p_b_i_covs) {
  Eigen::Matrix<double, 3, 6> H_i_ = Eigen::Matrix<double, 3, 6>::Zero();
  H_i_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  Eigen::MatrixXd H_(p_b_i_vecs.size() * 3, 6);
  Eigen::MatrixXd V_(p_b_i_vecs.size() * 3, p_b_i_vecs.size() * 3);
  Eigen::MatrixXd h_x_(3 * p_b_i_vecs.size(), 1);
  Eigen::MatrixXd y_(3 * p_b_i_vecs.size(), 1);
  V_.setZero();
  
  for (int i = 0; i < static_cast<int>(p_b_i_vecs.size()); i++) {
    H_.block<3, 6>(3 * i, 0) = H_i_;
    V_.block<3, 3>(3 * i, 3 * i) = p_b_i_covs.at(i);
    h_x_.block<3, 1>(3 * i, 0) = p_;
    y_.block<3, 1>(3 * i, 0) = p_b_i_vecs.at(i);
  }

  Eigen::MatrixXd K_(6, p_b_i_vecs.size() * 3);
  K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + V_).inverse();

  Eigen::Matrix<double, 6, 1> delta_x_ = K_ * (y_ - h_x_);

  P_ = (Eigen::Matrix<double, 6, 6>::Identity() - K_ * H_) * P_;

  p_ = p_ + delta_x_.segment<3>(0);
  v_ = v_ + delta_x_.segment<3>(3);
  std::cout << "ESKF Update! Covariance:\n" << P_ << std::endl;
}

Eigen::Matrix<double, 6, 1> ESKF::fast_propagate(const double& t) {
  Eigen::Vector3d p = p_;
  Eigen::Vector3d v = v_;
  double cur_t = t_;

  for (int i = 0; i < static_cast<int>(t_vec_.size()); i++) {
    if (t_vec_.at(i) <= cur_t) {
      continue;
    } else if (t_vec_.at(i) <= t) {

      p = p + delta_p_vecs_.at(i);
      v = v + delta_v_vecs_.at(i);

      cur_t = t_vec_.at(i);
    } else {
      break;
    }
  }

  Eigen::Matrix<double, 6, 1> state;
  state.segment<3>(0) = p;
  state.segment<3>(3) = v;

  return state;
}

void ESKF::remove_old_predictions(const double& t) {
  while (t_vec_.size() > 0 && t_vec_.at(0) <= t) {
    delta_p_vecs_.erase(delta_p_vecs_.begin());
    delta_v_vecs_.erase(delta_v_vecs_.begin());
    t_vec_.erase(t_vec_.begin());
  }
}