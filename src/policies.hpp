#ifndef INCLUDE_POLICIES_H
#define INCLUDE_POLICIES_H

#include <Eigen/Dense>
#include <random>

class policy_base {
public:

  virtual void reset() = 0;

  virtual int sample_arm(std::mt19937& rng) const = 0;

  virtual double get_prob(int arm) const = 0;

  virtual Eigen::VectorXd get_grad(int arm) const = 0;

  virtual void set_params(const Eigen::VectorXd& params) = 0;

  virtual Eigen::VectorXd get_params() const = 0;

  virtual int max_arm() const = 0;

};


class gibbs_policy : public policy_base {

public:

  gibbs_policy(int num_arms) 
    : num_arms(num_arms), prefs(Eigen::VectorXd::Zero(num_arms)) {}

  virtual void reset() override {
    prefs = Eigen::VectorXd::Zero(num_arms);
  }

  virtual int sample_arm(std::mt19937& rng) const override {
    double x = std::uniform_real_distribution<>(0, norm_const())(rng);
    double c = 0;
    for (int arm = 0; arm < num_arms; ++arm) {
      c += unnorm_prob(arm);
      if (c >= x) return arm;
    }
    return -1;
  }

  virtual double get_prob(int arm) const override {
    return unnorm_prob(arm) / norm_const();
  }

  virtual Eigen::VectorXd get_grad(int arm) const override {
    Eigen::VectorXd grad(num_arms);
    double c = norm_const(); // compute this only once
    double arm_prob = unnorm_prob(arm) / c;
    for (int i = 0; i < num_arms; ++i) {
      grad(i) = (i==arm ? arm_prob : 0) - arm_prob*unnorm_prob(i)/c;
    }
    return grad;
  }

  virtual void set_params(const Eigen::VectorXd& params) override {
    prefs = params;
  }

  virtual Eigen::VectorXd get_params() const override {
    return prefs;
  }

  virtual int max_arm() const override {
    return num_arms;
  }

private:

  double unnorm_prob(int arm) const {
    return exp(prefs(arm));
  }

  double norm_const() const {
    double c = 0;
    for (int i = 0; i < num_arms; ++i) c += unnorm_prob(i);
    return c;
  }

  int num_arms;
  Eigen::VectorXd prefs;
};

#endif
