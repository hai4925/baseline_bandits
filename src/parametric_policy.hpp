#ifndef INCLUDE_PARAMETRIC_POLICY_H
#define INCLUDE_PARAMETRIC_POLICY_H

#include <Eigen/Dense>
#include <random>

class parametric_policy_base {
public:

	virtual int sample_arm(std::mt19937& rng) const = 0;

	virtual double get_prob(int arm) const = 0;

	virtual Eigen::VectorXd get_grad(int arm) const = 0;

	virtual void set_params(const Eigen::VectorXd& params) = 0;

	virtual Eigen::VectorXd get_params() const = 0;

};

#endif
