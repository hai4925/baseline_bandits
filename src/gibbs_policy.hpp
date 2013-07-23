#ifndef INCLUDE_GIBBS_POILCY_H
#define INCLUDE_GIBBS_POLICY_H

#include <Eigen/Dense>
#include <random>

#include "parametric_policy.hpp"

class gibbs_policy : public parametric_policy_base {

public:

	gibbs_policy(int num_arms) 
		: num_arms(num_arms), prefs(Eigen::VectorXd::Zero(num_arms)) {}

	virtual int sample_arm(std::mt19937& rng) const override;

	virtual double get_prob(int arm) const override;

	virtual Eigen::VectorXd get_grad(int arm) const override;

	virtual void set_params(const Eigen::VectorXd& params) override;

	virtual Eigen::VectorXd get_params() const override;

private:

	double norm_const() const;
	double unnorm_prob(int arm) const;

	int num_arms;
	Eigen::VectorXd prefs;
};

#endif
