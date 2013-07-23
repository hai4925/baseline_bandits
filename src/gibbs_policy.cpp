#include "gibbs_policy.hpp"


using namespace Eigen;
using namespace std;


int gibbs_policy::sample_arm(mt19937& rng) const {
	double x = uniform_real_distribution<>(0,norm_const())(rng);
	double c = 0;
	for (int arm = 0; arm < num_arms; ++arm) {
		c += unnorm_prob(arm);
		if (c >= x) return arm;
	}
	return -1;
}


double gibbs_policy::get_prob(int arm) const {
	return unnorm_prob(arm) / norm_const();
}


VectorXd gibbs_policy::get_grad(int arm) const {
	VectorXd grad(num_arms);
	double c = norm_const();  // Only compute normalizing constant once
	double arm_prob = unnorm_prob(arm) / c;
	for (int i = 0; i < num_arms; ++i) {
		grad(i) = (i==arm ? arm_prob : 0) - arm_prob*unnorm_prob(i)/c;
	}
	return grad;
}


void gibbs_policy::set_params(const VectorXd& params) {
	prefs = params;
}


VectorXd gibbs_policy::get_params() const {
	return prefs;
}


double gibbs_policy::norm_const() const {
	double c = 0;
	for (int i = 0; i < num_arms; ++i) c += exp(prefs(i));
	return c;
}


double gibbs_policy::unnorm_prob(int arm) const {
	return exp(prefs(arm));
}
