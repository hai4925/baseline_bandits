#ifndef INCLUDE_BANDIT_H
#define INCLUDE_BANDIT_H
#include <random>
#include <vector>

class bandit {

public:

	typedef std::normal_distribution<double> dist_type;

	bandit(const std::vector<dist_type>& dists) 
		: arm_dists(dists) {}
	
	int num_arms() const { return arm_dists.size(); }
	
	double pull_arm(std::mt19937& rng, int arm) {
		return arm_dists[arm](rng);
	}

private:

	std::vector<dist_type> arm_dists;

};

#endif
