#ifndef INCLUDE_VALUE_ESTIMATES_H
#define INCLUDE_VALUE_ESTIMATES_H

#include <vector>

/** Base class for value estimation classes */
struct value_estimate_base {
	virtual double get_value(int arm) const = 0;
	virtual void update(int arm, double reward) = 0;
};


/** This estimator is used when we know the true values of the
 * arms. */
class valest_known : public value_estimate_base {
public:

	valest_known(const std::vector<double>& values) 
		: arm_values(values) {}

	double get_value(int arm) const { return arm_values[arm]; }

	void update(int arm, double reward) {}

private:
	std::vector<double> arm_values;
};


/** This estimator uses the last observed reward as an estimate of an
 * arm's value. */
class valest_last : public value_estimate_base {
public:

	valest_last(int num_arms, double default_value=0) 
		: last_reward(num_arms, default_value) {}

	double get_value(int arm) const { return last_reward[arm]; }

	void update(int arm, double reward) { last_reward[arm] = reward; }

private:
	std::vector<double> last_reward;
};


/** This estimator uses the average of observed rewards as an estimate
 * of an arms value */
class valest_avg : public value_estimate_base {
public:

	valest_avg(int num_arms, double default_value=0)
		:total_reward(num_arms, 0), num_pulls(num_arms, 0) {}

	double get_value(int arm) const {
		if (num_pulls[arm] == 0) return default_value;
		else return total_reward[arm] / num_pulls[arm];
	}

	void update(int arm, double reward) {
		total_reward[arm] += reward;
		num_pulls[arm] += 1;
	}
	
private:
	double default_value;
	std::vector<double> total_reward;
	std::vector<int> num_pulls;
};

#endif
