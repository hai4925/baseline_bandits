#ifndef INCLUDE_VALUE_ESTIMATES_H
#define INCLUDE_VALUE_ESTIMATES_H

#include <vector>

/** Base class for value estimation classes */
struct valest_base {
	virtual void reset() = 0;
	virtual double get_value(int arm) const = 0;
	virtual void update(int arm, double reward) = 0;
};


/** This estimator is used when we know the true values of the
 * arms. */
class valest_known : public valest_base {
public:

	valest_known(const std::vector<double>& values) 
		: arm_values(values) {}

	virtual void reset() override {}

	double get_value(int arm) const override { 
		return arm_values[arm]; 
	}

	void update(int arm, double reward) override {}

private:
	std::vector<double> arm_values;
};


/** This estimator uses the last observed reward as an estimate of an
 * arm's value. */
class valest_last : public valest_base {
public:

	valest_last(int num_arms, double default_value=0) 
		: default_value(default_value),
			last_reward(num_arms, default_value) {}

	void reset() override {
		for (auto& lr : last_reward) lr = default_value;
	}

	double get_value(int arm) const override { 
		return last_reward[arm]; 
	}

	void update(int arm, double reward) override { 
		last_reward[arm] = reward; 
	}

private:
	double default_value;
	std::vector<double> last_reward;
};


/** This estimator uses the average of observed rewards as an estimate
 * of an arms value */
class valest_avg : public valest_base {
public:

	valest_avg(int num_arms, double default_value=0)
		:total_reward(num_arms, 0), num_pulls(num_arms, 0) {}

	void reset() override {
		for (int i = 0; i < total_reward.size(); ++i) {
			total_reward[i] = 0;
			num_pulls[i] = 0;
		}
	}

	double get_value(int arm) const override {
		if (num_pulls[arm] == 0) return default_value;
		else return total_reward[arm] / num_pulls[arm];
	}

	void update(int arm, double reward) override {
		total_reward[arm] += reward;
		num_pulls[arm] += 1;
	}
	
private:
	double default_value;
	std::vector<double> total_reward;
	std::vector<int> num_pulls;
};

#endif
