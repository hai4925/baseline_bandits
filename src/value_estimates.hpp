#ifndef INCLUDE_VALUE_ESTIMATES_H
#define INCLUDE_VALUE_ESTIMATES_H

#include <vector>

/** Base class for value estimation classes */
struct valest_base {
  virtual ~valest_base() {}
  virtual void reset(const bandit& bandit) = 0;
  virtual double get_value(int arm) const = 0;
  virtual void update(int arm, double reward) = 0;
};


/** This estimator is used when we know the true values of the
 * arms. */
class valest_known : public valest_base {
public:

  virtual void reset(const bandit& bandit) {
    arm_values = bandit.arm_means();
  }

  virtual double get_value(int arm) const {
    return arm_values[arm]; 
  }

  virtual void update(int arm, double reward) {}

private:
  std::vector<double> arm_values;
};


/** This estimator uses the last observed reward as an estimate of an
 * arm's value. */
class valest_last : public valest_base {
public:

  valest_last(double default_value=0) 
    : default_value(default_value) {}

  virtual void reset(const  bandit& bandit) {
    last_reward = std::vector<double>(bandit.num_arms(), default_value);
  }

  virtual double get_value(int arm) const {
    return last_reward[arm]; 
  }

  virtual void update(int arm, double reward) { 
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

  valest_avg(double default_value=0)
    : default_value(default_value) {}

  virtual void reset(const bandit& bandit) {
    total_reward = std::vector<double>(bandit.num_arms(), 0);
    num_pulls = std::vector<int>(bandit.num_arms(), 0);
  }

  virtual double get_value(int arm) const {
    if (num_pulls[arm] == 0) return default_value;
    else return total_reward[arm] / num_pulls[arm];
  }

  virtual void update(int arm, double reward) {
    total_reward[arm] += reward;
    num_pulls[arm] += 1;
  }
  
private:
  double default_value;
  std::vector<double> total_reward;
  std::vector<int> num_pulls;
};

#endif
