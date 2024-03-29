#ifndef INCLUDE_BASELINE_H
#define INCLUDE_BASELINE_H

#include <boost/smart_ptr.hpp>
#include <vector>

#include "policies.hpp"
#include "value_estimates.hpp"

/* Baseline base class */
class baseline_base {

public:

  virtual ~baseline_base() {};
  virtual void reset(const bandit& bandit) {};
  virtual void update(int arm, double reward) {};
  virtual double get_value() = 0;

};


/* The baseline that always returns zero */
class zero_baseline : public baseline_base {
  
public:

  virtual double get_value() {
    return 0;
  }

};


/* The state-value function of the current policy */
class value_baseline : public baseline_base {

public:

  value_baseline(boost::shared_ptr<policy_base> policy,
                 boost::shared_ptr<valest_base> valest)
    : policy(policy), valest(valest) {}

  virtual void reset(const bandit& bandit) {
    valest->reset(bandit);
  }

  virtual void update(int arm, double reward) {
    valest->update(arm, reward);
  }

  virtual double get_value() {
    double b = 0;
    for (int arm = 0; arm < policy->max_arm(); ++arm) {
      b += policy->get_prob(arm) * valest->get_value(arm);
    }
    return b;
  }

private:

  boost::shared_ptr<policy_base> policy;
  boost::shared_ptr<valest_base> valest;
  
};


/* Baseline that minimizes the trace of the covariance matrix */
class trcov_baseline : public baseline_base {

public:

  trcov_baseline(boost::shared_ptr<policy_base> policy,
                 boost::shared_ptr<valest_base> valest)
    : policy(policy), valest(valest) {}

  virtual void reset(const bandit& bandit) {
    valest->reset(bandit);
  }

  virtual void update(int arm, double reward) {
    valest->update(arm, reward);
  }

  virtual double get_value() {    
    // Compute unnormalized weights and normalizing constant
    std::vector<double> weights(policy->max_arm());
    double total_weights = 0;
    for (int arm = 0; arm < policy->max_arm(); ++arm) {
      weights[arm] = policy->get_grad(arm).squaredNorm() / policy->get_prob(arm);
      total_weights += weights[arm];
    }

    // Compute baseline as a weighted average of action values
    double b = 0;
    for (int arm = 0; arm < policy->max_arm(); ++arm) {
      b += weights[arm] / total_weights * valest->get_value(arm);
    }

    return b;
  }

private:

  boost::shared_ptr<policy_base> policy;
  boost::shared_ptr<valest_base> valest;

};


class trcov_baseline_grad : public baseline_base {

public:

  trcov_baseline_grad(boost::shared_ptr<policy_base> policy,
                      boost::shared_ptr<valest_base> valest,
                      double step_size)
    : policy(policy), valest(valest), step_size(step_size), b(0) {}

  virtual void reset(const bandit& bandit) {
    valest->reset(bandit);
    b = 0;
  }

  virtual void update(int arm, double reward) {
    valest->update(arm, reward);
    Eigen::VectorXd grad = policy->get_grad(arm);
    double prob = policy->get_prob(arm);
    b = b + step_size*(valest->get_value(arm) - b)*grad.squaredNorm()/prob/prob;
  }

  virtual double get_value() { return b; }
  
private:

  boost::shared_ptr<policy_base> policy;
  boost::shared_ptr<valest_base> valest;
  double step_size;
  double b;

};


class naive_baseline_grad : public baseline_base {

public:

  naive_baseline_grad(boost::shared_ptr<valest_base> valest, double step_size)
    : valest(valest), step_size(step_size), b(0) {}

  virtual void reset(const bandit& bandit) {
    valest->reset(bandit);
    b = 0;
  }

  virtual void update(int arm, double reward) {
    valest->update(arm, reward);
    b = b + step_size*(valest->get_value(arm) - b);
  }

  virtual double get_value() { return b; }
  
private:

  boost::shared_ptr<valest_base> valest;
  double step_size;
  double b;

};

#endif
