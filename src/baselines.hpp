#ifndef INCLUDE_BASELINE_H
#define INCLUDE_BASELINE_H

#include <memory>
#include <vector>

#include "policies.hpp"
#include "value_estimates.hpp"

/* Baseline base class */
class baseline_base {

public:

  virtual double get_value(std::shared_ptr<const policy_base> policy, 
                           std::shared_ptr<const valest_base> values) = 0;

};


/* The baseline that always returns zero */
class zero_baseline : public baseline_base {
  
public:
  
  virtual double get_value(std::shared_ptr<const policy_base> policy, 
                           std::shared_ptr<const valest_base> values) override {
    return 0;
  }

};


/* The state-value function of the current policy */
class value_baseline : public baseline_base {

public:

  virtual double get_value(std::shared_ptr<const policy_base> policy, 
                           std::shared_ptr<const valest_base> values) override {

    double b = 0;
    for (int arm = 0; arm < policy->max_arm(); ++arm) {
      b += policy->get_prob(arm) * values->get_value(arm);
    }
    return b;
  }
  
};


/* Baseline that minimizes the trace of the covariance matrix */
class trcov_baseline : public baseline_base {

public:

  virtual double get_value(std::shared_ptr<const policy_base> policy, 
                           std::shared_ptr<const valest_base> values) override {
    
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
      b += weights[arm] / total_weights * values->get_value(arm);
    }

    return b;
  }

};

#endif
