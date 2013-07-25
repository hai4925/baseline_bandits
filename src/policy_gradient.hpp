#ifndef INCLUDE_POLICY_GRADIENT_H
#define INCLUDE_POLICY_GRADIENT_H

#include <Eigen/Dense>
#include <random>
#include <vector>

#include "bandit.hpp"
#include "policies.hpp"

double pg_step(std::mt19937& rng, double alpha, bandit& bandit,
               std::shared_ptr<policy_base> policy, std::shared_ptr<valest_base> valest,
               std::shared_ptr<baseline_base> baseline, std::shared_ptr<valest_base> baseline_valest) {

  // Sample an arm and corresponding reward
  int arm = policy->sample_arm(rng);
  double reward = bandit.pull_arm(rng, arm);
  // Update value estimates for the arms
  valest->update(arm, reward);
  baseline_valest->update(arm, reward);
  // Compute baseline value
  double baseline_value = baseline->get_value(policy, baseline_valest);
  // Compute gradient approximation
  Eigen::VectorXd grad = policy->get_grad(arm) / policy->get_prob(arm)
    * (valest->get_value(arm) - baseline_value);
  // Update policy parameters
  policy->set_params(policy->get_params() + alpha*grad);
  // Return the reward
  return reward;

}

#endif
