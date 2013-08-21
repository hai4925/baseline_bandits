#ifndef INCLUDE_POLICY_GRADIENT_H
#define INCLUDE_POLICY_GRADIENT_H

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <vector>

#include "bandit.hpp"
#include "policies.hpp"

class policy_gradient_agent {

public:

  policy_gradient_agent(boost::shared_ptr<policy_base> policy, 
                        boost::shared_ptr<valest_base> valest,
                        boost::shared_ptr<baseline_base> baseline,
                        double step_size)
    : policy(policy), valest(valest), baseline(baseline), step_size(step_size) {}


  int get_arm(boost::random::mt19937& rng) { return policy->sample_arm(rng); }

  void reset(const bandit& bandit) {
    policy->reset();
    valest->reset(bandit);
    baseline->reset(bandit);
  }

  void update(int arm, double reward) {
    valest->update(arm, reward);
    baseline->update(arm, reward);
    Eigen::VectorXd grad = policy->get_grad(arm) / policy->get_prob(arm)
      * (valest->get_value(arm) - baseline->get_value());
    policy->set_params(policy->get_params() + step_size*grad);
  }

private:

  boost::shared_ptr<policy_base> policy;
  boost::shared_ptr<valest_base> valest;
  boost::shared_ptr<baseline_base> baseline;
  double step_size;

};

#endif
