#ifndef INCLUDE_BANDIT_H
#define INCLUDE_BANDIT_H

#include <boost/random.hpp>
#include <vector>

class bandit {

public:

  typedef boost::random::normal_distribution<double> dist_type;

  bandit(boost::random::mt19937& rng, int num_arms) {
    dist_type mean_dist(0,1);
    for (int i = 0; i < num_arms; ++i) {
      double mean = mean_dist(rng);
      arm_dists.push_back(dist_type(mean, 1));
    }
  }

  void set_arms(const std::vector<dist_type>& arms) {
    arm_dists = arms;
  }

  int num_arms() const { return arm_dists.size(); }

  std::vector<double> arm_means() const {
    std::vector<double> means;
    for (int i = 0; i < arm_dists.size(); ++i)
      means.push_back(arm_dists[i].mean());
    return means;
  }

  double pull_arm(boost::random::mt19937& rng, int arm) {
    return arm_dists[arm](rng);
  }

private:

  std::vector<dist_type> arm_dists;

};

#endif
