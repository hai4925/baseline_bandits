#ifndef INCLUDE_BANDIT_H
#define INCLUDE_BANDIT_H

#include <random>
#include <vector>

class bandit {

public:

  typedef std::normal_distribution<double> dist_type;

  bandit(std::mt19937& rng, int num_arms) {
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
    for (auto& dist : arm_dists) means.push_back(dist.mean());
    return means;
  }

  double pull_arm(std::mt19937& rng, int arm) {
    return arm_dists[arm](rng);
  }

private:

  std::vector<dist_type> arm_dists;

};

#endif
