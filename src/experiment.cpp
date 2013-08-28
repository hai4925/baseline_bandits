#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>

#include "bandit.hpp"
#include "baselines.hpp"
#include "policies.hpp"
#include "policy_gradient.hpp"
#include "value_estimates.hpp"

using std::string;
using std::cout;
using boost::shared_ptr;
using boost::random::mt19937;
namespace po = boost::program_options;

shared_ptr<valest_base> make_valest(const string& name) {
  if (name == "known") return shared_ptr<valest_base>(new valest_known());
  else if (name == "last")  return shared_ptr<valest_base>(new valest_last());
  else if (name == "avg")   return shared_ptr<valest_base>(new valest_avg());
  throw "bad valest name!";
}

shared_ptr<baseline_base> make_baseline(const string& name, shared_ptr<policy_base> policy,
                   shared_ptr<valest_base> valest, double step_size) {
  if (name == "zero") 
    return shared_ptr<baseline_base>(new zero_baseline());
  else if (name == "value") 
    return shared_ptr<baseline_base>(new value_baseline(policy, valest));
  else if (name == "trcov") 
    return shared_ptr<baseline_base>(new trcov_baseline(policy, valest));  
  else if (name == "trcov_grad") 
    return shared_ptr<baseline_base>(new trcov_baseline_grad(policy, valest, step_size));
  else if (name == "naive_grad")
    return shared_ptr<baseline_base>(new naive_baseline_grad(valest, step_size));
  throw "bad baseline name!";
}

int main(int argc, char *argv[]) {
  // Get the parameters from the command line
  po::options_description desc("Experiment options");
  desc.add_options()
    ("help,h", "produce this help message.")
    ("value_estimate", 
     po::value<string>()->default_value("last"),
     "Which value estimate to use. Must be one of 'last', 'avg', or 'known'.")
    ("baseline", 
     po::value<string>()->default_value("zero"), 
     "Which baseline to use. Must be one of 'zero', 'value', or 'trcov'.")
    ("baseline_value_estimate", 
     po::value<string>()->default_value("avg"), 
     "Which value estimate to use in the baseline. Must be one of 'last', 'avg', or 'known'.")
    ("baseline_stepsize",
     po::value<double>()->default_value(0.1),
     "Stepsize for baseline, if it uses one.")
    ("stepsize,s", 
     po::value<double>()->default_value(0.1), 
     "The stepsize parameter (alpha).")
    ("num_arms,a",
     po::value<int>()->default_value(10),
     "Number of arms in the testbed.")
    ("num_runs,r", 
     po::value<int>()->default_value(10000), 
     "Number of runs for the experiment.")
    ("num_pulls,p", 
     po::value<int>()->default_value(200), 
     "Number of pulls per run.")
    ("seed,S", 
     po::value<int>()->default_value(0), 
     "The random seed.")
    ("bandit_seed",
     po::value<int>()->default_value(1),
     "The random seed used to generate bandits.")
    ("arm_mean",
     po::value<double>()->default_value(0),
     "The expected arm mean in the bandits.")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
  }
  // Show help if needed
  catch (...){
    cout << desc << "\n";
    return 1;
  }
  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  string valest_name = vm["value_estimate"].as<string>();
  string baseline_name = vm["baseline"].as<string>();
  string baseline_valest_name = vm["baseline_value_estimate"].as<string>();
  double baseline_step_size = vm["baseline_stepsize"].as<double>();
  double step_size = vm["stepsize"].as<double>();
  int num_arms = vm["num_arms"].as<int>();
  int num_runs = vm["num_runs"].as<int>();
  int num_pulls = vm["num_pulls"].as<int>();
  int seed = vm["seed"].as<int>();
  int bandit_seed = vm["bandit_seed"].as<int>();
  double arm_mean = vm["arm_mean"].as<double>();


  // Create random seed for agents
  mt19937 rng(seed);

  // Create random seed for generating bandits
  mt19937 bandit_rng(bandit_seed);

  // Create policy, value estimators, and the baseline
  shared_ptr<policy_base>   policy(new gibbs_policy(num_arms));
  shared_ptr<valest_base>   valest = make_valest(valest_name);
  shared_ptr<valest_base>   baseline_valest = make_valest(baseline_valest_name);
  shared_ptr<baseline_base> baseline = make_baseline(baseline_name, policy, baseline_valest, baseline_step_size);

  policy_gradient_agent pg_agent(policy, valest, baseline, step_size);

  // Run the experiment
  for (int i = 0; i < num_runs; ++i) {
    bandit b(bandit_rng, num_arms, arm_mean);
    pg_agent.reset(b);
    for (int pull = 0; pull < num_pulls; ++pull) {
      int arm = pg_agent.get_arm(rng);
      double reward = b.pull_arm(rng, arm);
      pg_agent.update(arm, reward);
      printf("%g ", reward);
    }
    printf("\n");
  }

}
