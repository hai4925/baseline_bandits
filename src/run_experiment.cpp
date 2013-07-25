#include <Boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "bandit.hpp"
#include "baselines.hpp"
#include "policies.hpp"
#include "policy_gradient.hpp"
#include "value_estimates.hpp"

using namespace std;
namespace po = boost::program_options;


shared_ptr<valest_base> make_valest(const string& name) {
  if (name == "known") return shared_ptr<valest_base>(new valest_known());
  else if (name == "last")  return shared_ptr<valest_base>(new valest_last());
  else if (name == "avg")   return shared_ptr<valest_base>(new valest_avg());
  throw "bad valest name!";
}


shared_ptr<baseline_base> make_baseline(const string& name) {  
  if (name == "zero") return shared_ptr<baseline_base>(new zero_baseline());
  else if (name == "value") return shared_ptr<baseline_base>(new value_baseline());
  else if (name == "trcov") return shared_ptr<baseline_base>(new trcov_baseline());  
  throw "bad baseline name!";
}


int main(int argc, char const *argv[])
{
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
  double alpha = vm["stepsize"].as<double>();
  int num_arms = vm["num_arms"].as<int>();
  int num_runs = vm["num_runs"].as<int>();
  int num_pulls = vm["num_pulls"].as<int>();
  int seed = vm["seed"].as<int>();

  // Create random seed
  mt19937 rng(seed);

  // Generate the testbed
  vector<bandit> bandits;
  for (int i = 0; i < num_runs; ++i) bandits.push_back(bandit(rng, num_arms));

  // Create policy, value estimators, and the baseline
  shared_ptr<policy_base>   policy(new gibbs_policy(num_arms));
  shared_ptr<valest_base>   valest = make_valest(valest_name);
  shared_ptr<baseline_base> baseline = make_baseline(baseline_name);
  shared_ptr<valest_base>   baseline_valest = make_valest(baseline_valest_name);

  // Run the experiment
  for (bandit& b : bandits) {
    policy->reset();
    valest->reset(b);
    baseline_valest->reset(b);
    for (int pull = 0; pull < num_pulls; ++pull) {
      cout << pg_step(rng, alpha, b, policy, valest, baseline, baseline_valest);
      cout << " ";
    }
    cout << "\n";
  }

}
