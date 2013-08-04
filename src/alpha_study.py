from datetime import datetime
from itertools import izip, repeat
import math
from multiprocessing import Pool
import numpy as np
from run_experiment import run_experiment

num_arms = 10
num_pulls = 20
num_runs = 2000000
alphas = np.logspace(0,2,20, base=2)

baselines = ["zero", "value", "trcov"]

def try_parameters((alpha,baseline)):
    trial_start = datetime.now();
    data = run_experiment(step_size=alpha, baseline=baseline, num_arms=num_arms, num_pulls=num_pulls, num_runs=num_runs)
    trial_means = data.mean(axis=1)
    trial_mean = trial_means.mean()
    trial_sigma = trial_means.std()
    trial_std_err = trial_sigma / math.sqrt(num_runs)
    print "  alpha=%g -- " % (alpha), str(datetime.now() - trial_start)
    return (trial_mean, trial_std_err)

if __name__ == "__main__":
    pool = Pool(2)
    results = dict()
    for baseline in baselines:
        print "Starting baseline='%s'" % baseline
        results[baseline] = []
        results[baseline].append(pool.map(try_parameters, izip(alphas, repeat(baseline))))
