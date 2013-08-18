import math
import numpy as np
from multiprocessing import Pool

from run_experiment import run_experiment
from param_study import ParameterSettings

parameters = ParameterSettings({
    "num_arms": [10],
    "num_pulls": [20],
    "num_runs": [1000],
    "step_size": np.logspace(0,2,20, base=2),
    "baseline": ["zero", "trcov", "value"]
})

def try_parameters(kwargs):
    data = run_experiment(**kwargs)
    trial_means = data.mean(axis=1)
    trial_mean = trial_means.mean()
    trial_sigma = trial_means.std()
    trial_std_err = trial_sigma / math.sqrt(kwargs["num_runs"])
    print kwargs["baseline"] + ", step_size=" + str(kwargs["step_size"]) + ": " + str(trial_mean)
    return (trial_mean, trial_std_err)

if __name__ == "__main__":
    pool = Pool(2)
    results = pool.map(try_parameters, parameters.all())














