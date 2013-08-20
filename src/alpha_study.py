import cPickle
from datetime import datetime
import math
import numpy as np
from multiprocessing import Pool
import sys

from run_experiment import run_experiment
from param_study import ParameterSettings

parameters = ParameterSettings({
    "num_arms": [10],
    "num_pulls": [20],
    "num_runs": [10000],
    "step_size": np.logspace(0,2,20, base=2),
    "baseline": ["zero", "trcov", "value"]
})

def try_parameters(kwargs):
    start_time = datetime.now()
    data = run_experiment(**kwargs)
    trial_means = data.mean(axis=1)
    trial_mean = trial_means.mean()
    trial_sigma = trial_means.std()
    trial_std_err = trial_sigma / math.sqrt(kwargs["num_runs"])
    time_elapsed = datetime.now() - start_time

    print "baseline = %s, step_size = %g -- %s" % (kwargs["baseline"], kwargs["step_size"], time_elapsed) 

    return (trial_mean, trial_std_err)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "torque": 
            i = int(sys.argv[2])
            result = try_parameters(parameters.get(i))
            cPickle.dump(result, file("result_%d.pkl" % i, "w"))
        elif sys.argv[1] == "gather":
            results = [cPickle.load(file("result_%d.pkl" % i, "r")) 
                       for i in parameters.list_indices()]
    else:
        pool = Pool(2)
        results = pool.map(try_parameters, parameters.all())

















