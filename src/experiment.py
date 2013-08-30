from cStringIO import StringIO
import math
import numpy as np
from os import path
import subprocess
import sys

def experiment(value="last", baseline="zero", baseline_value="avg",
               baseline_step_size=0.1, step_size=0.1, num_arms=10, num_runs=10000, 
               num_pulls=200, seed=0, bandit_seed=1, arm_mean=0):

    arguments = [path.join(path.dirname(__file__), "experiment"),
                 "--value_estimate", value,
                 "--baseline", baseline,
                 "--baseline_value_estimate", baseline_value,
                 "--baseline_stepsize", str(baseline_step_size),
                 "--stepsize", str(step_size),
                 "--num_arms", str(num_arms),
                 "--num_runs", str(num_runs),
                 "--num_pulls", str(num_pulls),
                 "--seed", str(seed),
                 "--bandit_seed", str(bandit_seed),
                 "--arm_mean", str(arm_mean)]
    
    result = StringIO(subprocess.check_output(arguments))
    return np.loadtxt(result)


def parameter_study(**kwargs):
    data = experiment(**kwargs)
    means = data.mean(axis=1)
    mean = means.mean()
    sigma = means.std()
    std_err = sigma / math.sqrt(kwargs["num_runs"])
    return {"mean": mean, "std_err": std_err}


def learning_curve(**kwargs):
    data = experiment(**kwargs)
    means = data.mean(axis=0)
    sigmas = data.std(axis=0)
    std_errs = sigmas / math.sqrt(kwargs["num_runs"])
    return {"means": means.tolist(), "std_errs": std_errs.tolist()}
    













