import numpy as np
from cStringIO import StringIO
import subprocess

def run_experiment(value="last", baseline="zero", baseline_value="avg",
                   baseline_step_size=0.1, step_size=0.1, num_arms=10, num_runs=10000, 
                   num_pulls=200, seed=0):
    arguments = ["./run_experiment",
                 "--value_estimate", value,
                 "--baseline", baseline,
                 "--baseline_value_estimate", baseline_value,
                 "--baseline_stepsize", str(baseline_step_size),
                 "--stepsize", str(step_size),
                 "--num_arms", str(num_arms),
                 "--num_runs", str(num_runs),
                 "--num_pulls", str(num_pulls),
                 "--seed", str(seed)]
    
    result = StringIO(subprocess.check_output(arguments))
    return np.loadtxt(result)













