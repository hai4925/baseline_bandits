from experiment import experiment
from experiment_runner import main
import math

def runner(**args):
    data = experiment(**args)
    means = data.mean(axis=1)
    mean = means.mean()
    sigma = means.std()
    std_err = sigma / math.sqrt(args["num_runs"])
    return {"mean": mean, "std_err": std_err}

if __name__ == "__main__": main(runner)
