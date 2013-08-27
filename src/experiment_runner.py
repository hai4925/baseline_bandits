from itertools import ifilter, imap, izip
import json
import multiprocessing
import operator
from os import path
import re
from subprocess import Popen, PIPE
import sys
from types import FunctionType, DictType

################################################################################
# Main function
################################################################################

def main(experiment):
    config_path = sys.argv[1]
    out_dir = path.dirname(config_path)
    param_index, options = load_config(config_path)
    
    modes = {
        "local": run_local,
        "torque": run_torque
    }
    mode = sys.argv[2]
    modes[mode](experiment, out_dir, param_index, options)
    
################################################################################
# Running modes
################################################################################

# ----- Run the experiment locally ----- #

class ArgWrapper:
    def __init__(self, f): self.f = f
    def __call__(self, args): return self.f(**args)

def run_local(experiment, out_dir, param_index, options):

    num_procs = int(sys.argv[3]) if len(sys.argv) >= 4 else 1
    if num_procs == 1:
        results = map(ArgWrapper(experiment), param_index.all())
    else:
        pool = multiprocessing.Pool(num_procs)
        results = pool.map(ArgWrapper(experiment), param_index.all())
    save_data(path.join(out_dir, "results.json"), results)

# ----- Run the experiment using the TORQUE resource manager ----- #

def run_torque(experiment, out_dir, param_index, options):

    if len(sys.argv) == 3: # Request that all unfinished jobs be scheduled
        # Get options
        if options.has_key("__resource_limits__"):
            resource_limits = options["__resource_limits__"]
        else:
            resource_limits = ""
        # First, calculate which jobs need to run
        def job_needed(i):
            job_path = path.join(out_dir, "result_%d.json"%i)
            return not path.exists(job_path)
        needed_jobs = map(job_needed, param_index.all_ixs())
        array_request = ",".join(map(range_to_str, true_ranges(needed_jobs)))
        # Next, submit the jobs to TORQUE using the qsub command
        bash_script = \
        """
        #!/bin/bash
        #PBS -l %(resource_limits)s
        cd $PBS_O_WORKDIR
        python %(script_path)s %(config_path)s torque job $PBS_ARRAYID
        """ % {
            "resource_limits": resource_limits,
            "script_path": sys.argv[0],
            "config_path": sys.argv[1]
        }
        print "Scheduling jobs: " + array_request
        p = Popen(["qsub", "-t", array_request, "-"], stdout=PIPE, stdin=PIPE)
        response = p.communicate(input=bash_script)
        print "qsub stdout:\n%s\nqsub stderr:\n%s" % response
        p.wait()

    elif sys.argv[3] == "job": # Run a specific job
        job_number = int(sys.argv[4]) - 1
        result = ArgWrapper(experiment)(param_index.get(job_number))
        save_data(path.join(out_dir, "result_%d.json"%job_number), result)

    elif sys.argv[3] == "gather": # Collect individual runs
        try:
            def load(i):
                return load_data(path.join(out_dir, "result_%d.json"%i))
            results = [load(i) for i in param_index.all_ixs()]
            save_data(path.join(out_dir, "results.json"), results)
        except:
            print "Not all jobs are done!"

def true_ranges(l):
    """ Given a list of booleans, returns a list of the true intervals. """
    l.append(False)
    ranges = []
    start = 0
    while start < len(l)-1:
        while start < len(l)-1 and l[start] == False: start += 1
        end = start
        while end < len(l) and l[end] == True: end += 1
        if end > start: ranges.append((start, end))
        start = end
    l.pop()
    return ranges

def range_to_str((start,end)):
    if end - start == 1: 
        return str(start+1)
    else: 
        return str(start+1)+"-"+str(end)

################################################################################
# Reading and writing to files
################################################################################

def load_config(path):
    config = load_data(path)
    params = dict()
    options = dict()
    for (k,v) in config.iteritems():
        if re.match("__.*__", k): 
            options[k] = v
        else: 
            params[k] = v
    return (ParameterIndex(params), options)

# these functions can be modified in case you don't want to use json

def save_data(out_path, data):
    json.dump(data, file(out_path, "w"), sort_keys=True, indent=2,
               separators=(",", ": "))


def load_data(in_path):
    return json.load(file(in_path, "r"))

################################################################################
# ParameterIndex
################################################################################

class ParameterIndex:
    
    def __init__(self, param_config):
        self.param_config = param_config
        self.names = sorted(param_config.keys())
        self.values = [param_config[name] for name in self.names]
        self.n_values = [len(vs) for vs in self.values]
        self.n_settings = reduce(operator.mul, self.n_values)
    
    def values(self, param): return self.param_config["param"]

    def num_settings(self): return self.n_settings

    def get(self, ix):
        params = dict()
        for (name, vs, n) in izip(self.names, self.values, self.n_values):
            params[name] = vs[ix%n]
            ix = int(ix/n)
        return params

    def all_ixs(self, return_list=True): 
        return self.find_ixs(return_list=return_list)

    def all(self, return_list=True): 
        return self.find(return_list=return_list)

    def find_ixs(self, criteria=None, return_list=True):
        def matches(i):
            params = self.get(i)
            if criteria==None: return True
            elif isinstance(criteria, DictType):
                matches = True
                for (k,v) in criteria.iteritems():
                    if not params.has_key(k) or params[k] != v:
                        matches = False
                        break
                return matches
            elif isinstance(criteria, FunctionType): return criteria(params)
        if return_list:
            return filter(matches, xrange(self.num_settings()))
        else:
            return ifilter(matches, xrange(self.num_settings()))

    def find(self, filter=None, return_list=True):
        if return_list: return map(self.get, self.find_ixs(filter))
        else: return imap(self.get, self.find_ixs(filter, False))
        
    
    
