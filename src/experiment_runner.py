from itertools import izip
import json
from multiprocessing import Pool
import operator
from os import path
from subprocess import Popen, PIPE, STDOUT
import sys
from types import FunctionType, DictType

def get_parameters(file_path):
    param_config = json.load(file(file_path,"r"))
    return ParameterProduct(param_config)

def main(runner):
    param_config_path = sys.argv[1]
    
    # Load the configuration file
    param_config = json.load(file(param_config_path,"r"))
    # Get resource limits out of the configuration file
    resource_limits = ""
    if param_config.has_key("__resource_limits__"):
        resource_limits = param_config["__resource_limits__"]
        del param_config["__resource_limits__"]
    parameters = ParameterProduct(param_config)

    # Function for outputting json in a consistent style
    out_dir = path.dirname(param_config_path)
    def write_output(basename, data):
        json.dump(data, file(path.join(out_dir, basename), "w"),
                  sort_keys=True, indent=2, separators=(",",": "))

    run_type = sys.argv[2]

    ############################################################################
    # Run experiment locally
    ############################################################################
    if run_type == "local":
        num_procs = int(sys.argv[3])
        if num_procs == 1: 
            results = map(RunnerWrapper(runner), parameters.all())
        else:
            pool = Pool(num_procs)
            results = pool.map(RunnerWrapper(runner), parameters.all())
        write_output("results.json", results)

    ############################################################################
    # Run experiment on a cluster controlled by torque
    ############################################################################
    if run_type == "torque":
        # The script run by qsub
        bash_script = \
        """
        #!/bin/bash
        #PBS -l %(resource_limits)s
        #PBS -k n
        export PATH=$PBS_O_PATH
        cd $PBS_O_WORKDIR
        python %(script_name)s %(param_config_path)s torque_job $PBS_ARRAYID
        """ % { "resource_limits": resource_limits,
                "script_name": sys.argv[0], 
                "param_config_path": param_config_path }

        # Which jobs still need to run?
        ranges = []
        def already_done(i):
            file_exists = path.exists(path.join(out_dir, "result_%d.json"%i))
            return file_exists
        i = 1
        while i <= parameters.num_settings:
            while i <= parameters.num_settings and already_done(i): i += 1
            start = i
            while i <= parameters.num_settings and not already_done(i): i += 1
            ranges.append((start, i-1))
            
        def range_to_str((a,b)):
            if a == b: return str(a)
            else: return str(a)+"-"+str(b)
        array_jobs = ",".join(map(range_to_str, ranges))
        
        arguments = ["qsub",
                     "-t", array_jobs,
                     "-"]
        p = Popen(arguments, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        p.communicate(input=bash_script)
        p.wait()

    ############################################################################
    # A single torque job
    ############################################################################
    if run_type == "torque_job":
        job_number = int(sys.argv[3]) - 1
        result = runner(**parameters.get(job_number))
        write_output("result_%d.json"%job_number, result)
        exit(0)

    ############################################################################    
    # Gather the results of a torque run
    ############################################################################    
    
    if run_type == "torque_gather":
        def load_result(job_number): 
            return json.load(file(path.join(out_dir, "result_%d.json"%job_number), "r"))
        results = map(load_result, parameters.all_ixs())
        write_output("results.json", results)


class ParameterProduct:
    def __init__(self, param_config):
        """Constructs a new ParameterProduct. The input param_config is a
        dictionary that maps parameter names to a collection of
        values. This class allows the cross product of the provided
        parameter settings to be iterated and efficiently indexed.
        """
        self.param_config = param_config
        self.names = list(param_config.iterkeys())
        self.values = list(param_config.itervalues())
        self.num_values = [len(vs) for vs in self.values]
        # Sort parameters by name, since dictionary iterator order is unpredictable
        self.names, self.values, self.num_values = \
            [list(x) for x in zip(*sorted(zip(self.names, self.values, self.num_values)))]
        self.num_settings = reduce(operator.mul, self.num_values)

    def get_values(self, parameter):
        """ Get the possible values for a parameter. """
        return self.param_config[parameter]

    def num_settings(self): 
        """ Get the number of parameter settings in the product. """
        return self.num_settings

    def get(self, i): 
        """ Get the ith parameter setting. """
        params = dict()
        for (n, vs, m) in izip(self.names, self.values, self.num_values):
            params[n] = vs[i%m]
            i = int(i/m)
        return params

    def all(self):
        """ Returns an iterator over all the parameter settings. """
        return self.find()

    def all_ixs(self):
        return self.find_ixs()

    def find_ixs(self, criterion=None): 
        """ Iterates over the parameter indices matching a criterion. """
        for i in xrange(self.num_settings):
            if self.matches(criterion, self.get(i)): yield i

    def list_ixs(self, criterion=None): 
        """ Same as find_ixs but returns a list instead of an iterator. """
        return list(self.find_ixs(criterion))

    def find(self, criterion=None):
        """ Iterates over the parameter settings matching a criterion. """
        for i in xrange(self.num_settings):
            ps = self.get(i)
            if self.matches(criterion, ps): yield ps
        
    def list(self, criterion=None): 
        """ Same as find but returns a list instead of an iterator. """
        return list(self.find(criterion))

    def matches(self, criterion, ps):
        """Checks whether some parameters match a criterion. The criterion may
        be:
          - None, in which case no filtering is done; 
          - A dictionary mapping parameters to values, in which case only
            parameter settings agreeing with the dictionary are matched.
          - A function from parameters to booleans that does the matching.
        """
        if criterion == None: return True
        elif isinstance(criterion, DictType):
            matches = True
            for (k,v) in criterion.iteritems():
                if not ps.has_key(k) or ps[k] != v:
                    matches = False
                    break
            return matches
        elif isinstance(criterion, FunctionType):
            return criterion(ps)


class RunnerWrapper:
    """ Wraps a function so that it takes a dictionary of named parameters. """
    
    def __init__(self, f):
        self.f = f
        
    def __call__(self, args):
        return self.f(**args)
        
