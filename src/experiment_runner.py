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
    param_config = json.load(file(param_config_path,"r"))
    parameters = ParameterProduct(param_config)

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
        bash_script = \
        """
        #!/bin/bash
        cd $PBS_O_WORKDIR
        python %(script_name)s %(param_config_path)s torque_job $PBS_ARRAYID
        """ % {"script_name": sys.argv[0], "param_config_path": param_config_path}
        arguments = ["qsub",
                     "-t", "1-%d" % parameters.num_settings,
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
        """Constructs a new ParameterProduct. The input parameters is a
        dictionary that maps parameter names to a collection of
        values. This class allows the cross product of the provided
        parameter settings to be iterated and indexed.
        """
        self.param_config = param_config
        self.names = list(param_config.iterkeys())
        self.values = list(param_config.itervalues())
        self.num_values = [len(vs) for vs in self.values]
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
        
