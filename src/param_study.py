from itertools import izip
from types import FunctionType, DictType

class ParameterSettings:

    def __init__(self, param_values):
        self.param_values = param_values
        self.names = []
        self.values = []
        self.num_values = []
        self.num_settings = 1
        for (name, values) in param_values.iteritems():
            self.names.append(name)
            self.values.append(values)
            self.num_values.append(len(values))
            self.num_settings *= len(values)

    def get_values(self, param):
        return self.param_values[param]

    def get(self, index):
        """ Get the ith parameter settings. """
        params = dict()
        for (n, vs, m) in izip(self.names, self.values, self.num_values):
            params[n] = vs[index%m]
            index = int(index / m)
        return params

    def find_indices(self, selector=None):
        if selector == None:
            for i in xrange(self.num_settings): yield i
        elif isinstance(selector, DictType):
            for i in xrange(self.num_settings):
                ps = self.get(i)
                matches = True
                for (k,v) in selector.iteritems():
                    if not ps.has_key(k) or ps[k] != v:
                        matches = False
                        break
                if matches: yield i
        elif isinstance(selector, FunctionType):
            for i in xrange(self.num_settings):
                if selector(self.get(i)): yield i

    def list_indices(self, selector=None):
        return list(self.find_indices(selector))

    def find(self, selector=None):
        for i in self.find_indices(selector): yield self.get(i)
        
    def list(self, selector=None):
        return list(self.find(selector))

    def all(self):
        return self.find()





















