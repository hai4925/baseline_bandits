from itertools import izip

class ParameterSettings:

    def __init__(self, param_values):
        self.names = []
        self.values = []
        self.num_values = []
        self.num_settings = 1
        for (name, values) in param_values.iteritems():
            self.names.append(name)
            self.values.append(values)
            self.num_values.append(len(values))
            self.num_settings *= len(values)

    def get(self, index):
        """ Get the ith parameter settings. """
        params = dict()
        for (n, vs, m) in izip(self.names, self.values, self.num_values):
            params[n] = vs[index%m]
            index = int(index / m)
        return params

    def indices(self, selector=None):
        for i in xrange(self.num_settings):
            if selector == None or selector(self.parameters(i)): yield i

    def all(self, selector=None):
        for i in self.indices(selector): 
            yield self.get(i)





















