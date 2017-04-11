import numpy as np

from GPflow.param import Parentable


class Domain(Parentable):
    """
    Basic class, representing an optimization domain by aggregating several parameters.
    """
    def __init__(self, parameters):
        super(Domain, self).__init__()
        self._parameters = parameters

    @property
    def lower(self):
        return np.array(map(lambda param: param.lower, self._parameters)).flatten()

    @property
    def upper(self):
        return np.array(map(lambda param: param.upper, self._parameters)).flatten()

    def optimize(self, optimizer, objectivefx):
        optimizer.domain = self
        result = optimizer.optimize(objectivefx)

    def __add__(self, other):
        assert isinstance(other, Domain)
        return Domain(self._parameters + other._parameters)

    @property
    def size(self):
        return sum(map(lambda param: param.size, self._parameters))

    def __setattr__(self, key, value):
        super(Domain, self).__setattr__(key, value)
        if key is not '_parent':
            if isinstance(value, Parentable):
                value._parent = self
            if isinstance(value, list):
                for val in (x for x in value if isinstance(x, Parentable)):
                    val._parent = self

    def __eq__(self, other):
        return self._parameters == other._parameters

    def __in__(self, X):
        return np.all(np.logical_and((self.lower <= X), (X <= self.upper)), axis=1)

    @property
    def value(self):
        return np.hstack(map(lambda p: p.value, self._parameters))

    @value.setter
    def value(self, x):
        x = np.atleast_2d(x)
        offset = 0
        for p in self._parameters:
            p.value = x[:,offset:]
            offset += p.size


class Parameter(Domain):
    """
    Abstract class representing a parameter (which corresponds to a one-dimensional domain)
    This class can be derived for continuous, discrete and categorical parameters
    """
    def __init__(self, label, xinit):
        super(Parameter, self).__init__([self])
        self.label = label
        self._xinit = xinit

    @Domain.size.getter
    def size(self):
        return 1

    @Domain.lower.getter
    def lower(self):
        return None

    @Domain.upper.getter
    def upper(self):
        return None

    @Domain.value.getter
    def value(self):
        return self._xinit

    @value.setter
    def value(self, x):
        self._xinit = x[:,0]

    def __eq__(self, other):
        return False


class ContinuousParameter(Parameter):
    def __init__(self, label, lb, ub, xinit=None):
        self._range = [lb, ub]
        super(ContinuousParameter,self).__init__(label, xinit or ((ub+lb) / 2))

    @Parameter.lower.getter
    def lower(self):
        return self._range[0]

    @Parameter.upper.getter
    def upper(self):
        return self._range[1]

    @lower.setter
    def lower(self, value):
        self._range[0] = value

    @upper.setter
    def upper(self, value):
        self._range[1] = value

    def __eq__(self, other):
        return isinstance(other, ContinuousParameter) and np.all(self.lb == other.lb) and np.all(self.ub == other.ub)





