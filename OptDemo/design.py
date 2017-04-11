import numpy as np


class Design(object):
    """
    Space-filling designs generated within a domain.
    """
    def __init__(self, size, domain):
        super(Design, self).__init__()
        self.size = size
        self.domain = domain

    def generate(self):
        pass


class RandomDesign(Design):
    """
    Random space-filling design
    """
    def __init__(self, size, domain):
        super(RandomDesign, self).__init__(size, domain)

    def generate(self):
        X = np.random.rand(self.size, self.domain.size)
        return X * (self.domain.upper - self.domain.lower) + self.domain.lower


class FactorialDesign(Design):
    """
    Grid-based design
    """
    def __init__(self, levels, domain):
        self.levels = levels
        size = levels ** domain.size
        super(FactorialDesign, self).__init__(size, domain)

    def generate(self):
        Xs = np.meshgrid(*[np.linspace(l, u, self.levels) for l, u in zip(self.domain.lower, self.domain.upper)])
        return np.vstack(map(lambda X: X.ravel(), Xs)).T


class EmptyDesign(Design):
    """
    No design, used as placeholder
    """
    def __init__(self, domain):
        super(EmptyDesign, self).__init__(0, domain)

    def generate(self):
        return np.empty((0, self.domain.size))