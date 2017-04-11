import numpy as np
from scipy.optimize import OptimizeResult, minimize
from GPflow import model
from GPflow._settings import settings

from .design import RandomDesign


class ObjectiveWrapper(model.ObjectiveWrapper):
    def __init__(self, objective, exclude_gradient):
        super(ObjectiveWrapper, self).__init__(objective)
        self._no_gradient = exclude_gradient

    def __call__(self, x):
        x = np.atleast_2d(x)
        f, g = super(ObjectiveWrapper, self).__call__(x)
        if self._no_gradient:
            return f
        return f,g


class Optimizer(object):
    """
    Generic class representing an optimizer. Wraps an optimization algorithm over a domain, starting from an initial
    (set of) point(s).
    """
    def __init__(self, domain, exclude_gradient=False):
        super(Optimizer, self).__init__()
        self.domain = domain
        self._wrapper_args = dict(exclude_gradient=exclude_gradient)
        self._initial = domain.value

    def optimize(self, objectivefx):
        objective = ObjectiveWrapper(objectivefx, **self._wrapper_args)
        try:
            result = self._optimize(objective)
        except KeyboardInterrupt:
            prev = objective._previous_x
            result = OptimizeResult(x=prev,
                                    success=False,
                                    fun=objectivefx(prev),
                                    message="Caught KeyboardInterrupt, returning most recent state.")
        result.x = np.atleast_2d(result.x)
        return result

    def get_initial(self):
        return self._initial

    def set_initial(self, initial):
        initial = np.atleast_2d(initial)
        assert(initial.shape[1] == self.domain.size)
        self._initial = initial

    def gradient_enabled(self):
        return not self._wrapper_args['exclude_gradient']


class CandidateOptimizer(Optimizer):
    """
    This optimizer optimizes an objective function by evaluating of a set of candidate points (and returning the point
    with minimal objective value.
    """
    def __init__(self, domain, candidates):
        super(CandidateOptimizer, self).__init__(domain, exclude_gradient=True)
        self.candidates = candidates

    def get_initial(self):
        return np.vstack((super(CandidateOptimizer, self).get_initial(), self.candidates))

    def _optimize(self, objective):
        points = self.get_initial()
        evaluations = objective(points)
        idx_best = np.argmin(evaluations, axis=0)

        return OptimizeResult(x=points[idx_best[0], :],
                              success=True,
                              fun=evaluations[idx_best,:],
                              nfev=self.candidates.shape[0],
                              message="OK")


class MCOptimizer(CandidateOptimizer):
    """
    This class represents optimization of an objective function by evaluating a set of random points. Each call to
    optimize, a different set of random points is evaluated.
    """
    def __init__(self, domain, nsamples):
        super(MCOptimizer, self).__init__(domain, np.empty((0, domain.size)))
        self._design = RandomDesign(nsamples, domain)

    def _optimize(self, objective):
        self.candidates = self._design.generate()
        return super(MCOptimizer, self)._optimize(objective)


class SciPyOptimizer(Optimizer):
    """
    Wraps optimization with scipy's minimize function.
    """
    def __init__(self, domain, method='L-BFGS-B', tol=None, maxiter=1000):
        super(SciPyOptimizer, self).__init__(domain)
        options = dict(disp=settings.verbosity.optimisation_verb,
                        maxiter=maxiter)
        self.config = dict(tol=tol,
                           method=method,
                           options=options)

    def _optimize(self, objective):
        result = minimize(fun=objective,
                          x0=self.get_initial(),
                          jac=self.gradient_enabled(),
                          bounds = zip(self.domain.lower, self.domain.upper),
                          **self.config)
        return result


class StagedOptimizer(Optimizer):
    """
    Represents an optimization pipeline. A list of optimizers can be specified (all on the same domain). The optimal
    solution of the an optimizer is used as an initial point for the next optimizer.
    """
    def __init__(self, optimizers):
        assert all(map(lambda opt: optimizers[0].domain == opt.domain, optimizers))
        super(StagedOptimizer, self).__init__(optimizers[0].domain)
        self.optimizers = optimizers

    def optimize(self, objectivefx):
        #self.optimizers[0].set_initial(self.get_initial())
        for current, next in zip(self.optimizers[:-1], self.optimizers[1:]):
            result = current.optimize(objectivefx)
            next.set_initial(result.x)
        return self.optimizers[-1].optimize(objectivefx)
