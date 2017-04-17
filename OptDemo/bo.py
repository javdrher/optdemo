import numpy as np
from scipy.optimize import OptimizeResult

from acquisition import Acquisition
from optim import Optimizer, SciPyOptimizer
from design import EmptyDesign


class BayesianOptimizer(Optimizer):
    """
    Specific optimizer representing bayesian optimization. Takes a domain, an acquisition function and an optimization
    strategy for the acquisition function.
    """
    def __init__(self, domain, acquisition, optimizer=None, initial=None):
        assert isinstance(acquisition, Acquisition)
        super(BayesianOptimizer, self).__init__(domain, exclude_gradient=True)
        self.acquisition = acquisition
        if initial is None:
            initial = EmptyDesign(domain)
        self.set_initial(initial.generate())
        self.optimizer = SciPyOptimizer(domain) if optimizer is None else optimizer

    def _update_model_data(self, newX, newY):
        """
        Update the underlying models of the acquisition function with new data
        :param newX: samples (# new samples x indim)
        :param newY: values obtained by evaluating the objective and constraint functions (# new samples x # targets)
        """
        assert self.acquisition.data[0].shape[1] == newX.shape[-1]
        assert self.acquisition.data[1].shape[1] == newY.shape[-1]
        assert newX.shape[0] == newY.shape[0]
        X = np.vstack((self.acquisition.data[0], newX))
        Y = np.vstack((self.acquisition.data[1], newY))
        self.acquisition.set_data(X, Y)

    def _acquisition_wrapper(self, x):
        """
        Negation of the acquisition score/gradient
        :param x: candidate points (# candidates x indim)
        :return: negative score and gradient (maximizing acquisition function vs minimization algorithm)
        """
        scores, grad = self.acquisition.evaluate_with_gradients(np.atleast_2d(x))
        return -scores, -grad

    def _evaluate(self, X, fxs):
        fxs = np.atleast_1d(fxs)
        if X.size > 0:
            evaluations = np.hstack(map(lambda f: f(X), fxs))
            assert evaluations.shape[1] == self.acquisition.data[1].shape[1]
            return evaluations
        else:
            return np.empty((0, self.acquisition.data[1].shape[1]))

    def _create_result(self, success, message):
        """
        Given all data evaluated after the optimization, analyse and return an OptimizeResult
        :param success: Optimization successfull? (True/False)
        :param message: return message
        :return: OptimizeResult object
        """
        X, Y = self.acquisition.data

        # Filter on constraints
        # Extend with valid column - in case of no constrains
        Yext = np.hstack((-np.ones((Y.shape[0], 1)), Y[:,self.acquisition.constraint_indices()]))
        valid = np.all(Yext  < 0, axis=1)

        if not np.any(valid):
            return OptimizeResult(success=False,
                                  message="No evaluations satisfied the constraints")

        valid_X = X[valid,:]
        valid_Y = Y[valid,:]
        valid_Yo = valid_Y[:,self.acquisition.objective_indices()]

        # Here is the place to plug in pareto front if valid_Y.shape[1] > 1
        # else
        idx = np.argmin(valid_Yo)

        return OptimizeResult(x=valid_X[idx,:],
                              success=success,
                              fun=valid_Yo[idx,:],
                              message=message)

    def optimize(self, objectivefx, n_iter=20):
        """
        Run Bayesian optimization for a number of iterations. Each iteration a new data point is selected by optimizing
        an acquisition function. This point is evaluated on the objective and black-box constraints. This retrieved
        information then updates the model
        :param objectivefx: (list of) expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """

        # Bayesian optimization is Gradient-free: provide wrapper. Also anticipate for lists and pass on a function
        # which satisfies the optimizer.optimize interface
        fx = lambda X: (self._evaluate(X, objectivefx), np.zeros((X.shape[0],0)))
        return super(BayesianOptimizer, self).optimize(fx, n_iter=n_iter)

    def _optimize(self, fx, n_iter):
        """
        Internal optimization function. Receives and ObjectiveWrapper
        :param fx: ObjectiveWrapper object wrapping expensive black-box objective and constraint functions
        :param n_iter: number of iterations to run
        :return: OptimizeResult object
        """

        # Evaluate and add the initial design (if any)
        initial = self.get_initial()
        values = fx(initial)
        self._update_model_data(initial, values)
        # Remove initial design for additional calls to optimize to proceed optimization
        self.set_initial(EmptyDesign(self.domain).generate())

        # Optimization loop
        for i in range(n_iter):
            result = self.optimizer.optimize(self._acquisition_wrapper)
            self._update_model_data(result.x, fx(result.x))

        return self._create_result(True, "OK")