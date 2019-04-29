"""Author: Juan Emmanuel Johnson
A very simple implementation of a Gaussian process regression algorithm.
It uses a class paradigm and can be used to walk a user through all of 
the steps for a GP algorithm."""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.base import BaseEstimator, RegressorMixin
import GPy


class BasicGP(GaussianProcessRegressor):
    def __init__(self, n_restarts=5, kernel=None):
        if kernel is None:
            kernel = C() * RBF() + WhiteKernel()

        super().__init__(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=123,
        )

    def predict_noiseless(self, X, return_std=True):

        if return_std:
            self.bias = np.sqrt(self.kernel_.k2.noise_level)
            mean, std = self.predict(X, return_std=True)
            std -= self.bias
            return mean, std
        else:
            mean = self.predict(X, return_std=False)
            return mean


class HeteroGP(GaussianProcessRegressor):
    def __init__(self, X, clusters=5, n_restarts=5):

        prototypes = KMeans(n_clusters=clusters).fit(X).cluster_centers_
        kernel = C() * RBF() + HeteroscedasticKernel.construct(prototypes)

        super().__init__(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=123,
        )
        self.prototypes = prototypes

    def predict_noiseless(self, X, return_std=True):

        if return_std:
            self.bias = np.sqrt(self.kernel_.k2.sigma_2.mean())
            mean, std = self.predict(X, return_std=True)
            std -= self.bias
            return mean, std
        else:
            mean = self.predict(X, return_std=False)
            return mean


class SparseGP(BaseEstimator, RegressorMixin):
    def __init__(self, x_variance, n_inducing=10, random_state=123, max_iters=200):
        self.x_variance = np.array(x_variance).reshape(1, -1)
        self.n_inducing = n_inducing
        self.rng = np.random.RandomState(random_state)
        self.max_iters = max_iters

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape
        assert self.x_variance.shape[1] == d_dimensions
        # Convert covariance into matrix

        x_variance = np.tile(self.x_variance, (n_samples, 1))

        # Get inducing points
        z = self.rng.uniform(X.min(), X.max(), (self.n_inducing, d_dimensions))

        # Kernel matrix
        kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)
        gp_model = GPy.models.SparseGPRegression(
            X, y, kernel=kernel, Z=z, X_variance=x_variance
        )

        # Optimize
        # gp_model.inducing_inputs.fix()
        gp_model.optimize("scg", messages=1, max_iters=self.max_iters)

        self.gp_model = gp_model

        return self

    def predict(self, X, return_std=False):

        mean, var = self.gp_model.predict(X)
        if return_std:
            return mean, var
        else:
            return mean

    def predict_noiseless(self, X, return_std=True):

        mean, var = self.gp_model.predict_noiseless(X)
        if return_std:
            return mean, var
        else:
            return mean


class HetGP(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):

        kernel = GPy.kern.MLP(1) + GPy.kern.Bias(
            1
        )  # GPy.kern.RBF(1, ARD=False) #+ GPy.kern.WhiteHeteroscedastic(1, y.shape[0])

        gp_model = GPy.models.GPHeteroscedasticRegression(X, y, kernel)
        # gp_model.het_Gauss.variance = .05

        print(gp_model)
        gp_model.optimize()
        self.gp_model = gp_model

        return self

    def predict(self, X, return_std=True):
        return self.gp_model.predict_noiseless(X)


# Authors: Jan Hendrik Metzen <janmetzen@mailbox.org>
#
# License: BSD 3 clause

""" Non-stationary kernels that can be used with sklearn's GP module. """

import numpy as np

from scipy.special import gamma, kv

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, _approx_fprime, Hyperparameter, RBF


class HeteroscedasticKernel(Kernel):
    """Kernel which learns a heteroscedastic noise model.
    This kernel learns for a set of prototypes values from the data space
    explicit noise levels. These exemplary noise levels are then generalized to
    the entire data space by means for kernel regression.
    Parameters
    ----------
    prototypes : array-like, shape = (n_prototypes, n_X_dims)
        Prototypic samples from the data space for which noise levels are
        estimated.
    sigma_2 : float, default: 1.0
        Parameter controlling the initial noise level
    sigma_2_bounds : pair of floats >= 0, default: (0.1, 10.0)
        The lower and upper bound on sigma_2
    gamma : float, default: 1.0
        Length scale of the kernel regression on the noise level
    gamma_bounds : pair of floats >= 0, default: (1e-2, 1e2)
        The lower and upper bound on gamma
    """

    def __init__(
        self,
        prototypes,
        sigma_2=1.0,
        sigma_2_bounds=(0.1, 10.0),
        gamma=1.0,
        gamma_bounds=(1e-2, 1e2),
    ):
        assert prototypes.shape[0] == sigma_2.shape[0]
        self.prototypes = prototypes

        self.sigma_2 = np.asarray(sigma_2)
        self.sigma_2_bounds = sigma_2_bounds

        self.gamma = gamma
        self.gamma_bounds = gamma_bounds

        self.hyperparameter_sigma_2 = Hyperparameter(
            "sigma_2", "numeric", self.sigma_2_bounds, self.sigma_2.shape[0]
        )

        self.hyperparameter_gamma = Hyperparameter(
            "gamma", "numeric", self.gamma_bounds
        )

    @classmethod
    def construct(
        cls,
        prototypes,
        sigma_2=1.0,
        sigma_2_bounds=(0.1, 10.0),
        gamma=1.0,
        gamma_bounds=(1e-2, 1e2),
    ):
        prototypes = np.asarray(prototypes)
        if prototypes.shape[0] > 1 and len(np.atleast_1d(sigma_2)) == 1:
            sigma_2 = np.repeat(sigma_2, prototypes.shape[0])
            sigma_2_bounds = np.vstack([sigma_2_bounds] * prototypes.shape[0])
        return cls(prototypes, sigma_2, sigma_2_bounds, gamma, gamma_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]
        n_gradient_dim = n_prototypes + (0 if self.hyperparameter_gamma.fixed else 1)

        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.eye(X.shape[0]) * self.diag(X)
            if eval_gradient:
                K_gradient = np.zeros((K.shape[0], K.shape[0], n_gradient_dim))
                K_pairwise = pairwise_kernels(
                    self.prototypes / prototypes_std,
                    X / prototypes_std,
                    metric="rbf",
                    gamma=self.gamma,
                )
                for i in range(n_prototypes):
                    for j in range(K.shape[0]):
                        K_gradient[j, j, i] = (
                            self.sigma_2[i] * K_pairwise[i, j] / K_pairwise[:, j].sum()
                        )
                if not self.hyperparameter_gamma.fixed:
                    # XXX: Analytic expression for gradient?
                    def f(gamma):  # helper function
                        theta = self.theta.copy()
                        theta[-1] = gamma[0]
                        return self.clone_with_theta(theta)(X, Y)

                    K_gradient[:, :, -1] = _approx_fprime([self.theta[-1]], f, 1e-5)[
                        :, :, 0
                    ]
                return K, K_gradient
            else:
                return K
        else:
            K = np.zeros((X.shape[0], Y.shape[0]))
            return K  # XXX: similar entries?

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        prototypes_std = self.prototypes.std(0)
        n_prototypes = self.prototypes.shape[0]

        # kernel regression of noise levels
        K_pairwise = pairwise_kernels(
            self.prototypes / prototypes_std,
            X / prototypes_std,
            metric="rbf",
            gamma=self.gamma,
        )

        return (K_pairwise * self.sigma_2[:, None]).sum(axis=0) / K_pairwise.sum(axis=0)

    def __repr__(self):
        return "{0}(sigma_2=[{1}], gamma={2})".format(
            self.__class__.__name__,
            ", ".join(map("{0:.3g}".format, self.sigma_2)),
            self.gamma,
        )

