from typing import Optional, Union, Tuple

import GPy
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y, check_array


class SparseGPRegressor(BaseEstimator, RegressorMixin):
    """Sparse Gaussian process regression algorithm. This algorithm 
    implements the GPR algorithm with considerations for uncertain inputs.
    
    Parameters
    ----------
    kernel : GPy.kern.Kern, default=None
        The kernel function. Default kernel is the RBF kernel.

    n_inducing : int, default=10
        The number of inducing inputs to use for the regression model.

    inference : str, default='vfe'
        option to choose inference algorithm
        * vfe  - approximates the posterior (default)
        * fitc - approximates the model
        * pep  - hybrid of above approaches

    X_variance : float,np.ndarray, default=None
        Option to do uncertain inputs.
    
    max_iters : int, default=200
        Maximum number of iterations to use for the optimizer.

    optimizer : str, default='scg'
        The optimizer to use for the maximum log-likelihood optimization.

    n_restarts : int, default=10
        Number of random restarts for the optimizer. Good for avoiding 
        local minima
    
    verbose : int, default=None
        Option to display messages during optimization
    
    normalize_y : bool, default=False
        Option to normalize the outputs before the optimization. Good
        for GP algorithms in general. Will do the reverse transformation
        for predictions.
    
    alpha : float, default=0.5
        Option only used for the 'PEP' inference algorithm; the factor for
        the hybrid between the two other inference algorithms.
        alpha = 1  -> VFE
        alpha = 0  -> FITC
    Attributes
    ----------
    X_variance : np.ndarray, (features, features)
        The error covariance matrix for the inputs.
    
    gp_model : GPy.core
        The trained GP model.

    _y_train_mean : np.ndarray, (features)
        The 

    _y_train_std : np.ndarray, (features)

    Examples
    --------
    >> from src.gpy import GPRegression
    >> from sklearn.datasets import make_friedman2
    >> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >> X_variance = 2.0
    >> n_restarts = 0
    >> verbose = None
    >> gp_clf = GPRegression(
        verbose=verbose, 
        n_restarts=n_restarts, 
        X_variance=X_variance,
    )
    >> gp_clf.fit(X, y);
    >> ymean, ystd = gp_clf.predict(
        Xtest, 
        return_std=True,
        noiseless=False,
        linearized=True
    )

    >> from src.gpy import GPRegression
    >> from sklearn.datasets import make_friedman2
    >> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >> X_variance = np.ndarry([0.1, 0.1, 0.1])
    >> n_restarts = 0
    >> verbose = None
    >> gp_clf = GPRegression(
        verbose=verbose, 
        n_restarts=n_restarts, 
        X_variance=X_variance,
    )
    >> gp_clf.fit(X, y);
    >> ymean, ystd = gp_clf.predict(
        Xtest, 
        return_std=True,
        noiseless=False,
        linearized=True
    )
    """

    def __init__(
        self,
        kernel: Optional[GPy.kern.Kern] = None,
        inference: str = "vfe",
        X_variance: Optional[Union[float, np.ndarray]] = None,
        n_inducing: int = 10,
        max_iters: int = 200,
        optimizer: str = "scg",
        n_restarts: int = 10,
        verbose: Optional[int] = None,
        alpha: float = 0.5,
        normalize_y: bool = False,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.X_variance = X_variance
        self.inference = inference
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.alpha = alpha
        self.normalize_y = normalize_y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Sparse GP regression model.
        
        Parameters
        ----------
        X : np.ndarray, (samples, features)
            Input vectors for the training regime
        
        y : np.ndarray, (samples, targets)
            Labels for the training regime

        Returns
        -------
        self : returns an instance of self
        """
        # Check inputs
        X, y = check_X_y(
            X, y, multi_output=True, y_numeric=True, ensure_2d=True, dtype="numeric"
        )

        # normalize outputs
        if self.normalize_y == True:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = np.std(y, axis=0)

            # remove mean to make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        # get shapes
        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # check X_variance
        self.X_variance = self._check_X_variance(self.X_variance, d_dimensions)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Kernel matrix
        gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

        # set the fitc inference
        if self.inference.lower() == "vfe":
            gp_model.inference_method = GPy.inference.latent_function_inference.VarDTC()

        elif self.inference.lower() == "fitc":
            gp_model.inference_method = GPy.inference.latent_function_inference.FITC()

        elif self.inference.lower() == "pep":
            gp_model.inference_method = GPy.inference.latent_function_inference.PEP(
                self.alpha
            )
        else:
            raise ValueError(f"Unrecognized inference method: {self.inference}")
        # Optimize
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01

        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model

    def _check_X_variance(
        self, X_variance: Union[None, float, np.ndarray], n_dims: int
    ) -> Union[None, np.ndarray]:
        """Private method to check the X_variance parameter
        
        Parameters
        ----------
        X_variance : float, None, np.ndarray 
            The input for the uncertain inputs
        
        Returns
        -------
        X_variance : np.ndarray, (n_features, n_features)
            The final matrix for the uncertain inputs.
        """
        if X_variance is None:
            return X_variance

        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=(n_dims, n_dims))

        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.identity(n_dims)
            elif X_variance.shape == n_dims:
                return np.diag(X_variance)
            elif X_variance.shape == (n_dims, n_dims):
                return X_variance
            else:
                raise ValueError(
                    f"Shape of 'X_variance' ({X_variance.shape}) "
                    f"doesn't match X ({n_dims})"
                )
        else:
            raise ValueError(f"Unrecognized type of X_variance.")

    def predict(
        self, X, return_std=False, full_cov=False, noiseless=True, linearized=True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the GP Model. Returns the mean and standard deviation 
        (optional) or the full covariance matrix (optional). Also includes an
        option to add the error correction term for the Linearized GP method.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
            Input vector to be predicted
        
        return_std : bool, default=False
            flag to return the standard deviation in the ouputs.
        
        full_cov : bool, default=False
            flag to return the full covariance for the outputs
        
        noiseless : bool, default=True
            flag to return the noise likelihood term with the 
            standard deviation
        
        linearized : bool, default=True
            flag to return the standard deviation with the 
            corrected input error.
        
        Returns
        -------
        y_mean : np.ndarray, (n_samples, n_targets)
            The mean predictions for the outputs
        
        y_std : np.ndarray, (n_samples)
            The standard deviations for the outputs. 
            * linearized with input errors if linearized=True
            * has the likelihood noise if noiseless=False
        
        y_cov : np.ndarray, (n_samples, n_samples)
            The covariance matrix foer the outputs. Only returned if
            full_cov = True
        """
        X = check_array(X, ensure_2d=True, dtype="numeric")

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        # undo normalization
        if self.normalize_y == True:
            mean = self._y_train_std * mean + self._y_train_mean
            var = var * self._y_train_std ** 2

        if return_std:
            # we want the variance correction
            if linearized == True and self.X_variance is not None:
                # get the variance correction
                var_add = self._variance_correction(X)

                # get diagonal elements only
                if full_cov == False:
                    var_add = np.diag(var_add)[:, None]
                # add correction to original variance
                var += var_add

                # return mean and standard deviation
                return mean, np.sqrt(var)

            else:
                return mean, np.sqrt(var)
        else:
            return mean

    def _variance_correction(self, X: np.ndarray) -> np.ndarray:
        """Private method to calculate the corrective term for the 
        predictive variance.
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)

        Returns
        -------
        var_add : np.ndarray, (n_samples)
        """
        x_der, _ = self.gp_model.predictive_gradients(X)

        # calculate correction
        var_add = x_der[..., 0] @ self.X_variance @ x_der[..., 0].T
        return var_add
