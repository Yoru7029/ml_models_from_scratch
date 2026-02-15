import numpy as np

from ._utils import stochastic_GD


class LinearRegression():
    """
    Own attempt on implementing Linear Regression model.
    Method used for this class is Ordinary Least Squares (OLS)

    Attributes
    ----------
    `coef_` : array of shape n_features
        Estimated coefficient for the linear regression problem.
    
    `rank_` : int
        Rank of the matrix `X`.
    
    `intercept_` : float
        Bias term in the linear model.
    """
    def fit(self, X, y):
        """
        Fit the linear model
        
        :param X: array-like object of the predictor 
        :param y: array-like object of the response
        """
        # Convert inputs into numpy array
        X = np.array(X)
        y = np.array(y)
        self.rank_ = X.shape[1]
        
        # Add a column with value of 1 to create the design matrix
        X = np.c_[np.array([1]*X.shape[0]), X]

        # Use normal equation to estimate parameters
        params = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = params[0]
        self.coef_ = params[1:]

        # Return the estimator
        print("Model fitted")
        return self
    
    def predict(self, X):
        """
        Predict using the model
        
        :param X: array-like
        """
        return self.intercept_ + (X @ self.coef_)
    
    def score(self, X, y) -> float:
        """
        Return R squared statistics of the fitted model

        :param X: array-like
        :param y: array-like
        """
        X = np.array(X)
        y_true = np.array(y)
        
        y_pred = self.predict(X)
        rss = sum((y_true - y_pred)**2)
        tss = sum((y_true - y_true.mean())**2)

        return 1 - (rss/tss)


class SGDLinearRegression(LinearRegression):
    """
    Linear Regression model estimated by Stochastic Gradient Descent
    """
    def __init__(self, init_intercept=None, init_coef=None, max_iter=1000, eta=1e-4, tol=1e-3, n_iter_no_change=5):
        self._max_iter = max_iter
        self._eta = eta
        self._init_intercept = init_intercept
        self._init_coef = init_coef
        self._tol = tol
        if not (1 < n_iter_no_change <= max_iter):
            raise ValueError("n_iter_no_change must be in range (1, max_iter]")
        self._n_iter_no_change = n_iter_no_change

    def fit(self, X, y):
        """
        Fit the linear model using SGD
        
        :param X: array-like object of the predictor 
        :param y: array-like object of the response
        """
        # Convert inputs into numpy array
        X = np.array(X)
        y = np.array(y)

        # Apply SGD to get the estimated parameters
        self.intercept_, self.coef_ = stochastic_GD(
            X, y, 
            init_coef=self._init_coef, 
            init_intercept=self._init_intercept, 
            max_iter=self._max_iter, 
            eta=self._eta,
            tol=self._tol,
            n_iter_no_change=self._n_iter_no_change
            )

        # Return the fitted model
        print("Model fitted")
        return self
