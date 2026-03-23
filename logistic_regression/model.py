import numpy as np

from ._utils import stochastic_gradient_descent, newton_raphson, sigmoid


class LogisticRegression():
    """
    Own attempt on implementing Logistic Regression model.
    Method used for this class is Maximum Likelihood Estimation (MLE) or Graidient Descent (GD)
    """
    # TODO: Implement Logistic Regression model
    def __init__(self,
                 solver="newton", 
                 init_intercept=None,
                 init_coef=None,
                 max_iter=1000,
                 eta=1e-4,
                 tol=1e-4,
                 n_iter_no_change=5):
        
        self._max_iter = max_iter
        self._eta = eta
        self._init_intercept = init_intercept
        self._init_coef = init_coef
        self._tol = tol
        self._solver = solver
        if not (1 < n_iter_no_change <= max_iter):
            raise ValueError("n_iter_no_change must be in range (1, max_iter]")
        self._n_iter_no_change = n_iter_no_change

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self._solver == "sgd":
            self.intercept_, self.coef_ = stochastic_gradient_descent(
                X, y, 
                init_coef=self._init_coef, 
                init_intercept=self._init_intercept, 
                max_iter=self._max_iter, 
                eta=self._eta,
                tol=self._tol,
                n_iter_no_change=self._n_iter_no_change
                )
        elif self._solver == 'newton':
            self.intercept_, self.coef_ = newton_raphson(
                X, y,
                init_coef=self._init_coef, 
                init_intercept=self._init_intercept, 
                max_iter=self._max_iter, 
                tol=self._tol,
                n_iter_no_change=self._n_iter_no_change
            )
        else:
            raise ValueError("Invalid solver")

        # Return the fitted model
        print("Model fitted")
        return self
    
    def predict(self, X):
        m, _ = X.shape
        proba = sigmoid(self.intercept_ + (X @ self.coef_))
        pred = np.array([0] * m)
        pred[proba > 0.5] = 1
        return pred
    
    def predict_proba(self, X):
        proba = sigmoid(self.intercept_ + (X @ self.coef_))
        return proba