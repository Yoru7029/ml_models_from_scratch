import numpy as np


def lin_reg_compute_gradient(coef, intercept, X, y):
    """
    Compute the gradient of the coefficient and inteceptias term for the least square regression.

    Args:
        coef (ndarray of shape n_feature): Initial coefficients
        intecept (float): Initial intercept
        y (ndarray of shape m_observation): True response value of each observation
        X (matrix-like of (n_features) x m_observation): Input

    Returns:
        gradient
    """
    m, n = X.shape
    errors = (X @ coef + intercept) - y

    dj_db = 1/m * errors.sum()
    dj_dw = np.empty((n,))
    for i in range(n):
        dj_dw[i] = 1/m * np.dot(errors, X[:, i])

    return dj_db, dj_dw

def stochastic_GD(X, y, init_coef, init_intercept, max_iter, eta, tol, n_iter_no_change):
    """
    Stochastic Gradient Descent to estimate the optimal coefficients and intercept for least square problem
    
    :param(ndarray of shape m_observations and n_features) X: Predictors
    :param(ndarray of shape m_observations) y: True values of response
    :param(ndarray of shape n_features) init_coeff: Initial coefficents set for the algorithm, if None then it will be set to ndarray of 0
    :param(float) init_intercept: Initial intercept set for the algorithm, if None then it will be set to 0
    :param(int) max_iter: Maximum number of epochs
    :param(float) eta: Learning rate
    """
    m, n = X.shape
    if init_coef is None and init_intercept is None:
        coef = np.array([0]*n)
        intercept = 0
    else:
        coef = init_coef
        intercept = init_intercept
    
    # Initalize several variables
    count_down = None
    loss = 0
    rng = np.random.default_rng()
    
    for epoch in range(max_iter):
        # Shuffling the data during each epoch gives the unbiased esimate of the true gradient
        indices = rng.permutation(m)
        for i in indices:
            dj_db, dj_dw = lin_reg_compute_gradient(coef, intercept, X[i, :].reshape(-1, n), y[i])
            # Update parameters
            intercept = intercept - eta * dj_db
            coef = coef - eta * dj_dw

        # Get loss
        new_loss = compute_cost(X, y, intercept, coef)
        
        # Check tolerance
        if np.abs(new_loss - loss) < tol:
            if count_down is None: 
                count_down = n_iter_no_change
            else:
                count_down -= 1
                if count_down == 0: 
                    break     
        
        # Update current loss
        loss = new_loss
        
        # Get loss to display it during training
        if epoch % (np.ceil(max_iter / 10)) == 0 or i == (max_iter - 1):
            print(f"Epoch: {epoch}   Loss: {loss}")    
    
    # Return the intercept and coefficients
    print("Iteration done:", epoch)
    return intercept, coef

def compute_cost(X, y, intercept, coef):
    """
    Computing least squared loss
    
    :param(ndarray of shape m_observations and n_features) X: Predictors
    :param(ndarray of shape m_observations) y: True values of response
    :param intercept: Bias term
    :param coef: Weight terms
    """
    m, _ = X.shape
    y_pred = (X @ coef) + intercept
    sse = np.sum((y - y_pred)**2)
    return 1/m * sse