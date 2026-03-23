import numpy as np


def sigmoid(z):
    g = np.exp(z) / (1 + np.exp(z))
    return g


def compute_diag(X, betas):
    """Compute diagonal matrix which has its diagonal contains p(1 - p)"""
    p = sigmoid(X @ betas)
    w = np.diag(p * (1 - p))
    return w


def adjusted_response(X, betas, w, y):
    p = sigmoid(X @ betas)
    z = X @ betas + np.linalg.inv(w) @ (y - p)
    return z


def newton_step(X, y, betas):
    w = compute_diag(X, betas)
    z = adjusted_response(X, betas, w, y)
    step = np.linalg.inv(X.T @ w @ X) @ X.T @ w @ z
    new_intercept = step[0]
    new_coef = step[1:]
    return new_intercept, new_coef


def newton_raphson(X, y, init_coef, init_intercept, max_iter, tol, n_iter_no_change):
    _, n = X.shape
    X_design = np.c_[np.array([1]*X.shape[0]), X]
    if init_coef is None and init_intercept is None:
        coef = np.array([0]*n)
        intercept = 0
    else:
        coef = init_coef
        intercept = init_intercept

    betas = np.concatenate(([intercept], coef))
    count_down = None
    loss = 0
    for epoch in range(max_iter):
        intercept, coef = newton_step(X_design, y, betas)
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
    
    # Return the intercept and coefficients
    print("Iteration done:", epoch)
    return intercept, coef


def compute_gradient(intercept, coef, X, y):
    m, n = X.shape
    p = sigmoid(intercept + X @ coef)
    
    dj_db = 1/m * np.sum(p - y)
    dj_dw = np.empty((n,))
    for i in range(n):
        dj_dw[i] = 1/m * np.dot((p - y), X[:, i])

    return dj_db, dj_dw


def compute_cost(X, y, intercept, coef):
    """Compute binary cross entropy loss"""
    m, _ = X.shape
    p = sigmoid(intercept + X @ coef)
    
    log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    loss = -1/m * log_likelihood
    
    return loss


def stochastic_gradient_descent(X, y, init_coef, init_intercept, max_iter, eta, tol, n_iter_no_change):
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
            dj_db, dj_dw = compute_gradient(intercept, coef, X[i, :].reshape(-1, n), y[i])
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

