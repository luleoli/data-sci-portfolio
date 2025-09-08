import math
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Log Loss function
def log_loss(y, preds):
    eps = 1e-15  # avoid log(0)
    loss = 0.0
    for yi, pi in zip(y, preds):
        pi = min(max(pi, eps), 1 - eps)  # clip predictions
        loss += -(yi * math.log(pi) + (1 - yi) * math.log(1 - pi))
    return loss / len(y)


# Logistic Regression training (Gradient Descent) with separate intercept and coefficients
def fit(X, y, lr=0.01, epochs=1000, batch_size=None, l2_lambda=None, decay=None):
    n_samples, n_features = len(X), len(X[0])
    intercept = 0.0
    coefs = [0.0] * n_features

    if batch_size is None or batch_size > n_samples:
        batch_size = n_samples

    for epoch in range(epochs):
        preds = []
        
        # Learning rate decay
        if decay:
            lr = lr / (1 + decay * epoch)

        # Mini-batch gradient descent
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]
            batch_y = y[start:end]

            batch_preds = []
            batch_grads = [0.0] * n_features
            batch_grad_intercept = 0.0

            for i in range(len(batch_X)):
                z = intercept + sum(w*x for w, x in zip(coefs, batch_X[i]))
                pred = sigmoid(z)
                batch_preds.append(pred)

                error = batch_y[i] - pred
                batch_grad_intercept += error
                for j in range(n_features):
                    batch_grads[j] += error * batch_X[i][j]

            # Update intercept and coefficients for the batch
            intercept += lr * (batch_grad_intercept / len(batch_X))

            for j in range(n_features):
                coefs[j] += lr * (batch_grads[j] / len(batch_X))
                if l2_lambda:
                    coefs[j] -= lr * l2_lambda * coefs[j]  # L2 regularization term

            preds.extend(batch_preds)

        # compute log loss at the end of the epoch
        loss = log_loss(y, preds)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Log Loss: {loss:.4f}")

    return intercept, coefs

def fit_newton(X, y, reg_lambda=1.0, tol=1e-6, max_iter=100):
        X = np.c_[np.ones(X.shape[0]), X]  # add intercept
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)

        for it in range(max_iter):
            # predictions
            z = X @ beta
            p = sigmoid(z)

            # gradient (L2 penalty on weights, not intercept)
            g = X.T @ (p - y) / n_samples
            g[1:] += reg_lambda * beta[1:] / n_samples

            # Hessian with L2 penalty
            W = np.diag(p * (1 - p))
            H = (X.T @ W @ X) / n_samples
            for j in range(1, n_features):
                H[j, j] += reg_lambda / n_samples

            # Newton step
            step = np.linalg.solve(H, g)
            beta -= step

            # convergence
            if np.linalg.norm(step) < tol:
                break

        # save coefficients in sklearn-style attributes
        intercept_ = beta[0]
        coef_ = beta[1:].reshape(1, -1)
        return intercept_, coef_

# Predict function
def predict_proba(X, intercept, coefs):
    preds = []
    for x in X:
        z = intercept + sum(w*x_i for w, x_i in zip(coefs, x))
        prob = sigmoid(z)
        preds.append(prob)
    return preds