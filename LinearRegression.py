import numpy as np
from utils import mean_squared_error

class LinearRegression:
    def __init__(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def __call__(self, X):
        return (X @ self.weights) + self.bias
    
    def fit(self, X, y, loss_fn = mean_squared_error, steps=10000, lr=1e-4, log_interval=1000):

        for step in range(steps):
            y_pred = self.__call__(X)

            w_grad = (X.T @ (y_pred - y)) / len(y)
            b_grad = np.mean(y_pred - y)

            # Proper gradient clipping for arrays
            w_grad = np.clip(w_grad, -1.0, 1.0)
            b_grad = np.clip(b_grad, -1.0, 1.0)
            
            # Check for numerical issues
            if np.any(np.isnan(w_grad)) or np.any(np.isinf(w_grad)):
                print(f"Warning: NaN/inf in gradients at step {step}")
                break

            self.weights -= lr * w_grad
            self.bias -= lr * b_grad

            if step % log_interval == 0:
                print(f"{step+1}/{steps} | Loss: {loss_fn(y, y_pred)}")


# Training Simulation
N_SAMPLES = 10000
N_FEATURES = 100

X = np.random.randn(N_SAMPLES, N_FEATURES) * 0.1  # Scale input data
w = np.random.randn(N_FEATURES) * 0.01  # Very small true weights
noise = np.random.randn(N_SAMPLES) * 0.01  # Small noise
y = X @ w + noise

model = LinearRegression(N_FEATURES)
model.fit(X, y)
    
