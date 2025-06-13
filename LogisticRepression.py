import numpy as np
from loss import binary_crossentropy, accuracy
from sklearn.datasets import make_classification

class LogisticRegression:
    
    def __init__(self, n_features) -> None:
        self.weights = np.random.uniform(-1/np.sqrt(n_features), 1/np.sqrt(n_features), n_features)
        self.bias = np.random.uniform(-1/np.sqrt(n_features), 1/np.sqrt(n_features))
    
    def __call__(self, X):
        return  1 / (1 + np.exp( - (X @ self.weights + self.bias)))
    
    def fit(self, X, y, lr=1e-3, steps=10000, log_interval=1000):
        for step in range(steps):
            y_pred = self.__call__(X)
            loss = binary_crossentropy(y, y_pred)

            self.weights -= lr * ((X.T @ (y_pred - y)) / len(y))
            self.bias -= lr * (np.mean(y_pred - y))
            if step % log_interval == 0:
                print(f"{step}/{steps}: Loss: {loss}, Accuracy: {accuracy(y, y_pred)}")

# Training Simulation
N_FEATURES = 20
X, y = make_classification(n_samples=1000, random_state=42)
model = LogisticRegression(N_FEATURES)
model.fit(X, y)



        