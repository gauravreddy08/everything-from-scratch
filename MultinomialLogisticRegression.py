import numpy as np
from loss import crossentropy
from utils import onehot_encoding
from sklearn.datasets import make_classification

class MultinomialLogisticRegression:
    def __init__(self, n_features, n_classes):
        self.weights = np.random.uniform(-1/np.sqrt(n_features),
                                         1/np.sqrt(n_features),
                                         size = (n_features, n_classes))
        
        self.bias = np.random.uniform(-1/np.sqrt(n_features),
                                         1/np.sqrt(n_features),
                                         size = n_classes)
    
    def __call__(self, X):
        pred = X @ self.weights + self.bias
        probs = np.exp(pred - np.max(pred, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def fit(self, X, y, steps=10000, lr=1e-3, log_interval=1000):
        for step in range(steps):
            pred = self.__call__(X)
            
            grad_w = X.T @ (pred - y) / len(y)
            grad_b = np.mean(pred - y)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

            if step % log_interval == 0:
                print(f"{step}/{steps} | Loss: {crossentropy(y, pred)}")



# Training Simulation
NUM_SAMPLES = 1000
NUM_FEATURES = 20
NUM_CLASSES = 3

X, y = make_classification(
    n_samples=NUM_SAMPLES,
    n_features=NUM_FEATURES,
    n_classes=NUM_CLASSES,
    n_clusters_per_class=1,
    random_state=42)

y = onehot_encoding(y, NUM_CLASSES)

model = MultinomialLogisticRegression(NUM_FEATURES, NUM_CLASSES)
model.fit(X, y)


