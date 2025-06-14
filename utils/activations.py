import numpy as np

class ReLU:
    def forward(X):
        return np.maximum(X, 0)
    def backward(X):
        return (X > 0).astype(float)
    
class Sigmoid:
    def forward(X):
        return 1 / ( 1 + np.exp(-X))
    def backward(X):
        s = 1 / ( 1 + np.exp(-X))
        return s * ( 1 - s )