import numpy as np
from sklearn.datasets import make_classification
from utils import ReLU, Sigmoid, binary_crossentropy, accuracy

# This is a Binary Classification Network, with BCE Loss

class MLP:
    def __init__(self, layers):

        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            r = 1 / np.sqrt(layers[i+1])
            self.weights.append(np.random.uniform(-r, r, (layers[i], layers[i+1])))
            self.biases.append(np.random.uniform(-r, r, (1, layers[i+1])))

    def forward(self, X):
        self.A = [X]
        self.Z = []

        for i in range(len(self.weights)):
            Z = self.A[-1] @ self.weights[i] + self.biases[i]
            if i == len(self.weights) - 1:
                A = Sigmoid.forward(Z)
            else:
                A = ReLU.forward(Z)
            
            self.Z.append(Z)
            self.A.append(A)
        
        return self.A[-1]

    def backward(self, y):
        batch_size = y.shape[0]
        # dZ = dL/dA * dA/dZ --> for BCE Loss only
        dZ = (self.A[-1] - y) # (batch_size, L)

        for i in reversed(range(len(self.weights))):
            # dL/dW = dL/dA * dA/dZ * dZ/dW
            # self.A[-1] shape --> (batch_size, L-1)
            dW =  self.A[i].T @ dZ / batch_size # (L-1, L)
            dB = np.sum(dZ, axis=0, keepdims=True) / batch_size # (1, L)

            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * dB

            if i > 0:
                # dZ (batch_size, L) @ (L, L-1) self.weights[i].T 
                dA = dZ @ self.weights[i].T
                dZ = dA * ReLU.backward(self.Z[i-1])
    
    def train(self, X, y, epochs, lr=1e-3, log_interval=0):
        log_interval = epochs//10 if not log_interval else log_interval
        self.lr = lr

        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(y)

            if epoch % log_interval == 0:
                loss = binary_crossentropy(y, y_pred)
                acc = accuracy(np.squeeze(y), np.squeeze(y_pred))
                print(f"Epoch {epoch}/{epochs} | Loss: {loss} Acc: {acc}")


# Training Simulation
X, y = make_classification()
y = np.expand_dims(y, axis=1)
layers = [20, 16, 8, 1]

model = MLP(layers)
model.train(X, y, epochs=10000, lr=0.0003)
