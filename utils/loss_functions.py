import numpy as np

# Linear Loss Functions
def mean_squared_error(y, y_pred):
  return np.mean(np.square(y_pred - y))

def root_mean_squared_error(y, y_pred):
  return np.sqrt(mean_squared_error(y, y_pred))

def mean_absolute_error(y, y_pred):
  return np.mean(np.absolute(y_pred - y))

def r2(y, y_pred):
  return 1 - (np.sum(np.square(y - y_pred)) / np.sum(np.square(y - np.mean(y))))

# Non Linear Loss Functions
def binary_crossentropy(y, y_pred):
  return np.mean(-(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

def crossentropy(y, y_pred):
  probs = np.sum(y * y_pred, axis=1)
  return - np.mean(np.log(probs + 1e-9))

def accuracy(y, y_pred):
  return sum(np.round(y_pred) == y) / len(y)

