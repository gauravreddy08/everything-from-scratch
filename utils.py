import numpy as np

def onehot_encoding(y, num_classes):
    return np.eye(num_classes)[y]