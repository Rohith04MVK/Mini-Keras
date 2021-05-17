import numpy as np


def one_hot_encoder(x, num_classes):
    out = np.zeros((x.shape[0], num_classes))
    out[np.arange(x.shape[0]), x[:, 0]] = 1
    return out
