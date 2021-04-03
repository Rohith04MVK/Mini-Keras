import numpy as np


class ActivationFunction:
    def function(self, x):
        raise NotImplementedError

    def derivative(self, x, cached_y=None):
        raise NotImplementedError


class Identify(ActivationFunction):
    def function(self, x):
        return x

    def derivative(self, x, cached_y=None):
        return np.full(x.shape, 1)
