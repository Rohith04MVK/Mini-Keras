import numpy as np

from .abc import BaseActivation


class Identify(BaseActivation):
    def function(self, x):
        return x

    def derivative(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(BaseActivation):
    pass


class ReLU(BaseActivation):
    pass


class Softmax(BaseActivation):
    pass
