import numpy as np

from .abc import BaseActivation


class Identify(BaseActivation):
    def f(self, x):
        return x

    def df(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(BaseActivation):
    pass


class ReLU(BaseActivation):
    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


class Softmax(BaseActivation):
    pass
