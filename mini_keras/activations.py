import math

import numpy as np

from .base import BaseActivation


class Identify(BaseActivation):
    def f(self, x):
        return x

    def df(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(BaseActivation):
    def f(self, x):
        s = 1 / (1 + math.exp(-x))
        return s

    def df(self, x, cached_y=None):
        ds = x * (1 - x)
        return ds


class ReLU(BaseActivation):
    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


class SoftMax(BaseActivation):
    def f(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    def df(self) -> None:
        raise NotImplementedError
