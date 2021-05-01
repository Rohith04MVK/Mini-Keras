import math

import numpy as np

from .base import BaseActivation


class Identify(BaseActivation):
    def f(self, x):
        return x

    def df(self, x, cached_y=None):
        return np.full(x.shape, 1)


class Sigmoid(BaseActivation):
    """
    Applies the sigmoid activation function. For small values (<-5),
    `sigmoid` returns a value close to zero, and for large values (>5)
    the result of the function gets close to 1.
    """

    def f(self, x):
        s = 1 / (1 + math.exp(-x))
        return s

    def df(self, x, cached_y=None):
        ds = x * (1 - x)
        return ds


class ReLU(BaseActivation):
    """
    Does the rectified linear unit function.

    Returns 0 if the number is lesser than 0
    Returns the number if the number is greater than 0

    Example
    -------
    relu = mini_keras.activations.ReLU()
    x = np.array([0.5, 10, -1, 5])
    relu(x)
    >>> [0.5 10.0 0.0 5.0]
    """

    def f(self, x):
        return np.maximum(0, x)

    def df(self, x, cached_y=None):
        return np.where(x <= 0, 0, 1)


class SoftMax(BaseActivation):
    """
    Softmax converts a vector of values to a probability distribution.
    The elements of the output vector are in range (0, 1) and sum to 1.

    Softmax is mostly used as the activation for the last
    layer of a classification network to get the result as
    a probability distribution.
    """

    def f(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))
        return y / np.sum(y, axis=1, keepdims=True)

    def df(self) -> None:
        raise NotImplementedError

identity = Identify()
sigmoid = Sigmoid()
relu = ReLU()
softmax = SoftMax()
