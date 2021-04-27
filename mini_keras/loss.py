import typing as t

import numpy as np

from .base import BaseCostFunction

epsilon = 1e-20


class SoftmaxCrossEntropy(BaseCostFunction):
    """
    Computes softmax cross entropy between `prediction` and `labels`.

    Return
    ------
    The Softmax cross entropy error between the labels and the predictions
    """

    def f(self, y_pred, y: t.Union[t.List, np.ndarray]):
        batch_size = y.shape[0]
        cost = -1 / batch_size * (y * np.log(np.clip(y_pred, epsilon, 1.0))).sum()

        return cost

    def grad(self, y_pred, y: t.Union[t.List, np.ndarray]) -> None:
        return - np.divide(y, np.clip(y_pred, epsilon, 1.0))
