import typing as t

import numpy as np

from .base import BaseCostFunction


class SoftmaxCrossEntropy(BaseCostFunction):
    def f(self, y_pred, y: t.Union[t.List, np.ndarray]):
        epsilon = 1e-20
        batch_size = y.shape[0]
        cost = -1 / batch_size * (y * np.log(np.clip(y_pred, epsilon, 1.0))).sum()

        return cost

    def grad(self, y_pred, y: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError
