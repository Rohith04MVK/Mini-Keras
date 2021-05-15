import numpy as np

from .base import BaseCostFunction

epsilon = 1e-20


class SigmoidCrossEntropy(BaseCostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        # It would be better to have the logits and use this instead
        # max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        cost = (
            -1 / batch_size * (y * np.log(a_last) + (1 - y) * np.log(1 - a_last)).sum()
        )
        return cost

    def grad(self, a_last, y):
        a_last = np.clip(a_last, epsilon, 1.0 - epsilon)
        return -(np.divide(y, a_last) - np.divide(1 - y, 1 - a_last))


class SoftmaxCrossEntropy(BaseCostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        cost = -1 / batch_size * (y * np.log(np.clip(a_last, epsilon, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        return -np.divide(y, np.clip(a_last, epsilon, 1.0))


# -- Assign to the short forms --
softmax_cross_entropy = SoftmaxCrossEntropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()
