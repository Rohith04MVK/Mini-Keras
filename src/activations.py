import numpy as np


class ActivationFunction:
    def function(self, x):
        raise NotImplementedError

    def derivative(self, x, cached_y=None):
        raise NotImplementedError
