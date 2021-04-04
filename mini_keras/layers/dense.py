import numpy as np

from .abc import BaseLayer


class Dense(BaseLayer):
    def __init__(self, size, activation):
        super().__init__
        self.size = size
        self.activation = activation
        self.weights = None
        self.bias = None
        self.cache = {}

    def init(self, input_dims):
        self.weights = np.random.randn(self.size, input_dims) * np.sqrt(2 / input_dims)

        self.bias = np.zeros((1, self.size))
