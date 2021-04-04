import numpy as np

from mini_keras.abc import BaseLayer


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

    def forward(self, a_prev, training):
        z = np.dot(a_prev, self.weights.T) + self.bias
        a = self.activation.f(z)

        if training:
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))

        batch_size = a_prev.shape[0]

        dz = da * self.activation.df(z, cached_y=a)

        dw = 1 / batch_size * np.dot(dz.T, a_prev)
        db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
        da_prev = np.dot(dz, self.weights)

        return da_prev, dw, db

    def update_params(self, dw, db):
        self.weights -= dw
        self.bias -= db

    def get_params(self):
        return self.w, self.b
