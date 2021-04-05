import numpy as np
from .abc import BaseOptimizer


class GradientDescent(BaseOptimizer):

    def __init__(self, trainable_layers):
        BaseOptimizer.__init__(trainable_layers)

    def initialize(self):
        raise NotImplementedError

    def update(self, learning_rate, w_grads, b_grads, step):
        for layer in self.trainable_layers:
            layer.update(dw=learning_rate * w_grads[layer],
                         db=learning_rate * b_grads[layer])


class RMSprop(BaseOptimizer):

    def __init__(self, trainable_layers, beta=0.9, epsilon=1e-8):
        BaseOptimizer.__init__(self, trainable_layers)
        self.cache = {}
        self.beta = beta
        self.epsilon = epsilon

    def initialize(self):
        for layer in self.trainable_layers:
            w, b, = layer.get_params
            w_shape = w.shape
            b_shape = b.shape
            self.cache[("dw", layer)] = np.zeros(w_shape)
            self.cache[("db", layer)] = np.zeros(b_shape)
