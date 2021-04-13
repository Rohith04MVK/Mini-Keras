import numpy as np

from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    def __init__(self, trainable_layers) -> None:
        super().__init__(trainable_layers)

    def initialize(self) -> None:
        raise NotImplementedError

    def update(self, learning_rate, w_grads, b_grads, step) -> None:
        for layer in self.trainable_layers:
            layer.update(
                dw=learning_rate * w_grads[layer],
                db=learning_rate * b_grads[layer]
            )


class RMSprop(BaseOptimizer):
    def __init__(self, trainable_layers, beta=0.9, epsilon=1e-8) -> None:
        super().__init__(trainable_layers)
        self.cache = {}
        self.beta = beta
        self.epsilon = epsilon

    def initialize(self) -> None:
        for layer in self.trainable_layers:
            w, b, = layer.get_params
            w_shape = w.shape
            b_shape = b.shape
            self.cache[("dw", layer)] = np.zeros(w_shape)
            self.cache[("db", layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step) -> None:
        s_corrected = {}
        s_correction_term = 1 - np.power(self.beta, step)

        for layer in self.trainable_layers:
            layer_dw = ("dw", layer)
            layer_db = ("db", layer)

            self.cache[layer_dw] = (self.beta * self.cache[layer_dw] + (1 - self.beta) * np.square(w_grads[layer]))
            self.cache[layer_db] = (self.beta * self.cache[layer_db] + (1 - self.beta) * np.square(b_grads[layer]))

            s_corrected[layer_dw] = self.cache[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.cache[layer_db] / s_correction_term

            dw = (learning_rate * (w_grads[layer] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon)))
            db = (learning_rate * (w_grads[layer] / (np.sqrt(s_corrected[layer_db]) + self.epsilon)))

            layer.update(dw, db)
