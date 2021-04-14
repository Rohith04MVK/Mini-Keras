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
    __slots__ = ("cache", "beta", "epsilon")

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


class Adam(BaseOptimizer):
    __slots__ = ("v", "s", "beta1", "beta2", "epsilon")

    def __init__(self, trainable_layers, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        BaseOptimizer.__init__(self, trainable_layers)
        self.v = {}
        self.s = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self) -> None:
        for layer in self.trainable_layers:
            w, b = layer.get_params()
            w_shape = w.shape
            b_shape = b.shape
            self.v[('dw', layer)] = np.zeros(w_shape)
            self.v[('db', layer)] = np.zeros(b_shape)
            self.s[('dw', layer)] = np.zeros(w_shape)
            self.s[('db', layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step):
        v_correction_term = 1 - np.power(self.beta1, step)
        s_correction_term = 1 - np.power(self.beta2, step)
        s_corrected = {}
        v_corrected = {}

        for layer in self.trainable_layers:
            layer_dw = ('dw', layer)
            layer_db = ('db', layer)

            self.v[layer_dw] = (self.beta1 * self.v[layer_dw] + (1 - self.beta1) * w_grads[layer])
            self.v[layer_db] = (self.beta1 * self.v[layer_db] + (1 - self.beta1) * b_grads[layer])

            v_corrected[layer_dw] = self.v[layer_dw] / v_correction_term
            v_corrected[layer_db] = self.v[layer_db] / v_correction_term

            self.s[layer_dw] = (self.beta2 * self.s[layer_dw] + (1 - self.beta2) * np.square(w_grads[layer]))
            self.s[layer_db] = (self.beta2 * self.s[layer_db] + (1 - self.beta2) * np.square(b_grads[layer]))

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = (learning_rate * v_corrected[layer_dw] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon))
            db = (learning_rate * v_corrected[layer_db] / (np.sqrt(s_corrected[layer_db]) + self.epsilon))

            layer.update_params(dw, db)
