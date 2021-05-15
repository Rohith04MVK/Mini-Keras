import numpy as np

from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    def __init__(self, trainable_layers):
        super(GradientDescent, self).__init__(trainable_layers)

    def initialize(self):
        pass

    def update(self, learning_rate, w_grads, b_grads, step):
        for layer in self.trainable_layers:
            layer.update_params(
                dw=learning_rate * w_grads[layer], db=learning_rate * b_grads[layer]
            )


class RMSProp(BaseOptimizer):
    def __init__(self, trainable_layers, beta=0.9, epsilon=1e-8):
        super(RMSProp, self).__init__(trainable_layers)
        self.s = {}
        self.beta = beta
        self.epsilon = epsilon

    def initialize(self):
        for layer in self.trainable_layers:
            w, b = layer.get_params()
            w_shape = w.shape
            b_shape = b.shape
            self.s[("dw", layer)] = np.zeros(w_shape)
            self.s[("db", layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step):
        s_corrected = {}
        s_correction_term = 1 - np.power(self.beta, step)

        for layer in self.trainable_layers:
            layer_dw = ("dw", layer)
            layer_db = ("db", layer)

            self.s[layer_dw] = self.beta * self.s[layer_dw] + (
                1 - self.beta
            ) * np.square(w_grads[layer])
            self.s[layer_db] = self.beta * self.s[layer_db] + (
                1 - self.beta
            ) * np.square(b_grads[layer])

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = learning_rate * (
                w_grads[layer] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon)
            )
            db = learning_rate * (
                b_grads[layer] / (np.sqrt(s_corrected[layer_db]) + self.epsilon)
            )

            layer.update_params(dw, db)


class Adam(BaseOptimizer):
    def __init__(self, trainable_layers, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(trainable_layers)
        self.v = {}
        self.s = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self):
        for layer in self.trainable_layers:
            w, b = layer.get_params()
            w_shape = w.shape
            b_shape = b.shape
            self.v[("dw", layer)] = np.zeros(w_shape)
            self.v[("db", layer)] = np.zeros(b_shape)
            self.s[("dw", layer)] = np.zeros(w_shape)
            self.s[("db", layer)] = np.zeros(b_shape)

    def update(self, learning_rate, w_grads, b_grads, step):
        v_correction_term = 1 - np.power(self.beta1, step)
        s_correction_term = 1 - np.power(self.beta2, step)
        s_corrected = {}
        v_corrected = {}

        for layer in self.trainable_layers:
            layer_dw = ("dw", layer)
            layer_db = ("db", layer)

            self.v[layer_dw] = (
                self.beta1 * self.v[layer_dw] + (1 - self.beta1) * w_grads[layer]
            )
            self.v[layer_db] = (
                self.beta1 * self.v[layer_db] + (1 - self.beta1) * b_grads[layer]
            )

            v_corrected[layer_dw] = self.v[layer_dw] / v_correction_term
            v_corrected[layer_db] = self.v[layer_db] / v_correction_term

            self.s[layer_dw] = self.beta2 * self.s[layer_dw] + (
                1 - self.beta2
            ) * np.square(w_grads[layer])
            self.s[layer_db] = self.beta2 * self.s[layer_db] + (
                1 - self.beta2
            ) * np.square(b_grads[layer])

            s_corrected[layer_dw] = self.s[layer_dw] / s_correction_term
            s_corrected[layer_db] = self.s[layer_db] / s_correction_term

            dw = (
                learning_rate * v_corrected[layer_dw] / (np.sqrt(s_corrected[layer_dw]) + self.epsilon)
            )
            db = (
                learning_rate * v_corrected[layer_db] / (np.sqrt(s_corrected[layer_db]) + self.epsilon)
            )

            layer.update_params(dw, db)


# -- Assign to the short forms --
adam = Adam
rmsprop = RMSProp
gradient_descent = GradientDescent
