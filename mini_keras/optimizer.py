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
