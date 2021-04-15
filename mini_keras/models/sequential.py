import typing as t

from ..base import BaseLayer
from ..loss import SoftmaxCrossEntropy
from ..optimizer import GradientDescent


class Sequential:
    __slots__ = (
        "input_dim", "layers", "w_grads", "b_grads", "cost_function", "optimizer", "l2_lambda", "trainable_layers"
    )

    def __init__(
            self, input_dim, layers=None, cost_function=SoftmaxCrossEntropy, optimizer=GradientDescent, l2_lambda=0
    ):
        self.input_dim = input_dim

        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

        self.w_grads = {}
        self.b_grads = {}
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

        # Initialize the layers in the model providing the input dimension they should expect
        if self.layers:
            self.layers[0].initialize(input_dim)
            for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
                curr_layer.initialize(prev_layer.get_output_dim())

        self.trainable_layers = set(layer for layer in self.layers if layer.get_params() is not None)
        self.optimizer = optimizer(self.trainable_layers)
        self.optimizer.initialize()

    def __call__(self):
        """Initialize by calling the object."""
        self.layers[0].initialize(self.input_dim)

        for prev_layer, curr_layer in zip(self.layers, self.layers[1:]):
            curr_layer.initialize(prev_layer.get_output_dim())

    def add(self, layers: t.Union[list, BaseLayer]):
        if isinstance(layers, list):
            self.layers += layers
        else:
            self.layers.append(layers)

    def forward_prop(self, x, training=True):
        for layer in self.layers:
            a = x
            a = layer.forward(a, training)
