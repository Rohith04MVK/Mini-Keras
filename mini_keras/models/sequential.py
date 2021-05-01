import typing as t
from functools import reduce

import numpy as np

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
        """
        Performs a forward propagation pass.

        Parameters
        ----------
        x : numpy.ndarray
            Input of the first layer.
        training : bool
            Is the model is training or not.
        Returns
        -------
        numpy.ndarray
            Model's output, corresponding to the last layer's activations.
        """
        for layer in self.layers:
            a = x
            a = layer.forward(a, training)
            return a

    def backward_prop(self, a_last, y):
        """
                Performs a backward propagation pass.
        Parameters
        ----------
        a_last : numpy.ndarray
            Last layer's activations.
        y : numpy.ndarray
            Target labels.
        """
        da = self.cost_function.grad(a_last, y)
        batch_size = da.shape[0]
        for layer in reversed(self.layers):
            da_prev, dw, db = layer.backward(da)

            if layer in self.trainable_layers:
                if self.l2_lambda != 0:
                    # Update the weights' gradients also wrt the l2 regularization cost
                    self.w_grads[layer] = dw + (self.l2_lambda / batch_size) * layer.get_params()[0]
                else:
                    self.w_grads[layer] = dw

                self.b_grads[layer] = db

            da = da_prev

    def update_params(self, learning_rate, step):
        """
        Updates the weighta and biases according to the optimizer
        """
        self.optimizer.update(learning_rate, self.w_grads, self.b_grads, step)

    def compute_cost(self, a_last, y):
        """
        Computes the cost, given the output and the target labels.

        Parameters
        ----------
        a_last : numpy.ndarray
            Output.
        y : numpy.ndarray
            Target labels.

        Returns
        -------
        float
            The cost.
        """
        cost = self.cost_function.f(a_last, y)
        if self.l2_lambda != 0:
            batch_size = y.shape[0]
            weights = [layer.get_params()[0] for layer in self.trainable_layers]
            l2_cost = (self.l2_lambda / (2 * batch_size)) * reduce(lambda ws, w: ws + np.sum(np.square(w)), weights, 0)
            return cost + l2_cost
        else:
            return cost

    @staticmethod
    def create_mini_batches(x, y, mini_batch_size) -> list:

        batch_size = x.shape[0]
        mini_batches = []

        p = np.random.permutation(x.shape[0])
        x, y = x[p, :], y[p, :]
        num_complete_minibatches = batch_size // mini_batch_size

        for k in range(0, num_complete_minibatches):
            mini_batches.append((
                x[k * mini_batch_size:(k + 1) * mini_batch_size, :],
                y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
            ))

        # Fill with remaining data, if needed
        if batch_size % mini_batch_size != 0:
            mini_batches.append((
                x[num_complete_minibatches * mini_batch_size:, :],
                y[num_complete_minibatches * mini_batch_size:, :]
            ))

        return mini_batches

    def train_step(self, x_train, y_train, learning_rate, step):
        a_last = self.forward_prop(x_train, training=True)
        self.backward_prop(a_last, y_train)
        cost = self.compute_cost(a_last, y_train)
        self.update_params(learning_rate, step)
        return cost
