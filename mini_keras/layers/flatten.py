import typing as t

import numpy as np

from ..base import BaseLayer


class Flatten(BaseLayer):
    __slots__ = ("original_dim", "output_dim")

    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def initialize(self, input_dims: t.Union[int, tuple]) -> None:
        """
        Parameters
        ----------
        input_dims : int or tuple
            Shape of the input data.
        """
        raise NotImplementedError

    def forward(self, a_prev: np.ndarray, training: bool) -> None:
        """
        a_prev : numpy.ndarray
            The input to this layer which corresponds to the previous layer's activations.
        training : bool
            Whether the model in which this layer is in is training.
        """
        raise NotImplementedError

    def backward(self, da: np.ndarray) -> None:
        """
        da : numpy.ndarray
            The gradients wrt the cost of this layer activations
        """
        raise NotImplementedError

    def update_params(self, dw: np.ndarray, db: np.ndarray) -> None:
        """
        dw : numpy.ndarray
            The gradients wrt the cost of this layer's weights.
        db : numpy.ndarray
            The gradients wrt the cost of this layer's biases.
        """
        raise NotImplementedError

    def get_params(self) -> None:
        raise NotImplementedError

    def get_output_dim(self) -> None:
        raise NotImplementedError
