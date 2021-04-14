import typing as t

import numpy as np

from functools import reduce
from ..base import BaseLayer


class Flatten(BaseLayer):
    __slots__ = ("original_dim", "output_dim")

    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def initialize(self, input_dims: t.Union[int, tuple]) -> None:
        self.original_dim = input_dims
        self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

    def forward(self, a_prev: np.ndarray, training: bool) -> None:
        return a_prev.reshape(a_prev.shape[0], -1)

    def backward(self, da: np.ndarray) -> None:
        return da.reshape(da.shape[0], *self.original_dim), None, None       

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