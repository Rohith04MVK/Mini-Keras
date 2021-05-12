import typing as t
from functools import reduce

import numpy as np

from ..base import BaseLayer


class Flatten(BaseLayer):
    __slots__ = ("original_dim", "output_dim")

    def __init__(self):
        super().__init__()
        self.original_dim = None
        self.output_dim = None

    def initialize(self, in_dim):
        self.original_dim = in_dim
        self.output_dim = reduce(lambda x, y: x * y, self.original_dim)

    def forward(self, a_prev, training):
        return a_prev.reshape(a_prev.shape[0], -1)

    def backward(self, da):
        return da.reshape(da.shape[0], *self.original_dim), None, None

    def get_params(self):
        pass

    def update_params(self, dw, db):
        pass

    def get_output_dim(self):
        return self.output_dim
