import numpy as np

from ..base import BaseLayer


class Pool(BaseLayer):
    def __init__(self, pool_size, stride, mode) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.n_h, self.n_w, self.n_c = None, None, None
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
        self.weights = None
        self.biases = None
        self.mode = mode
        self.cache = {}

    def initialize(self, in_dim) -> None:
        self.n_h_prev, self.n_w_prev, self.n_c_prev = in_dim
        self.n_h = int((self.n_h_prev - self.pool_size) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.pool_size) / self.stride + 1)
        self.n_c = self.n_c_prev
