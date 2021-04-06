import numpy as np

from mini_keras.activation import identity
from mini_keras.layer import BaseLayer


class Conv2D(BaseLayer):

        def __init__(self, kernel_size, stride, n_c, padding='valid', activation=identity):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.pad = None
            self.n_h, self.n_w, self.n_c = None, None, n_c
            self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
            self.weights = None
            self.baises = None
            self.activation = activation
            self.cache = {}