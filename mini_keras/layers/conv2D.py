import numpy as np

from mini_keras.layer import BaseLayer
from mini_keras.activation import identity


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

    def initialize(self, input_dims):
        self.pad = 0 if self.padding == 'valid' else int((self.kernel_size - 1) / 2)

        self.n_h_prev, self.n_w_prev, self.n_c_prev = input_dims

        self.n_h = int((self.n_h_prev - self.kernel_size + 2 * self.pad) / self.stride + 1)
        self.n_w = int((self.n_w_prev - self.kernel_size + 2 * self.pad) / self.stride + 1)

        self.weights = np.random.randn(self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c)
        self.baises = np.zeros((1, 1, 1, self.n_c))

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a_prev_padded = Conv2D.zero_pad(a_prev, self.pad)
        out = np.zeros((batch_size, self.n_h, self.n_w, self.n_c))

        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = i * v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                out[:, i, j, :] = np.sum(a_prev_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis]
                                         * self.weights[np.newaxis, :, :, :], axis=(1, 2, 3))

        z = out + self.b
        a = self.activation.f(z)

        if training:
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        batch_size = da.shape[0]
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        a_prev_pad = Conv2D.zero_pad(a_prev, self.pad) if self.pad != 0 else a_prev

        da_prev = np.zeros((batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev))
        da_prev_pad = Conv2D.zero_pad(da_prev, self.pad) if self.pad != 0 else da_prev

        dz = da * self.activation.df(z, cached_y=a)
        db = 1 / batch_size * dz.sum(axis=(0, 1, 2))
        dw = np.zeros((self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c))

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    @staticmethod
    def zero_pad(x, pad):
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
