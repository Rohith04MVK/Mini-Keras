import typing as t  # noqa: E902

import numpy as np

from ..activations import *
from ..base import BaseLayer


class Conv(BaseLayer):
    """2D convolutional layer.
    Attributes
    ----------
    kernel_size : int
        Height and Width of the 2D convolution window.
    stride : int
        Stride along height and width of the input volume on which the convolution is applied.
    padding: str
        Padding mode, 'valid' or 'same'.
    pad : int
        Padding size.
    n_h : int
        Height of the output volume.
    n_w : int
        Width of the output volume.
    n_c : int
        Number of channels of the output volume. Corresponds to the number of filters.
    n_h_prev : int
        Height of the input volume.
    n_w_prev : int
        Width of the input volume.
    n_c_prev : int
        Number of channels of the input volume.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    activation : Activation
        Activation function applied to the output volume after performing the convolution operation.
    cache : dict
        Cache.
    """

    def __init__(self, kernel_size, stride, n_c, padding="valid", activation=identity):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pad = None
        self.n_h, self.n_w, self.n_c = None, None, n_c
        self.n_h_prev, self.n_w_prev, self.n_c_prev = None, None, None
        self.w = None
        self.b = None
        self.activation_dict = activation_dict = {
            "relu": relu,
            "softmax": softmax,
            "sigmoid": sigmoid,
            "identity": identity,
        }

        if type(activation) is str:
            if activation in activation_dict:
                self.activation = activation_dict.get(activation.lower())
            else:
                raise (ValueError(f"The activation '{activation} is not supported'"))
        else:
            self.activation = activation

        self.cache = {}

    def init(self, in_dim):
        self.pad = 0 if self.padding == "valid" else int((self.kernel_size - 1) / 2)

        self.n_h_prev, self.n_w_prev, self.n_c_prev = in_dim
        self.n_h = int(
            (self.n_h_prev - self.kernel_size + 2 * self.pad) / self.stride + 1
        )
        self.n_w = int(
            (self.n_w_prev - self.kernel_size + 2 * self.pad) / self.stride + 1
        )

        self.w = np.random.randn(
            self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c
        ).astype("float32")
        self.b = np.zeros((1, 1, 1, self.n_c)).astype("float32")

    def forward(self, a_prev, training):
        batch_size = a_prev.shape[0]
        a_prev_padded = Conv.zero_pad(a_prev, self.pad).astype("float32")
        out = np.zeros((batch_size, self.n_h, self.n_w, self.n_c)).astype("float32")

        # Convolve
        for i in range(self.n_h):
            v_start = i * self.stride
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = j * self.stride
                h_end = h_start + self.kernel_size

                out[:, i, j, :] = np.sum(
                    a_prev_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis]
                    * self.w[np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                )

        z = out + self.b
        z = z.astype("float32")
        a = self.activation.f(z)
        a = a.astype("float32")

        if training:
            # Cache for backward pass
            self.cache.update({"a_prev": a_prev, "z": z, "a": a})

        return a

    def backward(self, da):
        batch_size = da.shape[0]
        a_prev, z, a = (self.cache[key] for key in ("a_prev", "z", "a"))
        a_prev_pad = Conv.zero_pad(a_prev, self.pad) if self.pad != 0 else a_prev

        da_prev = np.zeros(
            (batch_size, self.n_h_prev, self.n_w_prev, self.n_c_prev)
        ).astype("float32")
        da_prev_pad = Conv.zero_pad(da_prev, self.pad) if self.pad != 0 else da_prev

        dz = da * self.activation.df(z, cached_y=a)
        dz = dz.astype("float32")
        db = 1 / batch_size * dz.sum(axis=(0, 1, 2))
        db = db.astype("float32")
        dw = np.zeros((self.kernel_size, self.kernel_size, self.n_c_prev, self.n_c))
        dw = dw.astype("float32")

        # 'Convolve' back
        for i in range(self.n_h):
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.n_w):
                h_start = self.stride * j
                h_end = h_start + self.kernel_size

                da_prev_pad[:, v_start:v_end, h_start:h_end, :] += np.sum(
                    self.w[np.newaxis, :, :, :, :]
                    * dz[:, i : i + 1, j : j + 1, np.newaxis, :],
                    axis=4,
                )

                dw += np.sum(
                    a_prev_pad[:, v_start:v_end, h_start:h_end, :, np.newaxis]
                    * dz[:, i : i + 1, j : j + 1, np.newaxis, :],  # noqa: W503
                    axis=0,
                )

        dw /= batch_size

        if self.pad != 0:
            da_prev = da_prev_pad[:, self.pad : -self.pad, self.pad : -self.pad, :]

        return da_prev, dw, db

    def get_output_dim(self):
        return self.n_h, self.n_w, self.n_c

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    @staticmethod
    def zero_pad(
        x, pad
    ) -> t.Any:  # TODO: Soon to be replaced with the actual return value.
        return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
