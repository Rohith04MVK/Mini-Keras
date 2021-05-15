import typing as t

from .activations import identity, relu, sigmoid, softmax
from .layers.conv2D import Conv
from .layers.dense import Dense
from .layers.flatten import Flatten
from .loss import softmax_cross_entropy, sigmoid_cross_entropy
from .models.sequential import Sequential
from .optimizer import adam, gradient_descent, rmsprop

__author__ = "Deep alchemy team"
__email__ = "warriordefenderz@gmail.com"
__version__ = "0.1.0"
__license__ = "GPL-3.0 License"
__copyright__ = "Copyright 2021 Deep alchemy"

__all__: t.Tuple[str, ...] = (
    "__author__",
    "__email__",
    "__version__",
    "__license__",
    "__copyright__",
    "identity",
    "relu",
    "sigmoid",
    "softmax",
    "Conv",
    "Dense",
    "Flatten",
    "softmax_cross_entropy",
    "sigmoid_cross_entropy",
    "Sequential",
    "adam",
    "gradient_descent",
    "rmsprop",
)
