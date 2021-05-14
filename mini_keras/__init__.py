import typing as t

from .activations import Identify, ReLU, Sigmoid, SoftMax
from .layers.conv2D import Conv2D
from .layers.dense import Dense
from .layers.flatten import Flatten
from .loss import SoftmaxCrossEntropy
from .models.sequential import Sequential
from .optimizer import Adam, GradientDescent, RMSprop

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
    "Identify",
    "ReLU",
    "Sigmoid",
    "SoftMax",
    "Conv2D",
    "Dense",
    "Flatten",
    "SoftmaxCrossEntropy",
    "Sequential",
    "Adam",
    "GradientDescent",
    "RMSprop",
)
