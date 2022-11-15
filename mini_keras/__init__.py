from . import _mini_keras as _mk_rust  # Rust extension
from .activations import identity, relu, sigmoid, softmax
from .layers.conv2D import Conv
from .layers.dense import Dense
from .layers.flatten import Flatten
from .loss import sigmoid_cross_entropy, softmax_cross_entropy
from .models.sequential import Sequential
from .optimizer import adam, gradient_descent, rmsprop
from .utils.encoder import one_hot_encoder

__author__ = "Deep alchemy team"
__email__ = "warriordefenderz@gmail.com"
__version__ = "0.1.0"
__license__ = "MIT License"
__copyright__ = "Copyright 2021 Deep alchemy"
