import abc
import typing as t

import numpy as np


class BaseCostFunction(metaclass=abc.ABCMeta):
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, subclass: "BaseCostFunction") -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, 'f') and callable(subclass.f) and
            hasattr(subclass, 'grad') and callable(subclass.grad) or
            NotImplemented
        )

    @abc.abstractmethod
    def f(self, a_last, y: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def grad(self, a_last, y: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError


class BaseActivation(metaclass=abc.ABCMeta):
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, subclass: "BaseActivation") -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, 'f') and callable(subclass.f) and
            hasattr(subclass, 'df') and callable(subclass.df) or
            type(NotImplemented)
        )

    @abc.abstractmethod
    def f(self, x: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def df(self, x: t.Union[t.List, np.ndarray], cached_y=None) -> None:
        raise NotImplementedError


class BaseOptimizer(metaclass=abc.ABCMeta):
    __slots__ = ('trainable_layers',)

    def __init__(self, trainable_layers) -> None:
        self.trainable_layers = trainable_layers

    @classmethod
    def __subclasshook__(cls, subclass: "BaseOptimizer") -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, 'initialize') and callable(subclass.initialize) and  # noqa: W504
            hasattr(subclass, 'update') and callable(subclass.update) or  # noqa: W504
            NotImplemented
        )

    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, learning_rate, w_grads, b_grads, step) -> None:
        raise NotImplementedError


class BaseLayer:
    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, subclass: "BaseOptimizer") -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, 'initialize') and callable(subclass.initialize) and
            hasattr(subclass, 'forward') and callable(subclass.forward) and
            hasattr(subclass, 'backward') and callable(subclass.backward) and
            hasattr(subclass, 'update_params') and callable(subclass.update_params) and
            hasattr(subclass, 'get_params') and callable(subclass.get_params) and
            hasattr(subclass, 'get_output_dim') and callable(subclass.get_output_dim) or
            NotImplemented
        )

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

    def get_params(self):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError
