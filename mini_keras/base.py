import abc
import typing as t

import numpy as np


class BaseCostFunction(metaclass=abc.ABCMeta):
    __slots__ = ()

    @classmethod
    def __subclasshook__(
        cls, subclass: "BaseCostFunction"
    ) -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, "f")
            and callable(subclass.f)  # noqa: W503
            and hasattr(subclass, "grad")  # noqa: W503
            and callable(subclass.grad)  # noqa: W503
            or NotImplemented  # noqa: W503
        )

    @abc.abstractmethod
    def f(self, y_pred, y: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def grad(self, y_pred, y: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError


class BaseActivation(metaclass=abc.ABCMeta):
    __slots__ = ()

    @classmethod
    def __subclasshook__(
        cls, subclass: "BaseActivation"
    ) -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, "f")
            and callable(subclass.f)  # noqa: W503
            and hasattr(subclass, "df")  # noqa: W503
            and callable(subclass.df)  # noqa: W503
            or type(NotImplemented)  # noqa: W503
        )

    @abc.abstractmethod
    def f(self, x: t.Union[t.List, np.ndarray]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def df(self, x: t.Union[t.List, np.ndarray], cached_y=None) -> None:
        raise NotImplementedError


class BaseOptimizer(metaclass=abc.ABCMeta):
    __slots__ = ("trainable_layers",)

    def __init__(self, trainable_layers) -> None:
        self.trainable_layers = trainable_layers

    @classmethod
    def __subclasshook__(
        cls, subclass: "BaseOptimizer"
    ) -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, "initialize")
            and callable(subclass.initialize)  # noqa: W503
            and hasattr(subclass, "update")  # noqa: W503
            and callable(subclass.update)  # noqa: W503
            or NotImplemented  # noqa: W503
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
    def __subclasshook__(
        cls, subclass: "BaseOptimizer"
    ) -> t.Union[bool, type(NotImplemented)]:
        return (
            hasattr(subclass, "initialize")
            and callable(subclass.initialize)  # noqa: W503
            and hasattr(subclass, "forward")  # noqa: W503
            and callable(subclass.forward)  # noqa: W503
            and hasattr(subclass, "backward")  # noqa: W503
            and callable(subclass.backward)  # noqa: W503
            and hasattr(subclass, "update_params")  # noqa: W503
            and callable(subclass.update_params)  # noqa: W503
            and hasattr(subclass, "get_params")  # noqa: W503
            and callable(subclass.get_params)  # noqa: W503
            and hasattr(subclass, "get_output_dim")  # noqa: W503
            and callable(subclass.get_output_dim)  # noqa: W503
            or NotImplemented  # noqa: W503
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

    def get_params(self) -> None:
        raise NotImplementedError

    def get_output_dim(self) -> None:
        raise NotImplementedError
