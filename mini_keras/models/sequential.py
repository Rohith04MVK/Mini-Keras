import typing as t


class Sequential:
    __slots__ = ("layers",)

    def __init__(self, layers: t.Optional[list] = None) -> None:
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []

    def add(self, layer) -> None:
        self.layers.append(layer)
