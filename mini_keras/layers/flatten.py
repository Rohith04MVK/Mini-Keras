from functools import reduce

from mini_keras.layers import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__
        self.original_dim = None
        self.output_dim = None