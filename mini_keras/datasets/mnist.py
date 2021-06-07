import numpy as np

from ..utils.data_utils import get_file


def load_data():
    file_path = get_file("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz")

    with np.load(file_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)
