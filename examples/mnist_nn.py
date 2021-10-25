import numpy as np

# Importing everything we need from Mini-keras liberay
from mini_keras.activations import relu, softmax
from mini_keras.datasets import mnist
from mini_keras.layers.conv2D import Conv
from mini_keras.layers.dense import Dense
from mini_keras.layers.flatten import Flatten
from mini_keras.layers.pool import Pool
from mini_keras.loss import softmax_cross_entropy
from mini_keras.models.sequential import Sequential
from mini_keras.optimizer import adam
from mini_keras.utils.encoder import one_hot_encoder

# Loading the data for training
(x_train, y_train), (x_test, y_test) = mnist.data()


def preprocess(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
    y_train = one_hot_encoder(y_train.reshape(y_train.shape[0], 1), num_classes=10)
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


# Preprocessing the data to make it compatible for training
x_train, y_train, x_test, y_test = preprocess(x_train, y_train, x_test, y_test)

# Making the a cnn using the Sequential model with conv, dense, pool and flatten layers
cnn = Sequential(
    input_dim=(28, 28, 1),
    layers=[
        Conv(5, 1, 32, activation=relu),
        Pool(2, 2, "max"),
        Flatten(),
        Dense(64, relu),
        Dense(10, softmax),
    ],
    cost_function=softmax_cross_entropy,
    optimizer=adam,
)

# Feeding the data through the network aka learning or training
cnn.train(
    x_train,
    y_train,
    mini_batch_size=256,
    learning_rate=0.001,
    num_epochs=30,
    validation_data=(x_test, y_test),
)
