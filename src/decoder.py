import tensorflow as tf
from tensorflow.keras import layers, Model

class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(256, activation="relu")
        self.dense2 = layers.Dense(28 * 28, activation="sigmoid")
        self.reshape_layer = layers.Reshape((28, 28, 1))

    def call(self, z):
        z = self.dense1(z)
        z = self.dense2(z)
        x_hat = self.reshape_layer(z)
        return x_hat
