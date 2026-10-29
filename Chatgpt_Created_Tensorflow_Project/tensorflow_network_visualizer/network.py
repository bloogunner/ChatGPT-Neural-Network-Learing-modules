"""TensorFlow model and snapshots of its REAL learned parameters."""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class NetworkSnapshot:
    x: float
    target: float
    hidden: list[float]
    prediction: float
    hidden_weights: list[float]
    hidden_biases: list[float]
    output_weights: list[float]
    output_bias: float
    loss: float
    test_prediction: float
    epoch: int


class NetworkTrainer:
    """A tiny linear 1 -> 4 -> 1 network learning y = 2x + 1.

    The four hidden neurons make internal connections visible. Because both
    layers are linear, the network can still learn/extrapolate a straight line.
    """

    def __init__(self, max_epochs: int = 500):
        self.max_epochs = max_epochs
        self.x_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
        self.y_train = 2 * self.x_train + 1
        self.reset()

    def reset(self):
        tf.keras.utils.set_random_seed(7)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(4, activation="linear", name="hidden"),
            tf.keras.layers.Dense(1, activation="linear", name="output"),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            loss="mean_squared_error",
        )
        self.epoch = 0

    def _parameters(self):
        hidden_kernel, hidden_bias = self.model.layers[0].get_weights()
        output_kernel, output_bias = self.model.layers[1].get_weights()
        return hidden_kernel, hidden_bias, output_kernel, output_bias

    def _forward(self, batch):
        """The exact linear calculations made by the Keras layers."""
        w1, b1, w2, b2 = self._parameters()
        hidden = batch @ w1 + b1
        output = hidden @ w2 + b2
        return hidden, output

    def snapshot(self, sample_x: float = 3.0) -> NetworkSnapshot:
        x = float(sample_x)
        inputs = np.array([[x]], dtype=np.float32)
        hidden, output = self._forward(inputs)
        _, train_output = self._forward(self.x_train)
        _, test_output = self._forward(np.array([[10.0]], dtype=np.float32))
        w1, b1, w2, b2 = self._parameters()
        return NetworkSnapshot(
            x=x,
            target=2*x+1,
            hidden=[float(v) for v in hidden[0]],
            prediction=float(output[0, 0]),
            hidden_weights=[float(v) for v in w1[0]],
            hidden_biases=[float(v) for v in b1],
            output_weights=[float(v) for v in w2[:, 0]],
            output_bias=float(b2[0]),
            loss=float(np.mean((train_output - self.y_train)**2)),
            test_prediction=float(test_output[0, 0]),
            epoch=self.epoch,
        )

    def train_epoch(self):
        if self.epoch < self.max_epochs:
            # A single epoch: all 5 training examples, one gradient update.
            self.model.train_on_batch(self.x_train, self.y_train)
            self.epoch += 1
