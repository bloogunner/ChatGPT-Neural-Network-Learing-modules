"""Optional two-class transfer-learning head for labeled webcam snapshots.

Training the entire ImageNet network from scratch on a laptop would be slow
and would need a large dataset. Transfer learning is more practical: keep the
pretrained vision backbone frozen, then train a small head on camera examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingState:
    """Measurements needed to visualize actual trainable connection weights."""

    class_counts: tuple[int, int]
    epochs: int
    latest_loss: float | None
    weights: np.ndarray
    biases: np.ndarray
    ready_for_prediction: bool


class CameraHeadTrainer:
    """Train a small six-hidden-neuron classifier on camera embeddings."""

    def __init__(self, tensorflow, feature_count: int = 1280) -> None:
        """Create the head and initialize empty data for two user categories.

        Args:
            tensorflow: TensorFlow module already loaded by VisionModel.
            feature_count: Number of pooled features from MobileNetV2.
        """
        self.tf = tensorflow
        self.feature_count = feature_count
        self._make_head()
        self.examples: list[np.ndarray] = []
        self.labels: list[int] = []
        self.epochs = 0
        self.latest_loss: float | None = None

    def _make_head(self) -> None:
        """Create a trainable 1280 -> 6 -> 2 dense neural network.

        The six-to-two output weight matrix is small enough to draw every
        connection with its real learned weight value.
        """
        self.head = self.tf.keras.Sequential(
            [
                self.tf.keras.Input(shape=(self.feature_count,)),
                self.tf.keras.layers.Dense(6, activation="relu"),
                self.tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
        self.head.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def add_example(self, embedding: np.ndarray, label: int) -> TrainingState:
        """Store one labeled frozen-backbone feature vector in RAM.

        The webcam image itself is not saved to disk. Only a compact numeric
        embedding and an integer class label are retained in memory.
        """
        if label not in (0, 1):
            raise ValueError("Class label must be 0 (A) or 1 (B).")

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if len(vector) != self.feature_count:
            raise ValueError("Embedding length is incompatible with this head.")

        self.examples.append(vector.copy())
        self.labels.append(label)
        return self.state()

    def train(self, number_of_epochs: int = 1) -> list[TrainingState]:
        """Update actual dense-layer weights from labeled captured examples.

        Returns:
            A snapshot after every epoch, allowing the dashboard to display
            the learning trajectory instead of only the final parameters.

        Raises:
            ValueError: When fewer than three examples per class exist.
        """
        if self.labels.count(0) < 3 or self.labels.count(1) < 3:
            raise ValueError("Capture at least 3 examples of EACH class first.")

        features = np.stack(self.examples).astype(np.float32)
        labels = np.asarray(self.labels, dtype=np.int32)
        states = []

        for _ in range(number_of_epochs):
            # Each epoch makes one gradient-based update on the whole dataset.
            metrics = self.head.train_on_batch(
                features,
                labels,
                return_dict=True,
            )
            self.epochs += 1
            self.latest_loss = float(metrics["loss"])
            states.append(self.state())

        return states

    def predict(self, embedding: np.ndarray) -> list[float] | None:
        """Classify a frame with the custom head, only after training."""
        if self.epochs == 0:
            return None

        features = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        scores = self.head(features, training=False).numpy()[0]
        return [float(value) for value in scores]

    def reset(self) -> TrainingState:
        """Discard captured examples and create new random trainable weights."""
        self.examples.clear()
        self.labels.clear()
        self.epochs = 0
        self.latest_loss = None
        self._make_head()
        return self.state()

    def state(self) -> TrainingState:
        """Copy the real 6x2 final weight matrix for thread-safe drawing."""
        weights, biases = self.head.layers[-1].get_weights()

        return TrainingState(
            class_counts=(self.labels.count(0), self.labels.count(1)),
            epochs=self.epochs,
            latest_loss=self.latest_loss,
            weights=weights.copy(),
            biases=biases.copy(),
            ready_for_prediction=self.epochs > 0,
        )
