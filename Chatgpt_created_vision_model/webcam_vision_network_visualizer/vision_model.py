"""A pretrained TensorFlow vision model and real activation probes.

This application performs INFERENCE, not training: its pretrained weights
remain unchanged. The output heatmaps contain intermediate feature values
produced by the actual camera frame, not invented visual effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from settings import (
    FEATURE_MAP_COUNT,
    MODEL_INPUT_SIZE,
    OVERVIEW_CHANNELS,
    OVERVIEW_LAYERS,
)


@dataclass
class LayerDescription:
    """Human-readable metadata about one actual Keras layer."""

    name: str
    kind: str
    output_shape: str


@dataclass
class LayerActivity:
    """Measured responses from selected feature channels in a layer."""

    name: str
    channel_indices: list[int]
    channel_values: list[float]
    tensor_shape: str


@dataclass
class Heatmap:
    """A single feature channel rendered as RGB pixels for display."""

    channel: int
    mean_absolute_activation: float
    rgb_image: np.ndarray


@dataclass
class InferenceResult:
    """All the data needed to update the UI after one camera inference."""

    predictions: list[tuple[str, float]]
    overview: dict[str, LayerActivity]
    inspected_layer: str
    inspected_shape: str
    heatmaps: list[Heatmap]
    embedding: np.ndarray
    custom_scores: list[float] | None = None


def select_even_channels(channel_count: int, samples: int) -> list[int]:
    """Choose well-spaced channel indices without exceeding tensor bounds.

    Convolutional layers may contain hundreds of channels. Drawing every
    channel would be overwhelming, so we sample a small, named subset.
    """
    if channel_count <= 0:
        return []

    count = min(channel_count, samples)
    return np.linspace(0, channel_count - 1, count, dtype=int).tolist()


def render_heatmap(matrix: np.ndarray) -> np.ndarray:
    """Turn one real activation matrix into a displayable false-color image.

    Colors are NOT RGB colors perceived by the neuron. They are a visual
    encoding of relative feature values inside this one channel. Normalizing
    each image separately makes fine spatial patterns easier to see.
    """
    values = np.nan_to_num(
        np.asarray(matrix, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum - minimum < 1e-8:
        scaled = np.zeros_like(values, dtype=np.uint8)
    else:
        scaled = np.clip(
            (values - minimum) / (maximum - minimum) * 255.0,
            0,
            255,
        ).astype(np.uint8)

    # INTERPOLATION_NEAREST exposes the original low-resolution feature grid.
    scaled = cv2.resize(scaled, (148, 100), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def extract_heatmaps(tensor: np.ndarray) -> list[Heatmap]:
    """Extract up to six actual spatial maps or classifier-vector segments.

    Convolutional layer output: (1, height, width, channels).
    Dense/pooled layer output: (1, features).

    A one-dimensional vector does not have a spatial heatmap. For such
    layers it is displayed as a repeated stripe, NOT a spatial feature map.
    """
    sample = np.asarray(tensor)[0]

    if sample.ndim == 3:
        indices = select_even_channels(sample.shape[-1], FEATURE_MAP_COUNT)

        return [
            Heatmap(
                channel=index,
                mean_absolute_activation=float(np.mean(np.abs(sample[:, :, index]))),
                rgb_image=render_heatmap(sample[:, :, index]),
            )
            for index in indices
        ]

    if sample.ndim == 1:
        # Each preview is a different slice of the vector, not a CNN map.
        segments = np.array_split(sample, min(FEATURE_MAP_COUNT, len(sample)))
        output = []

        for index, segment in enumerate(segments):
            output.append(
                Heatmap(
                    channel=index,
                    mean_absolute_activation=float(np.mean(np.abs(segment))),
                    rgb_image=render_heatmap(np.tile(segment.reshape(1, -1), (12, 1))),
                )
            )

        return output

    # Some rare layers can have shapes other than spatial maps or vectors.
    flattened = sample.reshape(-1)
    return extract_heatmaps(flattened.reshape(1, -1))


def summarize_activity(name: str, tensor: np.ndarray) -> LayerActivity:
    """Sample channel magnitudes from a real intermediate Keras tensor.

    Magnitudes are means of absolute activations over spatial positions.
    Since layers use different numeric scales, the UI normalizes each set
    independently rather than comparing absolute magnitudes across layers.
    """
    sample = np.asarray(tensor)[0]

    if sample.ndim == 3:
        channel_means = np.mean(np.abs(sample), axis=(0, 1))
    else:
        channel_means = np.abs(sample.reshape(-1))

    indices = select_even_channels(len(channel_means), OVERVIEW_CHANNELS)

    return LayerActivity(
        name=name,
        channel_indices=indices,
        channel_values=[float(channel_means[i]) for i in indices],
        tensor_shape=str(tuple(sample.shape)),
    )


class VisionModel:
    """Load MobileNetV2 and expose activations from its actual layers."""

    def __init__(self) -> None:
        """Load ImageNet weights and configure the initial activation probe.

        The first run downloads the public pretrained weights and the
        ImageNet class-name mapping into Keras's local cache. This class
        never accesses the webcam or uploads camera frames to a service.
        """
        import tensorflow as tf

        self.tf = tf
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=True,
            weights="imagenet",
        )

        available = {layer.name for layer in self.model.layers}
        self.overview_layers = [
            name for name in OVERVIEW_LAYERS if name in available
        ]

        if not self.overview_layers:
            raise RuntimeError(
                "Expected MobileNetV2 activation layers were not found."
            )

        self.inspected_layer = self.overview_layers[0]
        self.pool_layer = self.model.get_layer("global_average_pooling2d")
        self.encoder = tf.keras.Model(self.model.input, self.pool_layer.output)
        self.probe = None
        self._build_probe()

    def describe_layers(self) -> list[LayerDescription]:
        """Return EVERY layer from the real, loaded Keras model graph.

        This makes the full network visible in the architecture explorer,
        even though only selected activations are captured per frame.
        """
        descriptions = []

        for layer in self.model.layers:
            descriptions.append(
                LayerDescription(
                    name=layer.name,
                    kind=type(layer).__name__,
                    output_shape=str(tuple(layer.output.shape)),
                )
            )

        return descriptions

    def set_inspected_layer(self, name: str) -> None:
        """Rebuild the TensorFlow probe to expose any selected layer output.

        The original model and all pretrained weights stay unchanged.
        Only the list of tensors returned during inference is modified.
        """
        if name == self.inspected_layer:
            return

        try:
            self.model.get_layer(name)
        except ValueError as error:
            raise ValueError(f"Unknown neural-network layer: {name}") from error

        self.inspected_layer = name
        self._build_probe()

    def _build_probe(self) -> None:
        """Create one Keras model that returns multiple intermediate tensors.

        Evaluating all requested outputs in a single forward pass avoids
        rerunning the entire pretrained neural network for each heatmap.
        """
        self.requested_layers = list(self.overview_layers)

        if self.inspected_layer not in self.requested_layers:
            self.requested_layers.append(self.inspected_layer)

        outputs = [
            self.model.get_layer(name).output
            for name in self.requested_layers
        ]
        # Always expose the pooled embedding for an optional training head.
        outputs.append(self.pool_layer.output)
        outputs.append(self.model.output)

        self.probe = self.tf.keras.Model(
            inputs=self.model.input,
            outputs=outputs,
        )

    def encode(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Run the frozen CNN once to get the 1,280 learned image features.

        Capturing labels uses this function to store features for training a
        small classifier. The complete image never needs to be retained.
        """
        resized = cv2.resize(rgb_frame, MODEL_INPUT_SIZE)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        batch = self.tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        return self.encoder(batch, training=False).numpy()[0]

    def analyze(self, rgb_frame: np.ndarray) -> InferenceResult:
        """Classify one real camera frame and gather intermediate activations.

        Steps:
            1. Resize the camera's RGB image to MobileNetV2's input size.
            2. Convert pixels to float32 and scale them to [-1, 1].
            3. Run ONE forward pass through the frozen neural network.
            4. Decode ImageNet top-five classification probabilities.
            5. Extract channel activity and selected-layer feature previews.
        """
        resized = cv2.resize(rgb_frame, MODEL_INPUT_SIZE)
        batch = np.expand_dims(resized.astype(np.float32), axis=0)
        batch = self.tf.keras.applications.mobilenet_v2.preprocess_input(batch)

        tensors = self.probe(batch, training=False)
        tensors = [tensor.numpy() for tensor in tensors]
        intermediate = dict(zip(self.requested_layers, tensors[:-2]))
        embedding = tensors[-2][0].copy()
        probabilities = tensors[-1]

        decoded = self.tf.keras.applications.mobilenet_v2.decode_predictions(
            probabilities,
            top=5,
        )[0]

        predictions = [
            (description.replace("_", " "), float(score))
            for _imagenet_id, description, score in decoded
        ]

        overview = {
            name: summarize_activity(name, intermediate[name])
            for name in self.overview_layers
        }
        inspected = intermediate[self.inspected_layer]

        return InferenceResult(
            predictions=predictions,
            overview=overview,
            inspected_layer=self.inspected_layer,
            inspected_shape=str(tuple(inspected.shape[1:])),
            heatmaps=extract_heatmaps(inspected),
            embedding=embedding,
        )
