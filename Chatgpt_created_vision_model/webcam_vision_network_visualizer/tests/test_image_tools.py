"""Pure NumPy/OpenCV tests that do not require a camera or TensorFlow."""

from __future__ import annotations

import unittest

import numpy as np

from vision_model import extract_heatmaps, render_heatmap, select_even_channels


class ImageToolsTests(unittest.TestCase):
    """Verify that probing helpers produce valid display data."""

    def test_even_channels(self) -> None:
        """Channel sampling should be bounded and monotonic."""
        self.assertEqual(select_even_channels(3, 6), [0, 1, 2])
        self.assertEqual(select_even_channels(1, 5), [0])
        self.assertEqual(select_even_channels(0, 5), [])
        self.assertEqual(select_even_channels(10, 3), [0, 4, 9])

    def test_heatmap_for_constant_features(self) -> None:
        """Constant activation channels should not divide by zero."""
        image = render_heatmap(np.ones((7, 7), dtype=np.float32))
        self.assertEqual(image.shape, (100, 148, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_convolution_output(self) -> None:
        """A spatial tensor should create six real channel previews."""
        activations = np.random.default_rng(42).normal(
            size=(1, 7, 7, 32)
        ).astype(np.float32)
        heatmaps = extract_heatmaps(activations)
        self.assertEqual(len(heatmaps), 6)
        self.assertEqual(heatmaps[0].rgb_image.shape, (100, 148, 3))

    def test_dense_output(self) -> None:
        """A pooled vector should be labeled as stripe segments."""
        activations = np.arange(100, dtype=np.float32).reshape(1, 100)
        self.assertEqual(len(extract_heatmaps(activations)), 6)


if __name__ == "__main__":
    unittest.main()
