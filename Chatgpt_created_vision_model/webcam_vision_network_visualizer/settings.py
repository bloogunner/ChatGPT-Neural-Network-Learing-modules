"""Shared configuration for the webcam neural-network visualizer."""

from __future__ import annotations


# MobileNetV2 takes RGB images with a resolution of 224 x 224 pixels.
MODEL_INPUT_SIZE = (224, 224)

# These real layer names are inspected for the network-flow overview.
# The full model contains many more layers, all visible in the layer list.
OVERVIEW_LAYERS = (
    "Conv1_relu",
    "expanded_conv_depthwise_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_10_expand_relu",
    "block_13_expand_relu",
    "block_16_project_BN",
    "out_relu",
)

# Showing just a few feature channels keeps the animation understandable.
OVERVIEW_CHANNELS = 5
FEATURE_MAP_COUNT = 6

# Camera and AI inference happen on different schedules.
CAMERA_DELAY_MS = 33
INFERENCE_INTERVAL_SECONDS = 1.2

# Shared dark-theme colors for the application.
BACKGROUND = "#0b1220"
PANEL = "#172235"
ACCENT = "#58c4ff"
TEXT = "#eaf2ff"
MUTED = "#a8b8cb"
SUCCESS = "#7ddfb2"
