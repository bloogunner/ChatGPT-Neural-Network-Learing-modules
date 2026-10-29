"""OpenCV webcam ownership and frame conversion.

Keeping camera operations in their own class makes it easier to switch
cameras later or reuse the video feed in another application.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class Webcam:
    """Open a webcam, read RGB frames, and release hardware cleanly."""

    def __init__(self, index: int = 0) -> None:
        """Store which camera to open, without accessing it immediately.

        Args:
            index: OpenCV camera index. The primary camera is usually zero.
        """
        self.index = index
        self.capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Open the physical camera and validate the resulting capture.

        On Windows, DirectShow is tried first. The ordinary OpenCV backend
        is used as a fallback when DirectShow cannot open the camera.

        Raises:
            RuntimeError: If neither backend can access the selected camera.
        """
        self.stop()

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.index, backend)

        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(self.index, cv2.CAP_ANY)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not open camera {self.index}. Check Windows camera "
                "permissions, the camera index, and other applications."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture = capture

    def read(self) -> np.ndarray | None:
        """Read one frame and convert OpenCV's BGR colors to RGB.

        Returns:
            An RGB uint8 array of shape (height, width, 3), or None when
            no valid frame was received.
        """
        if self.capture is None:
            return None

        success, bgr_frame = self.capture.read()

        if not success or bgr_frame is None:
            return None

        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        """Release the camera so other applications can use it again."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
