"""Background TensorFlow inference AND optional transfer learning.

Tkinter widgets must be updated on the main thread. This worker only sends
plain Python data through a thread-safe queue to the visualizer.
"""

from __future__ import annotations

import queue
import threading
import traceback

import numpy as np

from camera_trainer import CameraHeadTrainer
from vision_model import VisionModel


class VisionWorker:
    """Own the pretrained CNN and trainable head on a background thread."""

    def __init__(self) -> None:
        """Set up communication channels without launching TensorFlow yet."""
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.shutdown = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch a daemon to load weights and process camera frames."""
        if self.thread is not None:
            return

        self.thread = threading.Thread(
            target=self._run,
            name="TensorFlowVisionWorker",
            daemon=True,
        )
        self.thread.start()

    def submit(self, rgb_frame: np.ndarray) -> bool:
        """Offer one copied frame and skip it if another already waits."""
        try:
            self.frames.put_nowait(rgb_frame.copy())
            return True
        except queue.Full:
            return False

    def inspect_layer(self, name: str) -> None:
        """Request a new actual layer output for the next forward pass."""
        self.commands.put_nowait(("inspect", name))

    def capture_training_example(self, rgb_frame: np.ndarray, label: int) -> None:
        """Queue a camera image with a user-provided category A or B."""
        self.commands.put_nowait(("capture", (rgb_frame.copy(), label)))

    def train_head(self, epochs: int) -> None:
        """Queue a request to train the small dense classification head."""
        self.commands.put_nowait(("train", epochs))

    def reset_head(self) -> None:
        """Clear all examples and reinitialize trainable head parameters."""
        self.commands.put_nowait(("reset", None))

    def stop(self) -> None:
        """Signal the worker to finish after the current operation."""
        self.shutdown.set()

    def _run(self) -> None:
        """Load models and process all TensorFlow jobs on one worker thread."""
        try:
            self.events.put(("status", "Loading pretrained vision model ..."))
            model = VisionModel()
            trainer = CameraHeadTrainer(model.tf, model.encoder.output_shape[-1])
            self.events.put(
                ("ready", (model.describe_layers(), model.overview_layers))
            )
            self.events.put(("training", trainer.state()))

        except Exception:
            self.events.put(("fatal", traceback.format_exc()))
            return

        while not self.shutdown.is_set():
            try:
                # Explicit actions (layer changes, capture, train) have
                # priority over continuously incoming video frames.
                command, payload = self.commands.get_nowait()

                if command == "inspect":
                    model.set_inspected_layer(payload)

                elif command == "capture":
                    rgb_frame, label = payload
                    embedding = model.encode(rgb_frame)
                    state = trainer.add_example(embedding, label)
                    self.events.put(("training", state))
                    self.events.put(("status", "Labeled example captured in RAM"))

                elif command == "train":
                    self.events.put(("status", "Training custom classification head ..."))

                    for state in trainer.train(payload):
                        self.events.put(("training", state))

                    self.events.put(("status", "Head training complete"))

                elif command == "reset":
                    self.events.put(("training", trainer.reset()))
                    self.events.put(("status", "Custom training head reset"))

                continue

            except queue.Empty:
                pass

            except Exception as error:
                self.events.put(("training_error", str(error)))
                continue

            try:
                rgb_frame = self.frames.get(timeout=0.15)
            except queue.Empty:
                continue

            try:
                result = model.analyze(rgb_frame)
                result.custom_scores = trainer.predict(result.embedding)
                self.events.put(("result", result))

            except Exception:
                self.events.put(("error", traceback.format_exc()))
