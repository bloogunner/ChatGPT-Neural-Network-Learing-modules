"""Application entry point: run ONLY this file to start the explorer."""

from __future__ import annotations

import tkinter as tk

from visualizer import VisionVisualizer


def main() -> None:
    """Create the main Tkinter event loop and start the visualizer.

    Tkinter owns the main thread. VisionVisualizer creates a separate
    background thread for TensorFlow so camera preview stays responsive.
    """
    root = tk.Tk()
    VisionVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
