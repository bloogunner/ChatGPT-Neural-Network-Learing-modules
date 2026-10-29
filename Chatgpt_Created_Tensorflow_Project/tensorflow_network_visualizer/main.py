"""Launch the TensorFlow neural network visualizer."""
import tkinter as tk

from visualizer import NetworkVisualizer


def main():
    root = tk.Tk()
    NetworkVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
