"""Tkinter animation of a REAL, live TensorFlow 1-4-1 neural network."""
import tkinter as tk
from tkinter import ttk

from network import NetworkTrainer


BG = "#0b1220"
PANEL = "#151f31"
CANVAS = "#101a2c"
WHITE = "#eaf2ff"
MUTED = "#a9bad2"
TEAL = "#3dd6b4"
ORANGE = "#ffab70"
BLUE = "#5eacf7"
PURPLE = "#ba95fc"


class NetworkVisualizer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Live TensorFlow Neural Network | 1 → 4 → 1")
        self.root.geometry("1230x790")
        self.root.minsize(1050, 730)
        self.root.configure(bg=BG)

        self.trainer = NetworkTrainer(max_epochs=500)
        self.running = False
        self.phase = 0
        self.timer = None
        self.sample_x = tk.IntVar(value=3)
        self.cycle_ms = tk.IntVar(value=240)

        self._build_ui()
        self.draw()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        tk.Label(
            self.root, text="WATCH THE ENTIRE NETWORK LEARN", bg=BG,
            fg=WHITE, font=("Segoe UI", 21, "bold")
        ).pack(anchor="w", padx=26, pady=(20, 3))
        tk.Label(
            self.root,
            text="Actual TensorFlow weights, biases and activations — refreshed while the network trains",
            bg=BG, fg=MUTED, font=("Segoe UI", 11)
        ).pack(anchor="w", padx=28, pady=(0, 16))

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=22, pady=(0, 16))
        graph_card = tk.Frame(main, bg=PANEL, padx=9, pady=9)
        graph_card.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            graph_card, bg=CANVAS, highlightthickness=0, width=850, height=620
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw())

        side = tk.Frame(main, bg=PANEL, padx=18, pady=17, width=305)
        side.pack(side="right", fill="y", padx=(13, 0))
        side.pack_propagate(False)

        self.status = self._info(side, "WHAT'S HAPPENING", "Ready — press Start", PURPLE)
        self.epoch = self._info(side, "EPOCH", "0 / 500", WHITE)
        self.loss = self._info(side, "TRAINING LOSS (MSE)", "—", ORANGE)
        self.prediction = self._info(side, "CURRENT INPUT / OUTPUT", "x = 3 → …", TEAL)
        self.test_prediction = self._info(side, "TEST: x = 10", "Prediction: …  |  target: 21", BLUE)

        tk.Label(side, text="CHOOSE INPUT TO WATCH", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 3))
        tk.Scale(
            side, from_=1, to=10, orient="horizontal", variable=self.sample_x,
            command=lambda _value: self.draw(), bg=PANEL, fg=WHITE,
            troughcolor=BG, highlightthickness=0, activebackground=TEAL,
            length=252
        ).pack(anchor="w", fill="x", pady=(0, 12))
        tk.Label(side, text="SPEED (LEFT = FASTER)", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Scale(
            side, from_=80, to=1000, orient="horizontal", variable=self.cycle_ms,
            bg=PANEL, fg=WHITE, troughcolor=BG,
            highlightthickness=0, activebackground=TEAL, length=252,
            showvalue=False
        ).pack(anchor="w", fill="x", pady=(0, 10))

        buttons = tk.Frame(side, bg=PANEL)
        buttons.pack(fill="x", pady=(3, 6))
        self.start_button = self._button(buttons, "▶ Start", self.toggle, TEAL)
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self._button(buttons, "Step", self.step, BLUE).pack(
            side="left", expand=True, fill="x", padx=(0, 5))
        self._button(buttons, "Reset", self.reset, ORANGE).pack(
            side="left", expand=True, fill="x")

        tk.Label(
            side, text="TEAL = positive weight\nORANGE = negative weight\n"
                       "Thicker line = stronger connection\n"
                       "Moving dots = forward signal",
            bg=PANEL, fg=MUTED, justify="left", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(15, 0))

        tk.Label(
            self.root,
            text="This is a 1 → 4 → 1 linear network. The original Hello World had only one neuron; "
                 "four hidden neurons make the internal math visible.",
            bg=BG, fg=MUTED, font=("Segoe UI", 10)
        ).pack(anchor="w", padx=27, pady=(0, 14))

    def _info(self, parent, title, value, color):
        tk.Label(parent, text=title, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        result = tk.Label(parent, text=value, bg=PANEL, fg=color,
                          font=("Segoe UI", 12, "bold"), justify="left",
                          wraplength=260)
        result.pack(anchor="w", pady=(0, 21))
        return result

    def _button(self, parent, caption, callback, color):
        return tk.Button(
            parent, text=caption, command=callback, bg=color, fg="#0a1220",
            activebackground=WHITE, relief="flat", cursor="hand2",
            font=("Segoe UI", 10, "bold"), pady=10, padx=5
        )

    def toggle(self):
        if self.running:
            self._pause()
        elif self.trainer.epoch < self.trainer.max_epochs:
            self.running = True
            self.start_button.configure(text="Ⅱ Pause")
            self._schedule()

    def _pause(self):
        self.running = False
        self.start_button.configure(text="▶ Start")
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def reset(self):
        self._pause()
        self.trainer.reset()
        self.phase = 0
        self.draw()

    def step(self):
        """Advance exactly one animated phase; update weights on phase 3."""
        self._pause()
        self.advance()

    def _schedule(self):
        if self.running:
            self.timer = self.root.after(max(20, int(self.cycle_ms.get()) // 4), self._tick)

    def _tick(self):
        self.timer = None
        if self.running:
            self.advance()
            if self.trainer.epoch >= self.trainer.max_epochs:
                self._pause()
            else:
                self._schedule()

    def advance(self):
        if self.trainer.epoch >= self.trainer.max_epochs:
            self.draw()
            return
        self.phase = (self.phase + 1) % 4
        if self.phase == 0:
            # TensorFlow adjusts ALL weights and biases together from 5 examples.
            self.trainer.train_epoch()
        self.draw()

    def close(self):
        self._pause()
        self.root.destroy()

    def draw(self):
        if not hasattr(self, "canvas"):
            return
        s = self.trainer.snapshot(float(self.sample_x.get()))
        self.epoch.configure(text=f"{s.epoch} / {self.trainer.max_epochs}")
        self.loss.configure(text=f"{s.loss:.7f}")
        self.prediction.configure(
            text=f"x = {s.x:.0f}  →  {s.prediction:.3f}\nCorrect answer: {s.target:.0f}"
        )
        self.test_prediction.configure(text=f"{s.test_prediction:.3f}  (target: 21)")
        stages = [
            "INPUT: read x",
            "FORWARD PASS: input → hidden",
            "FORWARD PASS: hidden → output",
            "TRAIN: update all weights + biases",
        ]
        if s.epoch == self.trainer.max_epochs:
            message = "Training complete!"
        elif not self.running and self.phase == 0 and s.epoch == 0:
            message = "Ready — press Start"
        else:
            message = stages[self.phase]
        self.status.configure(text=message)

        c = self.canvas
        c.delete("all")
        width = max(780, c.winfo_width())
        height = max(570, c.winfo_height())
        center_y = height * 0.51
        left_x, middle_x, right_x = width * .12, width * .47, width * .83
        hidden_ys = [height * frac for frac in (.18, .39, .62, .83)]

        def text(x, y, value, size=11, color=WHITE, bold=False, anchor="center"):
            font = ("Segoe UI", size, "bold" if bold else "normal")
            c.create_text(x, y, text=value, fill=color, font=font, anchor=anchor)

        text(left_x, 28, "INPUT LAYER", 13, BLUE, True)
        text(middle_x, 28, "HIDDEN LAYER (4 NEURONS)", 13, BLUE, True)
        text(right_x, 28, "OUTPUT LAYER", 13, BLUE, True)

        def connection(x1, y1, x2, y2, weight, fraction, traveling):
            color = TEAL if weight >= 0 else ORANGE
            thick = 1.1 + min(5, abs(weight) * 2.5)
            c.create_line(x1, y1, x2, y2, fill=color, width=thick, arrow=tk.LAST)
            tx = x1 + (x2-x1)*fraction
            ty = y1 + (y2-y1)*fraction
            text(tx, ty-15, f"w={weight:+.2f}", 9, WHITE)
            if traveling:
                # The signal dot's route matches this exact connection.
                sx = x1 + (x2-x1) * .72
                sy = y1 + (y2-y1) * .72
                c.create_oval(sx-7, sy-7, sx+7, sy+7, fill="#fff6b3", outline="")

        for i, hy in enumerate(hidden_ys):
            connection(left_x+28, center_y, middle_x-31, hy,
                       s.hidden_weights[i], .48, self.phase == 1)
            connection(middle_x+31, hy, right_x-31, center_y,
                       s.output_weights[i], .55, self.phase == 2)

        def neuron(x, y, label, value, fill, active, bias=None):
            outline = "#fff4bd" if active else WHITE
            radius = 30 if active else 27
            c.create_oval(x-radius, y-radius, x+radius, y+radius,
                          fill=fill, outline=outline, width=4 if active else 2)
            text(x, y-6, label, 12, "#071322", True)
            text(x, y+11, f"{value:.2f}", 10, "#071322", True)
            if bias is not None:
                text(x, y+46, f"bias={bias:+.2f}", 9, MUTED)

        neuron(left_x, center_y, "x", s.x, BLUE, self.phase == 0)
        for i, hy in enumerate(hidden_ys):
            neuron(middle_x, hy, f"H{i+1}", s.hidden[i], TEAL,
                   self.phase == 1, s.hidden_biases[i])
        neuron(right_x, center_y, "ŷ", s.prediction, PURPLE,
               self.phase >= 2, s.output_bias)

        text(width/2, height-52, "Each connection has a REAL trainable weight.", 12, WHITE, True)
        text(width/2, height-28,
             "At the update step, TensorFlow calculates gradients using all 5 examples.",
             10, MUTED)
