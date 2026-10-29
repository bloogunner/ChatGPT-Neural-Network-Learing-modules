"""Tkinter dashboard for a live webcam and inspected CNN activations.

The flow drawing is a schematic of real MobileNetV2 layer STAGES, not a
literal map of millions of tensor elements and convolution connections.
Circle brightness comes from measured activation values. Moving dots
illustrate the order of a forward pass; they do not measure execution time.
"""

from __future__ import annotations

import queue
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from camera import Webcam
from settings import (
    ACCENT,
    BACKGROUND,
    CAMERA_DELAY_MS,
    INFERENCE_INTERVAL_SECONDS,
    MUTED,
    PANEL,
    SUCCESS,
    TEXT,
)
from worker import VisionWorker


class VisionVisualizer:
    """Combine a camera preview, a network diagram, and a layer inspector."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the whole interface before opening the camera hardware."""
        self.root = root
        self.root.title("Live TensorFlow Vision Network Explorer")
        self.root.geometry("1510x940")
        self.root.minsize(1130, 760)
        self.root.configure(bg=BACKGROUND)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=3, minsize=440)
        self.root.grid_rowconfigure(3, weight=1, minsize=180)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.webcam = Webcam(index=0)
        self.worker = VisionWorker()
        self.worker_ready = False
        self.camera_running = False
        self.ai_running = True
        self.request_pending = False
        self.last_request = 0.0
        self.last_frame: np.ndarray | None = None
        self.camera_photo = None
        self.feature_photos: list[ImageTk.PhotoImage] = []
        self.overview_names: list[str] = []
        self.layer_lookup: dict[str, object] = {}
        self.selected_layer = ""
        self.latest_result = None
        self.training_state = None
        self.flow_dot = None
        self.flow_position = 0.0
        self.flow_points: list[float] = []
        self.closed = False

        self.camera_number = tk.IntVar(value=0)
        self.update_interval = tk.DoubleVar(value=INFERENCE_INTERVAL_SECONDS)
        self.status_var = tk.StringVar(value="Initializing interface ...")
        self.camera_var = tk.StringVar(value="Camera is stopped")
        self.active_layer_var = tk.StringVar(value="Select a layer to inspect")
        self.top_prediction_var = tk.StringVar(value="No image analyzed yet")
        self.class_a_name = tk.StringVar(value="mug")
        self.class_b_name = tk.StringVar(value="phone")
        self.training_status_var = tk.StringVar(value="No examples captured")
        self.custom_prediction_var = tk.StringVar(
            value="Capture both classes, then train to enable predictions."
        )

        self._configure_style()
        self._build_layout()

        # TensorFlow loads on a worker thread, so the UI stays responsive.
        self.worker.start()
        self.root.after(100, self._poll_worker)
        self.root.after(CAMERA_DELAY_MS, self._update_camera)
        self.root.after(50, self._animate_flow)

    def _configure_style(self) -> None:
        """Choose consistent styling for buttons, lists, and scrollbar."""
        theme = ttk.Style(self.root)
        theme.theme_use("clam")
        theme.configure("Vision.TNotebook", background=PANEL, borderwidth=0)
        theme.configure(
            "Vision.TNotebook.Tab",
            background="#263448",
            foreground=TEXT,
            font=("Segoe UI", 10, "bold"),
            padding=(12, 7),
        )
        theme.map(
            "Vision.TNotebook.Tab",
            background=[("selected", "#315275")],
        )
        theme.configure(
            "Vision.Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=23,
            borderwidth=0,
        )
        theme.map(
            "Vision.Treeview",
            background=[("selected", "#24547b")],
            foreground=[("selected", "white")],
        )
        theme.configure(
            "Vision.Treeview.Heading",
            background="#27364c",
            foreground=TEXT,
            font=("Segoe UI", 10, "bold"),
        )

    def _make_label(self, parent, text: str, size: int = 10, **kwargs):
        """Create a consistently styled Tkinter text label."""
        return tk.Label(
            parent,
            text=text,
            font=("Segoe UI", size),
            bg=kwargs.pop("bg", PANEL),
            fg=kwargs.pop("fg", TEXT),
            **kwargs,
        )

    def _build_layout(self) -> None:
        """Arrange camera, predictions, network drawing, and layer explorer."""
        header = tk.Frame(self.root, bg=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(13, 7))

        self._make_label(
            header,
            "LIVE VISION • INSIDE THE NETWORK",
            17,
            bg=BACKGROUND,
            fg=ACCENT,
        ).pack(side="left")

        self._make_label(
            header,
            "MobileNetV2 inference + optional trainable 2-class head",
            10,
            bg=BACKGROUND,
            fg=MUTED,
        ).pack(side="right")

        controls = tk.Frame(self.root, bg=BACKGROUND)
        controls.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))

        self.camera_button = tk.Button(
            controls,
            text="Start camera",
            command=self._toggle_camera,
            width=14,
        )
        self.camera_button.pack(side="left", padx=(0, 9))

        self.ai_button = tk.Button(
            controls,
            text="Pause analysis",
            command=self._toggle_ai,
            width=15,
        )
        self.ai_button.pack(side="left", padx=(0, 9))

        tk.Button(
            controls,
            text="Analyze one frame",
            command=self._analyze_one_frame,
            width=19,
        ).pack(side="left", padx=(0, 14))

        self._make_label(
            controls,
            "Camera index",
            bg=BACKGROUND,
            fg=MUTED,
        ).pack(side="left")
        tk.Spinbox(
            controls,
            from_=0,
            to=9,
            textvariable=self.camera_number,
            width=3,
        ).pack(side="left", padx=(6, 18))

        self._make_label(
            controls,
            "Analyze every (sec)",
            bg=BACKGROUND,
            fg=MUTED,
        ).pack(side="left")
        tk.Scale(
            controls,
            from_=0.4,
            to=3.0,
            resolution=0.2,
            orient="horizontal",
            variable=self.update_interval,
            bg=BACKGROUND,
            fg=TEXT,
            troughcolor=PANEL,
            highlightthickness=0,
            length=175,
        ).pack(side="left", padx=8)

        main = tk.Frame(self.root, bg=BACKGROUND)
        main.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 10))
        main.grid_columnconfigure(0, weight=2)
        main.grid_columnconfigure(1, weight=3)
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_camera_panel(left)

        notebook = ttk.Notebook(right, style="Vision.TNotebook")
        notebook.pack(fill="both", expand=True)
        network_tab = tk.Frame(notebook, bg=PANEL, padx=5, pady=8)
        training_tab = tk.Frame(notebook, bg=PANEL, padx=5, pady=8)
        notebook.add(network_tab, text="Explore CNN layers")
        notebook.add(training_tab, text="Train camera classifier")
        self._build_network_panel(network_tab)
        self._build_training_panel(training_tab)
        self._build_layer_table()

        footer = tk.Frame(self.root, bg=BACKGROUND)
        footer.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 9))

        self._make_label(
            footer,
            "Dots are schematic signal flow • Channel brightness and heatmaps are measured values • Camera remains local",
            9,
            bg=BACKGROUND,
            fg=MUTED,
        ).pack(anchor="w")
        self._make_label(
            footer,
            "",
            9,
            bg=BACKGROUND,
            fg=SUCCESS,
            textvariable=self.status_var,
            anchor="w",
        ).pack(anchor="w")

    def _build_camera_panel(self, parent) -> None:
        """Create camera preview and classification score rows."""
        self._make_label(
            parent, "Camera input", 13, fg=ACCENT
        ).pack(anchor="w")

        self.camera_display = tk.Label(
            parent,
            text="Press 'Start camera' to grant camera access.",
            bg="#070d17",
            fg=MUTED,
            font=("Segoe UI", 11),
            width=55,
            height=11,
        )
        self.camera_display.pack(fill="both", expand=True, pady=(10, 7))

        self._make_label(
            parent,
            "",
            10,
            textvariable=self.camera_var,
            fg=MUTED,
        ).pack(anchor="w")

        self._make_label(parent, "Top ImageNet guesses", 12, fg=ACCENT).pack(
            anchor="w", pady=(12, 6)
        )
        self._make_label(
            parent, "", 10, textvariable=self.top_prediction_var, anchor="w"
        ).pack(fill="x", pady=(0, 8))

        self.prediction_rows = []

        for index in range(5):
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", pady=3)

            name = self._make_label(
                row, f"{index + 1}. —", 10, anchor="w", width=28
            )
            name.pack(side="left")

            percent = self._make_label(row, "0%", 10, fg=MUTED, width=7)
            percent.pack(side="right")

            bar = tk.Canvas(
                row,
                height=12,
                width=135,
                bg="#263247",
                highlightthickness=0,
            )
            bar.pack(side="right", fill="x", expand=True, padx=8)
            fill = bar.create_rectangle(0, 0, 0, 12, fill=ACCENT, width=0)
            self.prediction_rows.append((name, percent, bar, fill))

    def _build_network_panel(self, parent) -> None:
        """Create the stage-flow drawing and selected-layer feature maps."""
        self._make_label(
            parent, "Network activity (real intermediate tensors)", 13, fg=ACCENT
        ).pack(anchor="w")

        self.network_canvas = tk.Canvas(
            parent,
            bg="#101b2b",
            height=200,
            highlightthickness=0,
        )
        self.network_canvas.pack(fill="x", pady=(10, 0))

        scroll = tk.Scrollbar(
            parent,
            orient="horizontal",
            command=self.network_canvas.xview,
        )
        scroll.pack(fill="x")
        self.network_canvas.configure(xscrollcommand=scroll.set)
        self.network_canvas.bind("<Button-1>", self._click_stage)

        self._make_label(
            parent,
            "Flow is illustrative, not real-time layer execution timing. Click a stage or any layer below.",
            9,
            fg=MUTED,
        ).pack(anchor="w", pady=(4, 12))

        self._make_label(
            parent,
            "",
            12,
            textvariable=self.active_layer_var,
            fg=ACCENT,
            anchor="w",
        ).pack(fill="x", pady=(0, 9))

        self.heatmap_grid = tk.Frame(parent, bg=PANEL)
        self.heatmap_grid.pack(fill="both", expand=True)
        self.heatmap_labels = []

        for row in range(2):
            self.heatmap_grid.grid_rowconfigure(row, weight=1)

            for column in range(3):
                self.heatmap_grid.grid_columnconfigure(column, weight=1)
                card = tk.Frame(self.heatmap_grid, bg="#25334a", padx=5, pady=5)
                card.grid(
                    row=row,
                    column=column,
                    padx=4,
                    pady=4,
                    sticky="nsew",
                )
                title = self._make_label(
                    card, "Channel —", 9, bg="#25334a", fg=MUTED
                )
                title.pack()
                image = tk.Label(
                    card,
                    text="Waiting for a frame",
                    bg="#080f1b",
                    fg=MUTED,
                    width=21,
                    height=6,
                )
                image.pack(fill="both", expand=True, pady=(3, 0))
                self.heatmap_labels.append((title, image))

    def _build_training_panel(self, parent) -> None:
        """Display an optional small head that actually learns from camera.

        Users provide class labels by taking snapshots. The pretrained CNN
        remains frozen; only the 1280 -> 6 -> 2 classifier is trained.
        """
        self._make_label(
            parent, "Train your own two-object recognizer", 15, fg=ACCENT
        ).pack(anchor="w", pady=(2, 8))
        self._make_label(
            parent,
            "Capture several angles of TWO objects. The frozen vision CNN "
            "extracts 1280 features; the small head learns your labels.",
            10,
            fg=MUTED,
            wraplength=730,
            justify="left",
        ).pack(anchor="w", pady=(0, 13))

        names = tk.Frame(parent, bg=PANEL)
        names.pack(fill="x", pady=(0, 10))

        self._make_label(names, "Class A:", 10).pack(side="left")
        tk.Entry(
            names, textvariable=self.class_a_name, width=18
        ).pack(side="left", padx=(6, 22))
        self._make_label(names, "Class B:", 10).pack(side="left")
        tk.Entry(
            names, textvariable=self.class_b_name, width=18
        ).pack(side="left", padx=6)

        buttons = tk.Frame(parent, bg=PANEL)
        buttons.pack(fill="x", pady=7)

        tk.Button(
            buttons,
            text="Capture A",
            command=lambda: self._capture_training(0),
        ).pack(side="left", padx=(0, 9))
        tk.Button(
            buttons,
            text="Capture B",
            command=lambda: self._capture_training(1),
        ).pack(side="left", padx=(0, 9))
        tk.Button(
            buttons,
            text="Train 1 epoch",
            command=lambda: self._train_head(1),
        ).pack(side="left", padx=(0, 9))
        tk.Button(
            buttons,
            text="Train 10 epochs",
            command=lambda: self._train_head(10),
        ).pack(side="left", padx=(0, 9))
        tk.Button(
            buttons,
            text="Reset",
            command=self._reset_training,
        ).pack(side="left")

        self._make_label(
            parent, "", textvariable=self.training_status_var, fg=SUCCESS
        ).pack(anchor="w", pady=(7, 4))
        self._make_label(
            parent, "", textvariable=self.custom_prediction_var, fg=TEXT,
            wraplength=740, justify="left",
        ).pack(anchor="w", pady=(0, 10))

        self._make_label(
            parent,
            "Actual trainable head • every 6-to-2 connection is drawn "
            "from its real weight",
            11,
            fg=ACCENT,
        ).pack(anchor="w")

        self.head_canvas = tk.Canvas(
            parent,
            bg="#101b2b",
            height=280,
            highlightthickness=0,
        )
        self.head_canvas.pack(fill="both", expand=True, pady=(9, 6))

        self._make_label(
            parent,
            "Blue connections are positive; red connections are negative. "
            "Thickness = absolute weight. The 1280 input links are omitted.",
            9,
            fg=MUTED,
            wraplength=740,
            justify="left",
        ).pack(anchor="w", pady=(3, 6))
        self._draw_head_weights()

    def _capture_training(self, label: int) -> None:
        """Label the last real camera frame A or B and queue feature capture."""
        if self.last_frame is None or not self.worker_ready:
            self.status_var.set("Start the camera and load the model first")
            return

        self.worker.capture_training_example(self.last_frame, label)
        self.status_var.set(
            f"Capturing example for class {'A' if label == 0 else 'B'} ..."
        )

    def _train_head(self, epochs: int) -> None:
        """Request gradient descent on real, user-labeled image embeddings."""
        if not self.worker_ready:
            return

        self.worker.train_head(epochs)
        self.status_var.set(f"Requested {epochs} training epoch(s)")

    def _reset_training(self) -> None:
        """Clear local training data and all trainable-head updates."""
        if self.worker_ready:
            self.worker.reset_head()

    def _show_training_state(self, state) -> None:
        """Update epoch, sample counts, loss, and actual weight connections."""
        self.training_state = state
        a, b = state.class_counts
        loss = (
            f"{state.latest_loss:.4f}"
            if state.latest_loss is not None
            else "not trained"
        )
        self.training_status_var.set(
            f"Examples: A={a}, B={b} | Training epochs: {state.epochs} "
            f"| Loss: {loss}"
        )

        if state.epochs == 0:
            self.custom_prediction_var.set(
                "Capture at least 3 examples of EACH class, then train."
            )

        self._draw_head_weights()

    def _draw_head_weights(self) -> None:
        """Draw true learned weights connecting six hidden to two outputs.

        The first dense layer receives 1,280 features and therefore has
        7,680 input weight connections; only its block is drawn. Every one
        of the 12 connections in the final 6 -> 2 layer is represented.
        """
        if not hasattr(self, "head_canvas"):
            return

        canvas = self.head_canvas
        canvas.delete("all")
        canvas.configure(scrollregion=(0, 0, 760, 275))
        canvas.create_rectangle(
            20, 90, 164, 175, fill="#1e344d", outline="#557998"
        )
        canvas.create_text(
            92, 117, text="Frozen CNN", fill=TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        canvas.create_text(
            92, 143, text="1280 features", fill=MUTED,
            font=("Segoe UI", 10),
        )
        canvas.create_line(166, 132, 320, 132, fill=ACCENT, arrow=tk.LAST)
        canvas.create_text(
            244, 111, text="trainable weights", fill=MUTED,
            font=("Segoe UI", 9),
        )

        hidden_x, output_x = 365, 630
        hidden_ys = [38, 76, 114, 152, 190, 228]
        output_ys = [91, 177]
        state = self.training_state
        maximum = float(np.max(np.abs(state.weights))) if state is not None else 1.0
        maximum = max(maximum, 1e-6)

        for i, y1 in enumerate(hidden_ys):
            for j, y2 in enumerate(output_ys):
                weight = float(state.weights[i, j]) if state is not None else 0.0
                intensity = abs(weight) / maximum
                width = 1 + 5 * intensity
                color = "#5bb9ff" if weight >= 0 else "#fb7285"
                canvas.create_line(
                    hidden_x + 14, y1, output_x - 17, y2,
                    fill=color, width=width,
                )

        for i, y in enumerate(hidden_ys):
            canvas.create_oval(
                hidden_x - 14, y - 14, hidden_x + 14, y + 14,
                fill="#2a607e", outline=ACCENT,
            )
            canvas.create_text(
                hidden_x, y, text=str(i + 1), fill=TEXT,
                font=("Segoe UI", 9, "bold"),
            )

        for j, y in enumerate(output_ys):
            canvas.create_oval(
                output_x - 17, y - 17, output_x + 17, y + 17,
                fill="#326b59", outline=SUCCESS,
            )
            canvas.create_text(
                output_x, y, text="AB"[j], fill=TEXT,
                font=("Segoe UI", 10, "bold"),
            )
            bias = float(state.biases[j]) if state is not None else 0.0
            canvas.create_text(
                output_x + 67, y, text=f"bias {bias:+.3f}",
                fill=MUTED, font=("Consolas", 9),
            )

        canvas.create_text(
            hidden_x, 257, text="6 hidden neurons", fill=MUTED,
            font=("Segoe UI", 10),
        )
        canvas.create_text(
            output_x, 257, text="2 classes", fill=MUTED,
            font=("Segoe UI", 10),
        )

    def _build_layer_table(self) -> None:
        """Show all actual layers and allow any layer to be inspected."""
        panel = tk.Frame(self.root, bg=PANEL, padx=12, pady=9)
        panel.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 8))

        self._make_label(
            panel,
            "FULL MODEL LAYER EXPLORER — click any actual Keras layer to view its output",
            11,
            fg=ACCENT,
        ).pack(anchor="w")

        frame = tk.Frame(panel, bg=PANEL)
        frame.pack(fill="both", expand=True, pady=(5, 0))

        self.layer_tree = ttk.Treeview(
            frame,
            columns=("kind", "shape"),
            show="tree headings",
            height=7,
            style="Vision.Treeview",
        )
        self.layer_tree.heading("#0", text="Layer name")
        self.layer_tree.heading("kind", text="Layer type")
        self.layer_tree.heading("shape", text="Output tensor shape")
        self.layer_tree.column("#0", width=390, minwidth=230)
        self.layer_tree.column("kind", width=225, minwidth=120)
        self.layer_tree.column("shape", width=400, minwidth=165)
        self.layer_tree.pack(side="left", fill="both", expand=True)
        self.layer_tree.bind("<<TreeviewSelect>>", self._on_layer_selected)

        scrollbar = ttk.Scrollbar(
            frame, orient="vertical", command=self.layer_tree.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.layer_tree.configure(yscrollcommand=scrollbar.set)

    def _toggle_camera(self) -> None:
        """Open or release the user-selected physical webcam."""
        if self.camera_running:
            self.webcam.stop()
            self.camera_running = False
            self.last_frame = None
            self.camera_display.configure(image="", text="Camera stopped")
            self.camera_photo = None
            self.camera_button.configure(text="Start camera")
            self.camera_var.set("Camera stopped — AI receives no more frames")
            self.status_var.set("Camera access released")
            return

        try:
            self.webcam.index = self.camera_number.get()
            self.webcam.start()
        except (RuntimeError, tk.TclError) as error:
            messagebox.showerror("Webcam unavailable", str(error))
            self.status_var.set("Camera could not be opened")
            return

        self.camera_running = True
        self.camera_button.configure(text="Stop camera")
        self.camera_var.set(f"Camera {self.webcam.index}: live RGB preview")
        self.status_var.set("Camera opened on this computer")

    def _toggle_ai(self) -> None:
        """Pause or resume automatic classification without closing camera."""
        self.ai_running = not self.ai_running
        self.ai_button.configure(
            text="Pause analysis" if self.ai_running else "Resume analysis"
        )
        self.status_var.set(
            "Automatic classification enabled"
            if self.ai_running
            else "Automatic classification paused"
        )

    def _analyze_one_frame(self) -> None:
        """Queue the most recent camera frame regardless of auto-pause."""
        if self.last_frame is None:
            self.status_var.set("Start the camera before analyzing a frame")
            return

        self._submit_frame()

    def _submit_frame(self) -> None:
        """Send one frame only when TensorFlow is ready and not busy."""
        if self.last_frame is None or not self.worker_ready:
            return

        if self.request_pending:
            return

        if self.worker.submit(self.last_frame):
            self.request_pending = True
            self.last_request = time.monotonic()
            self.status_var.set("TensorFlow forward pass in progress ...")

    def _update_camera(self) -> None:
        """Refresh webcam video without waiting on neural-network inference.

        Tkinter schedules this callback periodically. Every video frame is
        displayed, but only occasional frames are sent for classification.
        """
        if self.closed:
            return

        if self.camera_running:
            frame = self.webcam.read()

            if frame is not None:
                self.last_frame = frame
                height, width = frame.shape[:2]
                maximum_width, maximum_height = 555, 290
                scale = min(maximum_width / width, maximum_height / height)
                size = (max(1, int(width * scale)), max(1, int(height * scale)))
                photo = ImageTk.PhotoImage(
                    Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
                )
                self.camera_photo = photo
                self.camera_display.configure(image=photo, text="")

                elapsed = time.monotonic() - self.last_request

                if (
                    self.ai_running
                    and self.worker_ready
                    and not self.request_pending
                    and elapsed >= self.update_interval.get()
                ):
                    self._submit_frame()

        self.root.after(CAMERA_DELAY_MS, self._update_camera)

    def _poll_worker(self) -> None:
        """Receive worker messages on Tkinter's thread and update widgets."""
        if self.closed:
            return

        try:
            while True:
                kind, value = self.worker.events.get_nowait()

                if kind == "ready":
                    descriptions, self.overview_names = value
                    self.worker_ready = True
                    self._populate_layers(descriptions)
                    self.status_var.set(
                        f"Ready: {len(descriptions)} real Keras layers loaded"
                    )
                    self._draw_network()

                elif kind == "result":
                    self.request_pending = False
                    self.latest_result = value
                    self._show_result(value)

                elif kind == "training":
                    self._show_training_state(value)

                elif kind == "training_error":
                    self.status_var.set(str(value))

                elif kind == "status":
                    self.status_var.set(str(value))

                elif kind == "error":
                    self.request_pending = False
                    self.status_var.set("Inference error — see terminal traceback")
                    print(value)

                elif kind == "fatal":
                    self.request_pending = False
                    self.status_var.set("Could not initialize TensorFlow")
                    print(value)
                    messagebox.showerror(
                        "TensorFlow failed to load",
                        "Check the terminal for the full traceback. "
                        "Pretrained weights need internet access on the first run.\n\n"
                        + str(value)[-1200:],
                    )

        except queue.Empty:
            pass

        self.root.after(100, self._poll_worker)

    def _populate_layers(self, descriptions) -> None:
        """Fill the complete model layer list, including residual Add layers."""
        self.layer_lookup.clear()

        for item in descriptions:
            self.layer_tree.insert(
                "",
                "end",
                iid=item.name,
                text=item.name,
                values=(item.kind, item.output_shape),
            )
            self.layer_lookup[item.name] = item

        if self.overview_names:
            first = self.overview_names[0]
            self.selected_layer = first
            self.layer_tree.selection_set(first)
            self.layer_tree.see(first)

    def _on_layer_selected(self, _event=None) -> None:
        """Request actual tensors from whichever Keras layer was selected."""
        selected = self.layer_tree.selection()

        if not selected:
            return

        name = selected[0]

        if name == self.selected_layer:
            return

        self.selected_layer = name
        self.worker.inspect_layer(name)
        self.active_layer_var.set(f"Inspecting: {name} (next analyzed frame)")
        self._draw_network()

        # If auto-analysis is paused, a manual snapshot can update the view.
        if not self.ai_running and self.last_frame is not None:
            self._submit_frame()

    def _click_stage(self, event) -> None:
        """Select a layer by clicking its stage in the network drawing."""
        if not self.overview_names:
            return

        x = self.network_canvas.canvasx(event.x)
        positions = [170 + index * 145 for index in range(len(self.overview_names))]
        closest = min(range(len(positions)), key=lambda index: abs(x - positions[index]))
        name = self.overview_names[closest]
        self.layer_tree.selection_set(name)
        self.layer_tree.see(name)
        self._on_layer_selected()

    def _draw_network(self) -> None:
        """Draw sampled feature channels and conceptual forward connections.

        The full CNN has convolutions, 2-D spatial positions, and residual
        connections. Lines here are a legible schematic, NOT real neuron
        edges or learned weights. The circles, however, are colored from
        actual sampled mean absolute activations in each stage.
        """
        canvas = self.network_canvas
        canvas.delete("all")

        if not self.overview_names:
            canvas.create_text(
                360,
                110,
                text="Loading network architecture ...",
                fill=MUTED,
                font=("Segoe UI", 12),
            )
            return

        positions = [170 + index * 145 for index in range(len(self.overview_names))]
        canvas.configure(scrollregion=(0, 0, positions[-1] + 125, 200))
        self.flow_points = positions

        for index in range(len(positions) - 1):
            x1 = positions[index] + 20
            x2 = positions[index + 1] - 20

            for channel in range(5):
                y1 = 65 + channel * 24
                canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y1,
                    fill="#31516d",
                    width=1,
                    arrow=tk.LAST,
                )

        for index, (x, name) in enumerate(zip(positions, self.overview_names)):
            is_selected = self.selected_layer == name
            canvas.create_rectangle(
                x - 62,
                25,
                x + 62,
                196,
                outline=ACCENT if is_selected else "#344d68",
                width=2 if is_selected else 1,
                fill="#1b2a40",
            )

            label = name.replace("expanded_conv_", "expanded_")
            label = label.replace("depthwise_relu", "depthwise")
            canvas.create_text(
                x,
                39,
                text=label,
                fill=TEXT,
                width=118,
                font=("Segoe UI", 8, "bold"),
            )

            activity = None

            if self.latest_result is not None:
                activity = self.latest_result.overview.get(name)

            values = activity.channel_values if activity else [0.0] * 5
            maximum = max(values, default=0.0)

            for channel in range(5):
                y = 65 + channel * 24
                value = values[channel] if channel < len(values) else 0.0
                relative = value / maximum if maximum > 1e-9 else 0.0
                color = self._activity_color(relative)
                canvas.create_oval(
                    x - 12,
                    y - 12,
                    x + 12,
                    y + 12,
                    fill=color,
                    outline="#8bdafa" if relative > 0.6 else "#465b73",
                )

            shape = activity.tensor_shape if activity else "—"
            canvas.create_text(
                x,
                185,
                text=shape[:21],
                fill=MUTED,
                width=115,
                font=("Consolas", 8),
            )

        self.flow_dot = canvas.create_oval(
            positions[0] - 5,
            10,
            positions[0] + 5,
            20,
            fill=SUCCESS,
            outline="",
        )

    def _activity_color(self, relative: float) -> str:
        """Color one stage node by its relative channel activation value."""
        amount = max(0.0, min(1.0, relative))
        red = int(30 + 65 * amount)
        green = int(60 + 150 * amount)
        blue = int(91 + 155 * amount)
        return f"#{red:02x}{green:02x}{min(255, blue):02x}"

    def _animate_flow(self) -> None:
        """Move a schematic pulse along the layer order between updates."""
        if self.closed:
            return

        if self.flow_dot is not None and self.flow_points:
            self.flow_position = (self.flow_position + 0.035) % 1.0
            start = self.flow_points[0]
            end = self.flow_points[-1]
            x = start + (end - start) * self.flow_position
            self.network_canvas.coords(self.flow_dot, x - 5, 10, x + 5, 20)

        self.root.after(50, self._animate_flow)

    def _show_result(self, result) -> None:
        """Render real CNN responses and top-five object classification."""
        if result.custom_scores is not None:
            name_a = self.class_a_name.get().strip() or "Class A"
            name_b = self.class_b_name.get().strip() or "Class B"
            probability_a, probability_b = result.custom_scores
            self.custom_prediction_var.set(
                f"Your classifier: {name_a} {probability_a * 100:.1f}% "
                f"| {name_b} {probability_b * 100:.1f}%"
            )
        else:
            self.custom_prediction_var.set(
                "Capture and train examples in the training tab to enable "
                "your own class predictions."
            )

        best_label, best_score = result.predictions[0]
        self.top_prediction_var.set(
            f"Top guess: {best_label}  ({best_score * 100:.1f}%)"
        )

        for index, (label, percent, bar, rectangle) in enumerate(
            self.prediction_rows
        ):
            if index < len(result.predictions):
                name, probability = result.predictions[index]
                label.configure(text=f"{index + 1}. {name[:29]}")
                percent.configure(text=f"{probability * 100:.1f}%")
                bar.update_idletasks()
                width = max(110, bar.winfo_width())
                bar.coords(rectangle, 0, 0, int(probability * width), 12)

        # Do not incorrectly label the output of an old in-flight selection.
        if self.selected_layer == result.inspected_layer:
            self.active_layer_var.set(
                f"Layer: {result.inspected_layer}   |   "
                f"Tensor shape: {result.inspected_shape}"
            )
            self._show_heatmaps(result.heatmaps, result.inspected_shape)
        else:
            self.active_layer_var.set(
                f"Layer: {self.selected_layer} (waiting for next frame)"
            )

        self._draw_network()
        self.flow_position = 0.0
        self.status_var.set(
            f"Inference complete • {len(result.overview)} stages sampled • "
            "Pretrained weights unchanged"
        )

    def _show_heatmaps(self, heatmaps, tensor_shape: str) -> None:
        """Paint six measured channel maps from the selected TensorFlow layer."""
        self.feature_photos.clear()
        is_spatial = len(tensor_shape.strip("()").split(",")) >= 3

        for index, (title, preview) in enumerate(self.heatmap_labels):
            if index >= len(heatmaps):
                title.configure(text="No more channels")
                preview.configure(image="", text="—")
                continue

            heatmap = heatmaps[index]
            descriptor = (
                f"Channel {heatmap.channel}"
                if is_spatial
                else f"Vector segment {heatmap.channel + 1}"
            )
            title.configure(
                text=f"{descriptor} • |act| {heatmap.mean_absolute_activation:.3f}"
            )
            photo = ImageTk.PhotoImage(Image.fromarray(heatmap.rgb_image))
            self.feature_photos.append(photo)
            preview.configure(image=photo, text="", width=148, height=100)

    def _close(self) -> None:
        """Stop further callbacks, release webcam, and end the application."""
        self.closed = True
        self.webcam.stop()
        self.worker.stop()
        self.root.destroy()
