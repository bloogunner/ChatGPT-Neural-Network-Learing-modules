# Live Webcam TensorFlow Neural-Network Explorer

A beginner-friendly multi-file Python app that uses **your computer's camera**
and a real **ImageNet-pretrained MobileNetV2** network.

It visualizes:

- Your webcam preview (on your computer).
- A simplified diagram of **real stages** inside a convolutional network.
- Five actual sampled feature-channel magnitudes per overview stage.
- A colored feature-map preview for **any** layer you click in the full list.
- All Keras layers in the real model, with type and output tensor shape.
- Top five ImageNet classification guesses for each sampled camera frame.
- **An optional learnable two-class head** trained on snapshots you label A/B,
  with an accurate diagram of the 12 final-layer connection weights.

**Important honesty about the visualization:** The flow lines and moving dot
are an **illustration** of signal travel. They are not individual synapses,
learned connection weights, or accurate per-layer execution timings. Every
shown layer is real; node brightness and feature maps are derived from actual
TensorFlow intermediate tensors. MobileNetV2 contains many thousands of
convolutional features and skip connections that cannot be drawn faithfully as
a small row of circles. The full model can be explored one layer at a time in
the layer browser.

## Windows + Python 3.12: installation

1. Install 64-bit Python 3.12 and VS Code if you do not already have them.
2. Unzip this folder and open **this folder** in VS Code.
3. In the VS Code **PowerShell** terminal, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Start the app:

```powershell
.\.venv\Scripts\python.exe main.py
```

**On the first run**, Keras downloads ImageNet-trained MobileNetV2 weights and
class labels. Internet is needed for that initial download only. Downloads
are cached under your local Keras folder, typically `%USERPROFILE%\.keras\`.
Later inferences run locally and webcam frames are not uploaded by this app.

**On native Windows**, current TensorFlow uses the CPU for this workflow.
TensorFlow versions after 2.10 do not offer standard native-Windows CUDA GPU
support. WSL2 has a separate GPU installation path if you want it later.
Your frame rate depends on CPU performance and model speed.

## Using the application

1. When status says the model is ready, click **Start camera**.
2. Allow Windows camera permissions if prompted.
3. Move different everyday objects in front of the camera.
4. Watch the approximate classification scores and changing network nodes.
5. Click a stage in the diagram, or any layer in the **full layer explorer**.
6. Inspect heatmaps: bright areas mean higher values **within that feature**.
7. Use **Pause analysis** to freeze the AI updates while keeping video live.
8. Use **Analyze one frame** to advance one inference at a time.
9. Use **Stop camera** to release your webcam.

## Files and separation of responsibilities

| File | What it does |
| --- | --- |
| `main.py` | Main entry point; starts Tkinter and the app. |
| `camera.py` | Opens the hardware camera, reads images, converts BGR -> RGB. |
| `vision_model.py` | Creates the actual pretrained network, probes intermediate outputs, converts selected features into heatmaps, and classifies images. |
| `camera_trainer.py` | Stores locally captured, labeled embeddings and trains a 1280 → 6 → 2 classifier head with gradient descent. |
| `worker.py` | Runs inference and optional head training on a background thread so the UI doesn't freeze. |
| `visualizer.py` | Draws the webcam, network stages, heatmap cards, and complete layer browser. |
| `settings.py` | Centralizes stage names, sizes, and visual settings. |
| `requirements.txt` | Installation packages and version ranges. |
| `tests/test_image_tools.py` | Small unit tests for processing activation arrays. |

### Spacing and comments

This project follows **PEP 8**, the Python community convention: **two blank
lines between top-level functions/classes** and one blank line between methods
inside a class. Every function/method also has a descriptive docstring
explaining its responsibility, inputs, outputs, and any important limitations.
Logical sections within functions have blank lines for readability.

### How a convolutional feature map differs from a normal neuron

For a simple Dense neuron, a scalar activation might be `0.7`. A convolutional
channel produces a two-dimensional *array* of activations, e.g. `56×56`.
A heatmap is a view of that 2-D channel; six thumbnail previews are six actual
sampled channels. As you move an object, the values change because the camera
image changes.

### What is learning here?

The primary ImageNet recognizer is **pretrained**: its weights remain fixed;
its **activations change** on new camera frames. Opening the training tab
activates a small, separately trainable head. This approach is called
**transfer learning**: use already learned visual features and teach a small
classifier what *you* mean by two categories.

To train the small head:

1. Place a distinctive object in front of your camera. In the **Train camera
   classifier** tab, rename **Class A** (e.g., `mug`) and click **Capture A**.
2. Move or rotate that object and capture at least three A examples. More
   varied shots are helpful; ten per class is a better starting point.
3. Switch to another object, rename **Class B** (e.g., `phone`), and capture
   at least three B examples.
4. Click **Train 1 epoch** to see an actual parameter update. **Train 10
   epochs** repeats the update ten times. Read the loss and look for weight
   color and thickness changes in the **6 → 2** diagram.
5. Watch **Your classifier** scores as you change objects in the webcam. If
   results are inconsistent, capture more varied examples and train again.
6. Click **Reset** to discard the small head's weights and collected samples.

The blue/red lines in the head diagram reflect **real trainable weights**;
red means negative weight and blue means positive. The full CNN still has far
too many weights to draw individually. Training is done **only** when you
click an explicit Train button; simply streaming the camera does not cause
weights to change. Your collected examples live only in RAM and are cleared
when you exit the program.

## Troubleshooting

**`numpy.core.multiarray failed to import`**: You probably installed mutually
incompatible packages into a shared Python environment. Use the clean `.venv`
commands above, and make sure VS Code launches its `.venv` Python. Do not
install global NumPy or Matplotlib for this app; Matplotlib is not required.

**Camera black / cannot open**: Check Windows **Settings > Privacy & security >
Camera** and allow desktop apps. Quit Teams, Zoom, or other software using the
camera. Change the **Camera index** from 0 to 1 and press Start again.

**First start seems slow**: ImageNet weights are downloaded on first launch.
The interface stays open while the model loads; the status bar reports work.

**Recognition is wrong**: MobileNetV2 was trained on ImageNet's **1,000 known
categories**. It is not a general-purpose object detector. It will make an
entire-image guess and cannot place labeled bounding boxes around multiple
objects. Your room's lighting and the object's placement can change results.

**Feature map doesn't look like the object**: Intermediate CNN feature maps
represent learned mathematical patterns, not photos or semantic labels.
Showing a single channel's values as false colors is primarily educational.

**Application cannot create a GUI in a remote terminal**: This program uses a
local desktop Tkinter window and a physical webcam; it must run in Windows
rather than a headless server.

## Privacy

The app sends webcam frames to TensorFlow **inside this same Python process**.
It never records video, writes images to disk, or sends camera frames to a
remote API. Labeled examples are temporarily retained in RAM as numeric
embeddings, not picture files. On first launch, Keras does make network requests to obtain public
model files and class-label data; those are not camera images.
