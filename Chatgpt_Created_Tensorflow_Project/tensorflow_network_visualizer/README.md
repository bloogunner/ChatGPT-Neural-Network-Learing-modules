# Live TensorFlow Neural Network Visualizer

Visualize an **actual TensorFlow/Keras network** training on the equation `y = 2x + 1`. Unlike an ordinary loss chart, this shows **all neurons**, **every connection weight**, **biases**, **live activations**, and a **forward-pass signal animation**. TensorFlow updates the weights during training; the visualizer reads and redraws the **real** parameters after every update.

## Project files

```text
Tensorflow_hello_world/
├── main.py          # Main entry point: starts the Tkinter window
├── network.py       # TensorFlow model, training, and state snapshots
├── visualizer.py    # Network diagram, animations, and controls
├── requirements.txt # Python package dependencies
└── README.md
```

## Setup (Windows / VS Code / Python 3.12)

From a VS Code terminal opened in the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

You can also select `.venv\Scripts\python.exe` as your interpreter in VS Code via **Python: Select Interpreter**.

**Why use a virtual environment?** Your previous `numpy.core.multiarray failed to import` error suggests incompatible packages in the global Python installation. A fresh `.venv` installs compatible dependencies independently of that installation. This project does **not** use Matplotlib; the animated network is drawn with built-in Tkinter.

If Windows reports Tkinter is missing, reinstall/modify Python with the **tcl/tk and IDLE** option selected.

## How to use it

- **Start/Pause:** run or pause the animation and training.
- **Step:** go one animation phase at a time (the weight update occurs every fourth phase).
- **Reset:** rebuild the model with its initial weights.
- **Choose input:** change the example whose activations are being shown (`x=1` through `x=10`). **Training still uses all five fixed examples**, so switching the displayed example doesn't change the lessons being learned.
- **Speed:** adjust the time between steps. Slower speed is easier to observe, faster is easier to reach 500 epochs.

## What the colors and values mean

- **Blue node:** the input value `x`.
- **Green nodes H1–H4:** outputs of four hidden neurons.
- **Purple node:** the output prediction `ŷ`.
- **Teal edge:** a positive connection weight.
- **Orange edge:** a negative connection weight.
- **Thicker edge:** a weight with larger magnitude.
- **Small moving dots:** a *visual representation* of the forward pass; they are not individual data packets produced by TensorFlow.
- **Bias numbers:** additional learned parameters added at a neuron.
- **MSE:** mean squared prediction error on all five training examples.

## Why not just one neuron?

The original Hello World model had a single neuron (`x → output`). It only had **one weight and one bias**, so there was almost no internal network to look at. This demonstration changes the architecture to **1 input → 4 hidden neurons → 1 output** to show several real neuron connections. Both layers are **linear**, so the whole network still represents a straight-line function and learns the same rule. In particular, the outputs and connection parameters shown here come from the actual trained model, **not from a precomputed or fictional animation**.

## Implementation notes

The GUI animation steps through input → hidden → output → weight update. A single weight-update step runs `model.train_on_batch()` for the complete set of five training examples. The highlighted input can be changed independently. Snapshots compute the same linear matrix operations as Keras using the *current* values read by `layer.get_weights()`.

Training uses Adam rather than the original SGD so convergence is generally smoother for the expanded network. Prediction at `x=10` should approach `21` as the model learns; the exact trajectory is determined by initialization and optimizer updates.
