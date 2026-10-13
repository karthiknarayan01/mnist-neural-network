# mnist-neural-network

A neural network that recognizes handwritten digits, built **entirely
from scratch** with [NumPy](https://numpy.org/) — no PyTorch, no
TensorFlow, no scikit-learn. Every piece of math a neural network needs
(a forward pass, backpropagation, gradient descent, regularization) is
written out explicitly in plain, heavily-commented Python, so the whole
project doubles as a from-first-principles explanation of how these
networks actually learn.

This is the training half of a two-part project. The trained weights
produced here are consumed by
[**mnist-digit-recognizer**](https://github.com/karthiknarayan01/mnist-digit-recognizer),
a web app (re-implemented in TypeScript, no server required) where you
can upload or draw a digit and see this exact network recognize it live
in your browser.

## Why this project exists

This is a first neural-network project, built while working through
Michael Nielsen's [*Neural Networks and Deep
Learning*](https://github.com/mnielsen/neural-networks-and-deep-learning).
The goal was to go a step past the book's introductory example and apply
the specific set of improvements the book itself recommends for pushing
a plain sigmoid network's accuracy past 98% on MNIST — and to write the
whole thing with enough explanation that it's readable by someone who
has never used NumPy before.

## The network

```
784 inputs  →  100 neurons  →  100 neurons  →  100 neurons  →  10 outputs
 (pixels)       (hidden 1)       (hidden 2)       (hidden 3)     (digits 0-9)
```

| Choice | Value | Why |
|---|---|---|
| Neuron type | Sigmoid | The classic choice for a first network; squashes any input into a smooth 0–1 output. |
| Hidden layers | 3 layers × 100 neurons | Deeper than the book's single-hidden-layer example, to give the network more room to build up abstractions. |
| Cost function | Cross-entropy | Avoids the "learning slowdown" that plain squared-error cost suffers from when a sigmoid neuron is very wrong (see `src/cost_functions.py` for the full explanation). |
| Regularization | L2, λ configurable | Discourages individual weights from growing too large, which reduces overfitting to the training set. |
| Weight initialization | Gaussian, scaled by 1/√(inputs) | Keeps each neuron's initial input in a sensible range, so training doesn't start out stuck on the sigmoid curve's flat tails. |
| Optimizer | Mini-batch SGD, batch size 50 | Frequent, cheap gradient updates rather than one slow, exact update per epoch. |
| Learning rate | Starts large (0.5), automatically halved after 5 epochs with no validation improvement, training stops once it decays below 0.001 | Fast progress early on, careful fine-tuning later, without needing to hand-tune a fixed schedule. |

Every one of these choices is explained in plain language, inline, in
the source files themselves:

- [`src/activations.py`](src/activations.py) — the sigmoid neuron.
- [`src/cost_functions.py`](src/cost_functions.py) — cross-entropy cost,
  and why it's a better fit than squared error for sigmoid networks.
- [`src/network.py`](src/network.py) — the network itself: feedforward,
  backpropagation, mini-batch SGD, L2 regularization, and the
  variable-learning-rate training loop.
- [`src/mnist_loader.py`](src/mnist_loader.py) — turning the raw MNIST
  file into the shape the network trains on.

## The data

[MNIST](http://yann.lecun.com/exdb/mnist/): 70,000 handwritten digit
images, 28×28 pixels each, split into:

- **50,000** training images
- **10,000** validation images (used only to decide when to shrink the
  learning rate — never learned from directly)
- **10,000** test images (used only once, at the very end, for a final
  unbiased accuracy score)

## Results

<!-- BENCHMARK_START -->
| Metric | Value |
|---|---|
| **Final test accuracy** | **98.02%** (9,802 / 10,000) |
| Final validation accuracy | 97.88% |
| Best validation accuracy (any epoch) | 97.95% |
| Epochs run | 60 (hit the epoch cap — accuracy had largely plateaued) |
| Learning rate schedule | 0.5 → 0.25 (epoch 30) → 0.125 (epoch 38) → 0.0625 (epoch 52) |
| Total training time | ~7.7 minutes, on a laptop CPU, no GPU |

Validation accuracy climbed quickly in the first ~5 epochs (86% → 97%),
then kept inching upward as the learning rate automatically stepped
down whenever progress stalled for 5 epochs in a row — each drop in eta
is visible as a small jump in accuracy right afterward:

```
Epoch  1  eta=0.5     val_acc=86.36%
Epoch  5  eta=0.5     val_acc=96.63%
Epoch 10  eta=0.5     val_acc=97.04%
Epoch 29  eta=0.5     val_acc=97.35%   -> no improvement for 5 epochs, reducing eta to 0.25
Epoch 30  eta=0.25    val_acc=97.80%
Epoch 37  eta=0.25    val_acc=97.83%   -> no improvement for 5 epochs, reducing eta to 0.125
Epoch 42  eta=0.125   val_acc=97.90%
Epoch 51  eta=0.125   val_acc=97.90%   -> no improvement for 5 epochs, reducing eta to 0.0625
Epoch 58  eta=0.0625  val_acc=97.95%
Epoch 60  eta=0.0625  val_acc=97.88%
```

Full hyperparameters and the complete per-epoch log are in
[`benchmarks/benchmark.json`](benchmarks/benchmark.json).
<!-- BENCHMARK_END -->

## Running it yourself

```bash
git clone https://github.com/karthiknarayan01/mnist-neural-network.git
cd mnist-neural-network

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Downloads the MNIST dataset (~17 MB) if you don't already have it
python3 scripts/download_mnist.py

python3 scripts/train.py
```

Training prints live progress after every epoch, and finishes by saving:

- `model/trained_network.json` — the trained weights and biases (also
  used by the companion web app).
- `benchmarks/benchmark.json` — a full report of every hyperparameter
  and every epoch's accuracy, for reproducibility.

On a normal laptop CPU (no GPU needed — the network and dataset are
small enough that plain NumPy is plenty fast), training takes a few
minutes.

### Using the trained network afterwards

```python
import sys
sys.path.insert(0, "src")
from network import Network

net = Network.load("model/trained_network.json")
prediction = net.feedforward(some_28x28_image_as_a_784x1_column)
```

## Project structure

```
mnist-neural-network/
├── src/
│   ├── activations.py      # sigmoid neuron
│   ├── cost_functions.py   # cross-entropy cost
│   ├── network.py          # the Network class: training + inference
│   └── mnist_loader.py     # loads and reshapes the MNIST dataset
├── scripts/
│   ├── download_mnist.py   # fetches the raw dataset
│   └── train.py            # trains the network end-to-end
├── model/
│   └── trained_network.json  # the trained weights, ready to use
├── benchmarks/
│   └── benchmark.json      # hyperparameters + full per-epoch results
└── data/                   # (gitignored) raw MNIST file lives here
```

## Credit

Built while studying Michael Nielsen's *[Neural Networks and Deep
Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)*
— the architecture and training strategy here follow the "improvements"
chapter of that book, reimplemented independently with original code and
much more extensive explanatory comments.

## License

[MIT](LICENSE)
