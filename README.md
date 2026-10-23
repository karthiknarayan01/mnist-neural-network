# mnist-neural-network

A neural network that recognizes handwritten digits, built **entirely
from scratch** with [NumPy](https://numpy.org/) — no PyTorch, no
TensorFlow, no scikit-learn. Every piece of math (forward pass,
backpropagation, gradient descent, regularization) is written out
explicitly, with comments aimed at someone who's never used NumPy before.

The trained weights here power
[**mnist-digit-recognizer**](https://github.com/karthiknarayan01/mnist-digit-recognizer),
a web app where you can upload or draw a digit and watch this network
recognize it live.

New to convolutional layers? Read
[`how-computers-see-numbers.pdf`](docs/how-computers-see-numbers.pdf) —
a short, genuinely kid-friendly explanation with pictures.

## Two architectures

This project implements two independent networks, trained and evaluated
through the same `scripts/train.py` entry point (flip `ARCHITECTURE` to
`"dense"` or `"cnn"`):

```
Dense (network.py):
  784 inputs → 100 → 100 → 100 → 10 outputs, sigmoid neurons, cross-entropy cost

CNN (conv_network.py):
  28×28 image → Conv(20, 5×5) → ReLU → Pool → Conv(40, 5×5) → ReLU → Pool
             → Dense(640→100) → ReLU → Dense(100→10), softmax + cross-entropy
```

Both use mini-batch SGD, L2 regularization, and an automatic
learning-rate decay schedule. The CNN's conv stages are configurable
(`conv_blocks` in `ConvNetwork`) — any number of `(filters, size)` pairs.

One deliberate difference worth knowing: the dense network's output is
10 *independent* sigmoid neurons (matching the book this project started
from), not true softmax — the ten values aren't forced to sum to 1. The
CNN uses real softmax. Both are explained inline in the source.

## Results

| Configuration | Test accuracy | Training time |
|---|---|---|
| Dense | 98.02% | ~8 min |
| Dense + rotation augmentation | 98.47% | ~39 min |
| CNN (1 conv layer) | 98.96% | ~24 min |
| CNN + augmentation | 99.20% | ~4 hr |
| **CNN, 2 conv layers + augmentation** | **99.35%** | ~7 hr |

Three independent things helped, each less than the last: switching
architecture (dense → CNN) gained the most, data augmentation (±15°
rotations, 50k → 550k images) added some more, and going deeper (1 conv
layer → 2) added a bit more on top. The final architecture wasn't a
guess — a handful of candidates (wider, deeper, deeper+wider, different
learning rates) were screened cheaply first on fast, unaugmented runs;
going deeper beat going wider at the same compute cost, and bigger
wasn't automatically better. See
[`scripts/screen_architectures.py`](scripts/screen_architectures.py)
and [`benchmarks/architecture_screening.json`](benchmarks/architecture_screening.json).

Full hyperparameters and per-epoch logs for every run above are in
`benchmarks/*.json`; corresponding weights are in `model/*.json`.

## Running it yourself

```bash
git clone https://github.com/karthiknarayan01/mnist-neural-network.git
cd mnist-neural-network

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/download_mnist.py
python3 scripts/train.py
```

`scripts/train.py` trains whichever architecture `ARCHITECTURE` is set
to, saving weights to `model/` and a benchmark report to `benchmarks/`.
Training augmented data takes hours on a laptop CPU — keep the machine
awake, since a sleep mid-run gets logged as a very slow epoch.

```python
import sys
sys.path.insert(0, "src")

from network import Network
net = Network.load("model/trained_network.json")
prediction = net.feedforward(some_784x1_column)

from conv_network import ConvNetwork
net = ConvNetwork.load("model/trained_cnn.json")
logits = net.forward(some_batch_shaped_Nx1x28x28)
```

## Project structure

```
mnist-neural-network/
├── src/
│   ├── activations.py, cost_functions.py   # sigmoid + cross-entropy (dense)
│   ├── network.py                          # Network: dense training + inference
│   ├── conv_layers.py, conv_network.py     # Conv2D/ReLU/Pool/Dense + ConvNetwork
│   ├── mnist_loader.py                     # loads and reshapes MNIST
│   └── augmentation.py                     # rotation-based data augmentation
├── scripts/
│   ├── download_mnist.py
│   ├── train.py                 # trains either architecture (see ARCHITECTURE flag)
│   └── screen_architectures.py  # cheap comparison of CNN architecture candidates
├── model/        # trained weights, one file per configuration above
├── benchmarks/   # per-run hyperparameters + per-epoch logs
└── data/         # (gitignored) raw MNIST file
```

## Credit

Built while studying Michael Nielsen's *[Neural Networks and Deep
Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)*
— the dense network follows that book's "improvements" chapter,
reimplemented independently with original code and comments.

## License

[MIT](LICENSE)
