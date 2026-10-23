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

**New to the idea of a convolutional layer?** Read
[`how-computers-see-numbers.pdf`](docs/how-computers-see-numbers.pdf)
first — a short, plain-language (genuinely kid-friendly) explanation of
what makes the CNN below different from the plain network, with
pictures, before diving into the code.

## Why this project exists

This is a first neural-network project, built while working through
Michael Nielsen's [*Neural Networks and Deep
Learning*](https://github.com/mnielsen/neural-networks-and-deep-learning).
The goal was to go a step past the book's introductory example and apply
the specific set of improvements the book itself recommends for pushing
a plain sigmoid network's accuracy past 98% on MNIST — and to write the
whole thing with enough explanation that it's readable by someone who
has never used NumPy before.

## Two architectures, side by side

This project implements **two** completely different kinds of network,
on purpose, so they can be read and compared directly:

| | `network.py` → `Network` | `conv_network.py` → `ConvNetwork` |
|---|---|---|
| Architecture | Fully-connected (784 → 100 → 100 → 100 → 10) | Convolutional (conv → pool → conv → pool → 100 → 10) |
| Neuron type | Sigmoid | ReLU (hidden), softmax (output) |
| Cost function | Cross-entropy | Softmax cross-entropy |
| Processes a mini-batch as | 50 separate images, one at a time, in a Python loop | One batched 4D array, all 50 images at once |

Both are trained and evaluated through the exact same `scripts/train.py`
entry point — flip the `ARCHITECTURE` constant at the top of that file
between `"dense"` and `"cnn"` to choose which one a run trains. Both
expose the same `.SGD(...)`, `.accuracy(...)`, `.save(...)` /
`.load(...)` interface, so the training loop, L2 regularization, and
adaptive learning-rate schedule are genuinely shared code — the only
real difference between a `"dense"` run and a `"cnn"` run is the layers
themselves.

## The dense network

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
| Data augmentation | Each training image is rotated ±15°, in 10 evenly-spaced steps, adding 10 extra copies per image (50,000 → 550,000 training examples) | Teaches the network that a digit's identity doesn't change with small slants in handwriting, instead of letting it memorize incidental quirks of the original 50,000 images. See `src/augmentation.py`. |

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
- [`src/augmentation.py`](src/augmentation.py) — expanding the training
  set with rotated copies of each image.

**A note on sigmoid vs. softmax:** the output layer here is 10
independent sigmoid neurons paired with cross-entropy cost, exactly as
in the book — *not* a softmax layer. Each output neuron estimates "how
much does this look like my digit?" on its own 0–1 scale; the ten
values aren't forced to sum to 1 the way true softmax probabilities
are. This keeps the math consistent throughout the project (the same
`CrossEntropyCost.delta` simplification that avoids the "learning
slowdown" problem relies on a sigmoid output layer), and is what the
companion web app's confidence bars are built on, normalized by their
sum to display as friendly percentages.

## The convolutional network (CNN)

```
28x28x1 image
    -> Conv2D(1 -> 20 filters, 5x5)     => 24x24x20
    -> ReLU
    -> MaxPool2D(2x2)                   => 12x12x20
    -> Conv2D(20 -> 40 filters, 5x5)    => 8x8x40
    -> ReLU
    -> MaxPool2D(2x2)                   => 4x4x40
    -> Flatten                          => 640
    -> Dense(640 -> 100)
    -> ReLU
    -> Dense(100 -> 10)                 => 10 raw scores ("logits")
    -> softmax + cross-entropy cost
```

Two rounds of "scan for patterns, then zoom out," followed by the same
kind of fully-connected "voting" layers the dense network uses. The
second round doesn't see raw pixels — it sees the *first* round's 20
pattern maps, so its filters can learn combinations of simple patterns
(a curve next to a line, say) that one scanning pass alone can't
express. See
[`how-computers-see-numbers.pdf`](docs/how-computers-see-numbers.pdf) for
the plain-language version of why scanning for patterns helps at all;
the "why two rounds instead of one" result is measured below, under
[Choosing this architecture](#choosing-this-architecture-architecture-screening).

| Choice | Value | Why |
|---|---|---|
| Convolutional layers | 2 stages: 20 filters then 40, both 5×5, "valid" padding | Each filter scans for one simple pattern; stacking a second stage on the first's output lets it recognize combinations instead of just individual edges and curves. Chosen over 1 stage, 1 wider stage, and a bigger 2-stage version by screening all four cheaply first — see below. |
| Activation | ReLU (`max(0, z)`) | Unlike sigmoid, ReLU's slope never shrinks for positive inputs, which keeps gradients flowing cleanly through a deeper stack of layers — the standard choice for modern CNNs. See `conv_layers.py`. |
| Pooling | Max pooling, 2×2, after each conv stage | Shrinks each feature map by keeping only its strongest signal in each small zone, which both speeds up later layers and makes the network less sensitive to a pattern landing a pixel or two off from where it did in training. |
| Output layer | Softmax + cross-entropy | *True* softmax this time (unlike the dense network) — the 10 output probabilities are forced to sum to exactly 1, which is the mathematically "correct" way to express "exactly one of these 10 digits is right." |
| Everything else | Same as the dense network | Mini-batch SGD (batch size 50), L2 regularization, the same automatic learning-rate decay schedule. |

Implemented in [`src/conv_layers.py`](src/conv_layers.py) (the
individual layer types: `Conv2D`, `ReLU`, `MaxPool2D`, `Flatten`,
`Dense`, `softmax_cross_entropy`) and
[`src/conv_network.py`](src/conv_network.py) (`ConvNetwork`, which
chains an arbitrary number of these "scan, then zoom out" stages
together — one line, `conv_blocks=((20, 5), (40, 5))`, is the only
difference between this architecture and the single-stage version this
project started with).

**A note on batching:** the dense network's training loop processes one
image at a time in a Python loop (simple to follow, and plenty fast for
a fully-connected network). A CNN's convolution is a much more expensive
operation, so `ConvNetwork` instead processes a whole mini-batch as a
single 4D array (`batch_size x channels x height x width`) and runs it
through every layer in one batched, vectorized pass — the same
"let NumPy do the looping, not Python" idea used everywhere else in this
project, just applied at the scale of a whole mini-batch instead of a
single number.

**A note on correctness:** convolutional backpropagation is
considerably trickier to get right than the dense network's. Before
trusting this implementation with a real training run, every layer's
`backward()` was checked against a numerical (finite-difference)
gradient on small random inputs — see the gradient-check approach
described in `src/conv_layers.py`'s docstrings; all layers matched to
within floating-point tolerance.

Stacking a second conv stage on top surfaced a subtlety worth recording
rather than quietly fixing and moving on: checking the *whole* two-stage
network end-to-end with the same finite-difference epsilon used for a
single layer produced spurious failures — ReLU and max-pooling are both
non-smooth (a kink at zero, a hard switch between which position in a
pool "wins"), and a deeper chain gives a numeric perturbation more
chances to land exactly on one of those seams, which looks like a wrong
gradient but is actually a limitation of finite differences, not a bug.
Shrinking the step size and sampling many points (rather than trusting
one worst-case sample) made the real picture clear: median relative
error around 1e-9, consistent with correct backward math. Each layer's
*own* math didn't change — this was purely about how to verify a deeper
composition of already-correct pieces without fooling yourself.

## The data

[MNIST](http://yann.lecun.com/exdb/mnist/): 70,000 handwritten digit
images, 28×28 pixels each, split into:

- **50,000** training images — expanded to **550,000** via rotation-based
  augmentation before training (see above); only the training set is
  ever augmented
- **10,000** validation images (used only to decide when to shrink the
  learning rate — never learned from directly, never augmented)
- **10,000** test images (used only once, at the very end, for a final
  unbiased accuracy score — never augmented)

## Results — dense network

<!-- BENCHMARK_START -->
| Metric | Value |
|---|---|
| **Final test accuracy** | **98.47%** (9,847 / 10,000) |
| Final validation accuracy | 98.36% |
| Best validation accuracy (any epoch) | 98.40% |
| Training examples | 550,000 (50,000 original + 500,000 rotated copies) |
| Epochs run | 30 (hit the epoch cap — accuracy had largely plateaued) |
| Learning rate schedule | 0.5 → 0.25 (epoch 19) → 0.125 (epoch 25) → 0.0625 (epoch 30) |
| Total training time | ~38.8 minutes, on a laptop CPU, no GPU |

### Effect of rotation-based data augmentation

The network was first trained on the plain 50,000-image training set,
then again after expanding it with rotated copies (see "Data
augmentation" above). Augmentation measurably helped:

| | Without augmentation | With augmentation (±15° rotations) |
|---|---|---|
| Training examples | 50,000 | 550,000 |
| Epochs run | 60 | 30 |
| Final test accuracy | 98.02% | **98.47%** (+0.45 points) |
| Training time | ~7.7 min | ~38.8 min |

This matches the general expectation for this kind of technique on a
plain, fully-connected sigmoid network (no convolutional layers): it
reliably helps — by making the network more robust to the natural
variation in real handwriting rather than letting it overfit to the
exact slant of the original 50,000 images — but the improvement is
modest, not dramatic. Architectures without convolutional layers tend
to plateau somewhere in the 98–98.5% range on MNIST regardless of how
much more data you add; pushing meaningfully past that would need a
different kind of network (e.g. a CNN), not just more training examples.

Validation accuracy on the augmented run climbed past 97% within the
first epoch (each epoch now sees 11x more examples) and kept inching
upward as the learning rate automatically stepped down whenever
progress stalled for 5 epochs in a row:

```
Epoch  1  eta=0.5     val_acc=97.08%
Epoch  9  eta=0.5     val_acc=98.15%
Epoch 18  eta=0.5     val_acc=97.94%   -> no improvement for 5 epochs, reducing eta to 0.25
Epoch 19  eta=0.25    val_acc=98.40%   (best epoch)
Epoch 24  eta=0.25    val_acc=98.29%   -> no improvement for 5 epochs, reducing eta to 0.125
Epoch 29  eta=0.125   val_acc=98.26%   -> no improvement for 5 epochs, reducing eta to 0.0625
Epoch 30  eta=0.0625  val_acc=98.36%
```

Full hyperparameters and the complete per-epoch log are in
[`benchmarks/benchmark.json`](benchmarks/benchmark.json).
<!-- BENCHMARK_END -->

## Results — single-layer CNN, no augmentation

<!-- CNN_BENCHMARK_START -->
| Metric | Value |
|---|---|
| **Final test accuracy** | **98.96%** (9,896 / 10,000) |
| Final validation accuracy | 98.94% |
| Best validation accuracy (any epoch) | 98.99% (epoch 14) |
| Training examples | 50,000 (no augmentation on this run) |
| Epochs run | 25 (hit the epoch cap — accuracy had largely plateaued) |
| Learning rate schedule | 0.1 → 0.05 (epoch 19) → 0.025 (epoch 25) |
| Total training time | ~24 minutes, on a laptop CPU, no GPU |

The CNN reached 97.67% validation accuracy after just its **first**
epoch — roughly what the dense network needed 3-4 epochs to reach, and
close to what it took the dense network 10+ epochs to *exceed*. By
epoch 4 it had already passed the dense network's best-ever result with
augmentation (98.47%, achieved only with the help of 11x more training
data) using the *plain*, unaugmented 50,000-image training set:

```
Epoch  1  eta=0.1    val_acc=97.67%
Epoch  4  eta=0.1    val_acc=98.67%   (already beats dense+augmentation's 98.47%)
Epoch 10  eta=0.1    val_acc=98.93%
Epoch 14  eta=0.1    val_acc=98.99%   (best epoch)
Epoch 19  eta=0.1    val_acc=98.98%   -> no improvement for 5 epochs, reducing eta to 0.05
Epoch 24  eta=0.05   val_acc=98.97%   -> no improvement for 5 epochs, reducing eta to 0.025
Epoch 25  eta=0.025  val_acc=98.94%
```

Full hyperparameters and the complete per-epoch log are in
[`benchmarks/benchmark_cnn_no_augmentation.json`](benchmarks/benchmark_cnn_no_augmentation.json).
<!-- CNN_BENCHMARK_END -->

## Results — single-layer CNN + augmentation

<!-- CNN_AUG_BENCHMARK_START -->
The one combination the section above left untried: training the CNN
on the same rotation-augmented, 550,000-image set that helped the dense
network. It was the most promising remaining lever at the time — the
strongest architecture paired with the technique that had already
helped the weaker one — and it worked:

| Metric | Value |
|---|---|
| **Final test accuracy** | **99.20%** (9,920 / 10,000) |
| Final validation accuracy | 99.13% |
| Best validation accuracy (any epoch) | 99.16% (epoch 15) |
| Training examples | 550,000 (50,000 original + 500,000 rotated copies) |
| Epochs run | 25 (hit the epoch cap) |
| Learning rate schedule | 0.1 → 0.05 (epoch 14) → 0.025 (epoch 21) |
| Total training time | ~3.9 hours, on a laptop CPU, no GPU |

Unlike every run before it, validation accuracy was already past 99%
by epoch 4 and never dropped back below it — the combination is both
more accurate and more stable than either technique alone:

```
Epoch  1  eta=0.1    val_acc=98.78%
Epoch  4  eta=0.1    val_acc=99.06%   (first epoch over 99%)
Epoch 15  eta=0.05   val_acc=99.16%   (best epoch)
Epoch 25  eta=0.025  val_acc=99.13%
```

**A note on that training time.** The logged total was actually 4.9
hours; one epoch (23) took 70 minutes instead of its usual 9, because
the laptop went to sleep mid-epoch and the process simply resumed when
it woke — no lost work, no effect on the result, just inflated
wall-clock time for that one epoch. ~3.9 hours is that run with the
sleep gap subtracted back out, which matches the other 24 epochs'
pace. If you're running this yourself, keep the machine from sleeping.

This was this project's best result for a while. It's since been
superseded by a deeper architecture (below) and kept here, with its
weights and benchmark, as
[`model/trained_cnn_single_conv_augmented_backup.json`](model/trained_cnn_single_conv_augmented_backup.json)
and
[`benchmarks/benchmark_cnn_single_conv_augmented_backup.json`](benchmarks/benchmark_cnn_single_conv_augmented_backup.json),
for comparison.
<!-- CNN_AUG_BENCHMARK_END -->

## Choosing this architecture: architecture screening

Going from one convolutional stage to two is a bigger change than
tuning a learning rate, and a full run to test it costs hours — not
something to spend on a hunch. So before committing to any single
change, several candidates were screened cheaply: 8 epochs each, on the
fast unaugmented 50,000-image set, same evaluation each time, so
whichever curve is highest after the same amount of work is the one
worth the expensive full run.

| Candidate | Best val. accuracy (8 epochs) | Seconds/epoch |
|---|---|---|
| Baseline: 1 conv stage, 20 filters | 98.80% | 54.4s |
| Wider: 1 conv stage, 40 filters | 98.85% | 108.7s |
| **Deeper: 2 conv stages, 20→40 filters** | **99.01%** | **173.5s** |
| Deeper + wider: 2 conv stages, 32→64 filters, bigger hidden layer | 99.03% | 385.5s |
| Deeper, higher learning rate (eta 0.2) | 99.03% | 177.8s |
| Deeper, lighter regularization (λ 0.02) | 98.94% | 165.7s |

Two things stand out. First, **depth beat width**: going from 20 to 40
filters in a single stage bought +0.05 points for roughly 2x the
compute; splitting those same filters across two stages instead bought
+0.21 points for about the same 2x cost. Second, **bigger wasn't
better**: the deeper-and-wider candidate matched "deeper, higher
learning rate" at more than double the training time, for a +0.02-point
difference well within the noise of an 8-epoch run — scaling up every
dimension at once mostly bought slower epochs, not a better network.

Depth with a higher starting learning rate (0.2 instead of 0.1) won
outright — same cost as the plain deeper candidate, best peak of the
six — so that's what the full run below uses. Full details for every
candidate are in
[`benchmarks/architecture_screening.json`](benchmarks/architecture_screening.json);
the screening script itself is
[`scripts/screen_architectures.py`](scripts/screen_architectures.py).

## Results — deeper CNN + augmentation

<!-- CNN_DEEP_AUG_BENCHMARK_START -->
The screening winner, trained for real: 2 convolutional stages
(20 filters then 40), on the full rotation-augmented 550,000-image set.

| Metric | Value |
|---|---|
| **Final test accuracy** | **99.35%** (9,935 / 10,000) |
| Final validation accuracy | 99.33% |
| Best validation accuracy (any epoch) | 99.35% (epoch 12) |
| Training examples | 550,000 (50,000 original + 500,000 rotated copies) |
| Epochs run | 15 (hit the epoch cap) |
| Learning rate schedule | 0.2 → 0.1 (epoch 10) |
| Total training time | ~7.0 hours, on a laptop CPU, no GPU |

```
Epoch  1  eta=0.2    val_acc=99.18%   (one epoch from matching the single-layer run's 25-epoch result)
Epoch  9  eta=0.2    val_acc=99.15%   -> no improvement for 5 epochs, reducing eta to 0.1
Epoch 10  eta=0.1    val_acc=99.32%
Epoch 12  eta=0.1    val_acc=99.35%   (best epoch)
Epoch 15  eta=0.1    val_acc=99.33%
```

The first epoch alone (99.18%) nearly matched the single-layer CNN's
*entire 25-epoch, 3.9-hour result* (99.20%) — a reminder that the
screening table above wasn't measuring a subtle effect. Progress inside
each learning-rate stage was noisy rather than monotonic (epoch 7 dipped
to 99.02% mid-run), which is ordinary mini-batch variance, not a
problem; the adaptive schedule's job is exactly to ride that out and
drop the learning rate once the noise stops resolving into further
gains, which is what the halving at epoch 9 did.

**A note on that training time, again.** This run hit the same
sleeping-laptop issue as the single-layer augmented run, twice: epoch 3
and epoch 15 both show as roughly 2-8x their normal length in the raw
log. The ~7.0 hours above has both gaps subtracted back out, consistent
with the other 13 epochs' pace (~28 minutes each). The lesson from last
time evidently didn't take — keep the machine awake for anything
multi-hour.

Full hyperparameters and the complete per-epoch log are in
[`benchmarks/benchmark_cnn.json`](benchmarks/benchmark_cnn.json).
<!-- CNN_DEEP_AUG_BENCHMARK_END -->

## The full comparison

| | Dense | Dense + aug. | 1-layer CNN | 1-layer CNN + aug. | **2-layer CNN + aug.** |
|---|---|---|---|---|---|
| Training examples | 50,000 | 550,000 | 50,000 | 550,000 | 550,000 |
| Epochs run | 60 | 30 | 25 | 25 | 15 |
| Training time | ~7.7 min | ~38.8 min | ~24 min | ~3.9 hr | ~7.0 hr |
| **Final test accuracy** | 98.02% | 98.47% | 98.96% | 99.20% | **99.35%** |

Three independent levers, all real, and each one smaller than the last:
switching architecture (dense → CNN) bought the biggest single jump
(+0.94 points); augmentation added +0.24 more on top of it; going
deeper (1 conv stage → 2) added a further +0.15. Diminishing returns,
but not yet zero — each change made a network that had more room left
to use what the next change offered, which matches the "why would a CNN
help?" discussion earlier in this README: a fully-connected network's
ceiling on MNIST is architectural, so data alone couldn't close the gap
to a network that understood pixel neighborhoods; a single-stage CNN's
ceiling is *how much structure it can represent in one pass*, so going
deeper could still buy more even after architecture and data had both
already helped once.

The cost grows faster than the accuracy does. A single-stage CNN epoch
is roughly 7-8x slower than a dense epoch; stacking a second conv stage
roughly triples that again (measured in screening, same unaugmented
data: 173.5s/epoch for 2 stages versus 54.4s for 1). Training time,
start to finish, tells the cumulative story directly: ~7.7 minutes for
the original dense baseline versus ~7.0 hours for the deepest,
augmented configuration — about 55x longer, for a cumulative
+1.33-point gain. Whether that trade is worth it depends entirely on
what the accuracy is for — for this project, recording the result,
honestly costed, was the point.

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

`scripts/train.py` trains whichever architecture its `ARCHITECTURE`
constant is set to (`"dense"` or `"cnn"`), prints live progress after
every epoch, and finishes by saving:

- `model/trained_network.json` (dense) or `model/trained_cnn.json`
  (CNN) — the trained weights, ready to reload in Python or in the
  companion web app.
- `benchmarks/benchmark.json` (dense) or `benchmarks/benchmark_cnn.json`
  (CNN) — a full report of every hyperparameter and every epoch's
  accuracy, for reproducibility.

`CNN_CONV_BLOCKS` at the top of the script controls how many
convolutional stages the CNN gets — a tuple of `(filters, filter_size)`
pairs, one per stage; it defaults to `((20, 5), (40, 5))`, the
screening winner described above. Set it to a single-entry tuple like
`((20, 5),)` to go back to the simpler, faster, single-stage network.

On a normal laptop CPU (no GPU needed — plain NumPy is fast enough for
these sizes, just not fast):

| Configuration | Roughly |
|---|---|
| Dense, no augmentation | under 10 minutes |
| Dense, with rotation augmentation (550,000 examples) | ~35-40 minutes |
| 1-layer CNN, no augmentation | ~20-25 minutes |
| 1-layer CNN, with augmentation | ~3.9 hours |
| Architecture screening (6 candidates, 8 epochs each, unaugmented) | ~2.4 hours |
| **2-layer CNN, with augmentation (the default)** | **~7 hours** |

`AUGMENT_WITH_ROTATIONS` at the top of `scripts/train.py` defaults to
on — it's what produced every best result in this README. Flip it off,
or set `CNN_CONV_BLOCKS` back to one stage, for a much faster run if
you don't need the last point or two of accuracy; the full default
configuration is most of a working day on a laptop CPU, so make sure
the machine won't sleep partway through (see the notes under "Results —
deeper CNN + augmentation" above for what happens if it does — twice).

### Using a trained network afterwards

```python
import sys
sys.path.insert(0, "src")

# The dense network:
from network import Network
net = Network.load("model/trained_network.json")
prediction = net.feedforward(some_28x28_image_as_a_784x1_column)

# ...or the CNN (this repo's best result: 2 conv stages, trained with augmentation):
from conv_network import ConvNetwork
net = ConvNetwork.load("model/trained_cnn.json")
logits = net.forward(some_batch_of_images_shaped_Nx1x28x28)
```

## Project structure

```
mnist-neural-network/
├── src/
│   ├── activations.py      # sigmoid neuron (dense network)
│   ├── cost_functions.py   # cross-entropy cost (dense network)
│   ├── network.py          # Network: the dense network's training + inference
│   ├── conv_layers.py      # Conv2D, ReLU, MaxPool2D, Flatten, Dense, softmax (CNN)
│   ├── conv_network.py     # ConvNetwork: the CNN's training + inference, any number of conv stages
│   ├── mnist_loader.py     # loads and reshapes the MNIST dataset
│   └── augmentation.py     # expands training data with rotated copies
├── scripts/
│   ├── download_mnist.py          # fetches the raw dataset
│   ├── train.py                   # trains either architecture end-to-end (see ARCHITECTURE flag)
│   └── screen_architectures.py    # cheaply compares CNN architecture candidates (see "Choosing this architecture")
├── model/
│   ├── trained_network.json                        # dense network's trained weights
│   ├── trained_cnn.json                             # CNN's trained weights (2 conv stages + augmentation, 99.35%, the best in this repo)
│   ├── trained_cnn_single_conv_augmented_backup.json  # 1 conv stage + augmentation (99.20%), kept for comparison
│   └── trained_cnn_no_augmentation.json             # 1 conv stage, no augmentation (98.96%), kept for comparison
├── benchmarks/
│   ├── benchmark.json                               # dense network: hyperparameters + per-epoch results
│   ├── benchmark_cnn.json                           # best CNN (2 conv stages + augmentation): hyperparameters + per-epoch results
│   ├── benchmark_cnn_single_conv_augmented_backup.json  # 1 conv stage + augmentation: same, kept for comparison
│   ├── benchmark_cnn_no_augmentation.json           # 1 conv stage alone: same, kept for comparison
│   └── architecture_screening.json                  # all 6 screened candidates: hyperparameters + per-epoch results
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
