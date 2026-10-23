"""
train.py
========

The command-line entry point for this project: loads the MNIST data,
builds a network, trains it, prints a benchmark summary, and saves the
trained weights so they can be reused later -- either back in Python or
ported into the TypeScript web app.

**Two architectures, one feature flag.** This project implements two
completely different kinds of network:

- ``"dense"`` -- ``network.py``'s ``Network``: a plain, fully-connected
  sigmoid network, trained with cross-entropy cost.
- ``"cnn"`` -- ``conv_network.py``'s ``ConvNetwork``: a convolutional
  network (a conv layer + pooling + fully-connected layers), trained
  with softmax and cross-entropy cost.

Flip the ``ARCHITECTURE`` constant below to switch which one this script
trains. Everything else -- loading the data, optional rotation
augmentation, the adaptive learning-rate schedule, saving a benchmark
report -- is shared, so the only thing that changes between the two runs
is genuinely the network architecture itself. Each architecture saves to
its own files (``model/trained_network.json`` / ``benchmarks/benchmark.json``
for the dense network, ``model/trained_cnn.json`` / ``benchmarks/benchmark_cnn.json``
for the CNN), so training one never overwrites the other -- both can
live in this repo side by side for comparison.

Run it from the project root with:

    source .venv/bin/activate
    python3 scripts/train.py
"""

import json
import os
import sys
import time

# Make the `src/` package importable regardless of which directory this
# script is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mnist_loader import load_data_wrapper  # noqa: E402
from network import Network  # noqa: E402
from conv_network import ConvNetwork, prepare_conv_data  # noqa: E402
from augmentation import expand_training_data  # noqa: E402

# --- Pick which architecture to train ---
ARCHITECTURE = "cnn"  # "dense" or "cnn"

# --- Dense network hyperparameters (used when ARCHITECTURE == "dense") ---
# These choices are explained in depth in the project README; in short,
# they follow the "improved" configuration from chapter 3 of Michael
# Nielsen's "Neural Networks and Deep Learning", targeting 98%+ test
# accuracy on MNIST with a plain, from-scratch sigmoid network.
DENSE_SIZES = [784, 100, 100, 100, 10]        # 3 hidden layers of 100 neurons each
DENSE_EPOCHS = 30                             # upper bound (lower when augmenting -- see below)
DENSE_MINI_BATCH_SIZE = 50
DENSE_ETA = 0.5                               # starting ("large") learning rate
DENSE_LMBDA = 5.0                             # L2 regularization strength

# --- CNN hyperparameters (used when ARCHITECTURE == "cnn") ---
# See conv_network.py for the full architecture diagram. ReLU + softmax
# is the standard pairing for a CNN (as opposed to the dense network's
# sigmoid + cross-entropy), so this network needs its own, smaller
# learning rate -- 0.5 would overshoot badly with ReLU/softmax.
#
# CNN_CONV_BLOCKS is a tuple of (out_channels, filter_size) pairs, one
# per "scan, then zoom out" stage -- see ConvNetwork's docstring. This
# value was chosen by screening several architectures on the fast,
# unaugmented data first; see scripts/screen_architectures.py and
# docs/architecture_screening.md for that comparison.
CNN_CONV_BLOCKS = ((20, 5), (40, 5))
CNN_HIDDEN_UNITS = 100
CNN_EPOCHS = 15
CNN_MINI_BATCH_SIZE = 50
CNN_ETA = 0.2
CNN_LMBDA = 0.1

# --- Shared training-loop settings (both architectures) ---
ETA_DECREASE_FACTOR = 2.0                 # halve eta when progress stalls
NO_IMPROVEMENT_LIMIT = 5                  # epochs of no improvement before halving eta
MIN_ETA = 0.001                           # stop once eta decays below this

# --- Data augmentation (shared) ---
# Artificially grow the training set by adding rotated copies of every
# training image (see src/augmentation.py for the full explanation).
# This trades training time for a (usually small) accuracy and
# robustness improvement, since the network sees more varied examples of
# each digit instead of just the original 50,000. A CNN epoch is already
# much more expensive than a dense-network epoch, and the augmented set
# is 11x bigger, so CNN + augmentation together cost real time (several
# hours) -- set explicitly below rather than derived, so that cost is a
# deliberate choice each time, not a side effect of the architecture flag.
AUGMENT_WITH_ROTATIONS = True
ROTATION_MAX_ANGLE = 15.0                 # degrees, in each direction
ROTATION_COPIES_PER_IMAGE = 10            # extra rotated copies per original image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mnist.pkl.gz")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmarks")


def load_and_augment_data():
    print("Loading MNIST data...")
    training_data, validation_data, test_data = load_data_wrapper(DATA_PATH)
    print(
        f"  training examples:   {len(training_data)}\n"
        f"  validation examples: {len(validation_data)}\n"
        f"  test examples:       {len(test_data)}"
    )

    original_training_size = len(training_data)
    if AUGMENT_WITH_ROTATIONS:
        print(
            f"\nExpanding training data with rotated copies "
            f"(+/-{ROTATION_MAX_ANGLE:.0f} degrees, "
            f"{ROTATION_COPIES_PER_IMAGE} copies per image)..."
        )
        start_augment = time.time()
        training_data = expand_training_data(
            training_data,
            max_angle=ROTATION_MAX_ANGLE,
            copies_per_image=ROTATION_COPIES_PER_IMAGE,
        )
        print(
            f"  done in {time.time() - start_augment:.1f}s -- "
            f"training set grew from {original_training_size} to {len(training_data)} examples"
        )

    return training_data, validation_data, test_data, original_training_size


def train_dense(training_data, validation_data, test_data, original_training_size):
    epochs = DENSE_EPOCHS if AUGMENT_WITH_ROTATIONS else 60

    print(f"\nBuilding dense network with layer sizes {DENSE_SIZES}...")
    net = Network(DENSE_SIZES)

    print("\nTraining...\n")
    start = time.time()
    log = net.SGD(
        training_data,
        epochs=epochs,
        mini_batch_size=DENSE_MINI_BATCH_SIZE,
        eta=DENSE_ETA,
        lmbda=DENSE_LMBDA,
        eta_decrease_factor=ETA_DECREASE_FACTOR,
        no_improvement_limit=NO_IMPROVEMENT_LIMIT,
        min_eta=MIN_ETA,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
    )
    total_time = time.time() - start

    print("\nEvaluating on the held-out test set (used only once, at the very end)...")
    test_correct = net.accuracy(test_data)
    test_total = len(test_data)
    test_accuracy_pct = 100.0 * test_correct / test_total
    val_accuracy_pct = (
        100.0 * log["evaluation_accuracy"][-1] / len(validation_data)
        if log["evaluation_accuracy"]
        else None
    )
    print(f"\nFinal test accuracy: {test_correct}/{test_total} ({test_accuracy_pct:.2f}%)")

    weights_path = os.path.join(MODEL_DIR, "trained_network.json")
    net.save(weights_path)
    print(f"\nSaved trained weights to {weights_path}")

    benchmark = {
        "hyperparameters": {
            "architecture": "dense",
            "sizes": DENSE_SIZES,
            "epochs_run": len(log["eta"]),
            "epochs_configured": epochs,
            "mini_batch_size": DENSE_MINI_BATCH_SIZE,
            "initial_eta": DENSE_ETA,
            "final_eta": log["eta"][-1] if log["eta"] else DENSE_ETA,
            "lmbda": DENSE_LMBDA,
            "cost": "cross-entropy",
            "activation": "sigmoid",
        },
        "dataset": {
            "original_training_examples": original_training_size,
            "training_examples": len(training_data),
            "validation_examples": len(validation_data),
            "test_examples": len(test_data),
            "augmentation": (
                {
                    "technique": "rotation",
                    "max_angle_degrees": ROTATION_MAX_ANGLE,
                    "copies_per_image": ROTATION_COPIES_PER_IMAGE,
                }
                if AUGMENT_WITH_ROTATIONS
                else None
            ),
        },
        "results": {
            "final_validation_accuracy_pct": val_accuracy_pct,
            "final_test_accuracy_pct": test_accuracy_pct,
            "test_correct": test_correct,
            "test_total": test_total,
            "total_training_time_seconds": total_time,
        },
        "per_epoch_log": log,
    }
    benchmark_path = os.path.join(BENCHMARK_DIR, "benchmark.json")
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Saved benchmark report to {benchmark_path}")


def _describe_conv_blocks(conv_blocks):
    stages = " -> ".join(f"Conv2D({c} filters, {f}x{f})->ReLU->Pool" for c, f in conv_blocks)
    return stages


def train_cnn(training_data, validation_data, test_data, original_training_size):
    print(
        f"\nBuilding CNN: {_describe_conv_blocks(CNN_CONV_BLOCKS)} "
        f"-> Dense({CNN_HIDDEN_UNITS}) -> ReLU -> Dense(10) -> softmax..."
    )
    net = ConvNetwork(
        conv_blocks=CNN_CONV_BLOCKS,
        hidden_units=CNN_HIDDEN_UNITS,
    )

    print("Converting data into the CNN's (1, 28, 28) image format...")
    conv_training_data = prepare_conv_data(training_data, one_hot_labels=True)
    conv_validation_data = prepare_conv_data(validation_data, one_hot_labels=False)
    conv_test_data = prepare_conv_data(test_data, one_hot_labels=False)

    print("\nTraining...\n")
    start = time.time()
    log = net.SGD(
        conv_training_data,
        epochs=CNN_EPOCHS,
        mini_batch_size=CNN_MINI_BATCH_SIZE,
        eta=CNN_ETA,
        lmbda=CNN_LMBDA,
        eta_decrease_factor=ETA_DECREASE_FACTOR,
        no_improvement_limit=NO_IMPROVEMENT_LIMIT,
        min_eta=MIN_ETA,
        evaluation_data=conv_validation_data,
        monitor_evaluation_accuracy=True,
    )
    total_time = time.time() - start

    print("\nEvaluating on the held-out test set (used only once, at the very end)...")
    test_correct = net.accuracy(conv_test_data)
    test_total = len(conv_test_data)
    test_accuracy_pct = 100.0 * test_correct / test_total
    val_accuracy_pct = (
        100.0 * log["evaluation_accuracy"][-1] / len(conv_validation_data)
        if log["evaluation_accuracy"]
        else None
    )
    print(f"\nFinal test accuracy: {test_correct}/{test_total} ({test_accuracy_pct:.2f}%)")

    weights_path = os.path.join(MODEL_DIR, "trained_cnn.json")
    net.save(weights_path)
    print(f"\nSaved trained weights to {weights_path}")

    benchmark = {
        "hyperparameters": {
            "architecture": "cnn",
            "conv_blocks": list(CNN_CONV_BLOCKS),
            "hidden_units": CNN_HIDDEN_UNITS,
            "epochs_run": len(log["eta"]),
            "epochs_configured": CNN_EPOCHS,
            "mini_batch_size": CNN_MINI_BATCH_SIZE,
            "initial_eta": CNN_ETA,
            "final_eta": log["eta"][-1] if log["eta"] else CNN_ETA,
            "lmbda": CNN_LMBDA,
            "cost": "softmax cross-entropy",
            "activation": "ReLU (hidden layers), softmax (output)",
        },
        "dataset": {
            "original_training_examples": original_training_size,
            "training_examples": len(training_data),
            "validation_examples": len(validation_data),
            "test_examples": len(test_data),
            "augmentation": (
                {
                    "technique": "rotation",
                    "max_angle_degrees": ROTATION_MAX_ANGLE,
                    "copies_per_image": ROTATION_COPIES_PER_IMAGE,
                }
                if AUGMENT_WITH_ROTATIONS
                else None
            ),
        },
        "results": {
            "final_validation_accuracy_pct": val_accuracy_pct,
            "final_test_accuracy_pct": test_accuracy_pct,
            "test_correct": test_correct,
            "test_total": test_total,
            "total_training_time_seconds": total_time,
        },
        "per_epoch_log": log,
    }
    benchmark_path = os.path.join(BENCHMARK_DIR, "benchmark_cnn.json")
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Saved benchmark report to {benchmark_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    training_data, validation_data, test_data, original_training_size = load_and_augment_data()

    if ARCHITECTURE == "dense":
        train_dense(training_data, validation_data, test_data, original_training_size)
    elif ARCHITECTURE == "cnn":
        train_cnn(training_data, validation_data, test_data, original_training_size)
    else:
        raise ValueError(f"Unknown ARCHITECTURE: {ARCHITECTURE!r} (expected 'dense' or 'cnn')")


if __name__ == "__main__":
    main()
