"""
train.py
========

The command-line entry point for this project: loads the MNIST data,
builds the network, trains it, prints a benchmark summary, and saves the
trained weights and biases so they can be reused later -- either back in
Python (``model/trained_network.json`` via ``Network.load``) or ported
into the TypeScript web app (same JSON file, loaded there too).

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
from augmentation import expand_training_data  # noqa: E402

# --- Hyperparameters for this project ---
# These choices are explained in depth in the project README; in short,
# they follow the "improved" configuration from chapter 3 of Michael
# Nielsen's "Neural Networks and Deep Learning", targeting 98%+ test
# accuracy on MNIST with a plain, from-scratch sigmoid network.
SIZES = [784, 100, 100, 100, 10]          # 3 hidden layers of 100 neurons each
MINI_BATCH_SIZE = 50                      # mini-batch size for SGD
ETA = 0.5                                 # starting ("large") learning rate
LMBDA = 5.0                               # L2 regularization strength
ETA_DECREASE_FACTOR = 2.0                 # halve eta when progress stalls
NO_IMPROVEMENT_LIMIT = 5                  # epochs of no improvement before halving eta
MIN_ETA = 0.001                           # stop once eta decays below this

# --- Data augmentation ---
# Artificially grow the training set by adding rotated copies of every
# training image (see src/augmentation.py for the full explanation).
# This trades training time for a (usually small) accuracy and
# robustness improvement, since the network sees more varied examples of
# each digit instead of just the original 50,000.
AUGMENT_WITH_ROTATIONS = True
ROTATION_MAX_ANGLE = 15.0                 # degrees, in each direction
ROTATION_COPIES_PER_IMAGE = 10            # extra rotated copies per original image

# Augmented training runs see ~11x more examples per epoch (and take
# roughly ~11x longer per epoch as a result), so this project uses a
# correspondingly smaller epoch cap to keep total training time
# reasonable -- the automatic learning-rate decay schedule below still
# lets training stop early once progress plateaus.
EPOCHS = 30 if AUGMENT_WITH_ROTATIONS else 60

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mnist.pkl.gz")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "benchmarks")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

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

    print(f"\nBuilding network with layer sizes {SIZES}...")
    net = Network(SIZES)

    print("\nTraining...\n")
    start = time.time()
    log = net.SGD(
        training_data,
        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        eta=ETA,
        lmbda=LMBDA,
        eta_decrease_factor=ETA_DECREASE_FACTOR,
        no_improvement_limit=NO_IMPROVEMENT_LIMIT,
        min_eta=MIN_ETA,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=False,
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

    print(
        f"\nFinal test accuracy: {test_correct}/{test_total} "
        f"({test_accuracy_pct:.2f}%)"
    )

    # Save the trained weights/biases for reuse (by Python or by the
    # TypeScript web app).
    weights_path = os.path.join(MODEL_DIR, "trained_network.json")
    net.save(weights_path)
    print(f"\nSaved trained weights to {weights_path}")

    # Save a benchmark report alongside the model, so the exact numbers
    # behind the README's claims are reproducible and auditable later.
    benchmark = {
        "hyperparameters": {
            "sizes": SIZES,
            "epochs_run": len(log["eta"]),
            "epochs_configured": EPOCHS,
            "mini_batch_size": MINI_BATCH_SIZE,
            "initial_eta": ETA,
            "final_eta": log["eta"][-1] if log["eta"] else ETA,
            "lmbda": LMBDA,
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


if __name__ == "__main__":
    main()
