"""
screen_architectures.py
========================

A cheap, fast comparison of several CNN architecture/hyperparameter
candidates, run on the plain (unaugmented) 50,000-image training set so
each candidate only takes minutes rather than hours. The winner here is
what `scripts/train.py`'s `CNN_CONV_BLOCKS` gets set to for the real,
full (augmented, many-epoch) training run -- this script's job is only
to find a promising direction cheaply, not to produce a final model.

Every candidate trains for the same small, fixed number of epochs, so
the comparison is apples-to-apples: whichever curve is highest after
the same amount of work is the one worth spending hours of augmented
training on.

Usage:
    source .venv/bin/activate
    python3 scripts/screen_architectures.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mnist_loader import load_data_wrapper  # noqa: E402
from conv_network import ConvNetwork, prepare_conv_data  # noqa: E402

SCREEN_EPOCHS = 8
MINI_BATCH_SIZE = 50

#: name -> (conv_blocks, hidden_units, eta, lmbda)
CANDIDATES = {
    "baseline (current production config)": (((20, 5),), 100, 0.1, 0.1),
    "wider (40 filters, still 1 conv layer)": (((40, 5),), 100, 0.1, 0.1),
    "deeper (2 conv layers, 20->40)": (((20, 5), (40, 5)), 100, 0.1, 0.1),
    "deeper + wider (2 conv layers, 32->64, bigger hidden)": (((32, 5), (64, 5)), 128, 0.1, 0.1),
    "deeper, higher eta": (((20, 5), (40, 5)), 100, 0.2, 0.1),
    "deeper, lighter regularization": (((20, 5), (40, 5)), 100, 0.1, 0.02),
}

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mnist.pkl.gz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "architecture_screening.json")


def main():
    print("Loading MNIST data...")
    training_data, validation_data, _test_data = load_data_wrapper(DATA_PATH)
    conv_training_data = prepare_conv_data(training_data, one_hot_labels=True)
    conv_validation_data = prepare_conv_data(validation_data, one_hot_labels=False)
    print(f"  {len(conv_training_data)} training, {len(conv_validation_data)} validation\n")

    results = {}
    for name, (conv_blocks, hidden_units, eta, lmbda) in CANDIDATES.items():
        print(f"=== {name} ===")
        print(f"    conv_blocks={conv_blocks}  hidden_units={hidden_units}  eta={eta}  lmbda={lmbda}")
        net = ConvNetwork(conv_blocks=conv_blocks, hidden_units=hidden_units)

        start = time.time()
        log = net.SGD(
            list(conv_training_data),  # SGD shuffles in place; give each candidate a fresh copy
            epochs=SCREEN_EPOCHS,
            mini_batch_size=MINI_BATCH_SIZE,
            eta=eta,
            lmbda=lmbda,
            evaluation_data=conv_validation_data,
            monitor_evaluation_accuracy=True,
        )
        elapsed = time.time() - start
        final_val_acc = 100.0 * log["evaluation_accuracy"][-1] / len(conv_validation_data)
        best_val_acc = 100.0 * max(log["evaluation_accuracy"]) / len(conv_validation_data)
        per_epoch = elapsed / len(log["eta"])

        print(
            f"    -> {elapsed:.0f}s total, {per_epoch:.1f}s/epoch, "
            f"final val_acc={final_val_acc:.2f}%, best val_acc={best_val_acc:.2f}%\n"
        )
        results[name] = {
            "conv_blocks": list(conv_blocks),
            "hidden_units": hidden_units,
            "eta": eta,
            "lmbda": lmbda,
            "epochs_run": len(log["eta"]),
            "seconds_per_epoch": round(per_epoch, 1),
            "total_seconds": round(elapsed, 1),
            "final_val_accuracy_pct": round(final_val_acc, 2),
            "best_val_accuracy_pct": round(best_val_acc, 2),
            "evaluation_accuracy": log["evaluation_accuracy"],
        }

    print("\n=== Summary (sorted by best validation accuracy) ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["best_val_accuracy_pct"], reverse=True)
    for name, r in ranked:
        print(
            f"  {r['best_val_accuracy_pct']:6.2f}%  "
            f"({r['seconds_per_epoch']:5.1f}s/epoch)  {name}"
        )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"screen_epochs": SCREEN_EPOCHS, "candidates": results}, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
