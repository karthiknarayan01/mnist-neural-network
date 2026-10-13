"""
conv_network.py
================

``ConvNetwork``: a convolutional neural network (CNN) for recognizing
MNIST digits, built by chaining together the layers defined in
``conv_layers.py``.

This is the second of two network implementations in this project.
``network.py``'s ``Network`` class is a plain, fully-connected sigmoid
network; this file's ``ConvNetwork`` is a convolutional network instead.
They're deliberately kept as two separate, independent classes rather
than one class with a mode switch, so each can be read on its own terms
-- see ``network.py``'s module docstring and this project's README for
the plain-language explanation of why a CNN tends to do better on images.

The two classes are still built to feel as similar as possible to use:
both expose an ``.SGD(...)`` method with a matching signature, an
``.accuracy(data)`` method, and ``.save(path)`` / ``.load(path)`` class
methods, so ``scripts/train.py`` can switch between them with a single
feature flag (``ARCHITECTURE = "dense"`` or ``"cnn"``) without the rest
of the training script needing to know which one it's using.

This network's architecture
------------------------------
::

    28x28x1 image
        -> Conv2D(1 -> 20 filters, 5x5)     => 24x24x20
        -> ReLU
        -> MaxPool2D(2x2)                   => 12x12x20
        -> Flatten                          => 2880
        -> Dense(2880 -> 100)
        -> ReLU
        -> Dense(100 -> 10)                 => 10 raw scores ("logits")
        -> softmax + cross-entropy cost

One round of "scan for patterns with 20 flashlights, then zoom out",
followed by the same kind of fully-connected "voting" layer the plain
network uses -- just handed a much more useful starting point (20
pattern maps) instead of raw pixels.
"""

import json

import numpy as np

from conv_layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, softmax_cross_entropy, softmax


def prepare_conv_data(data, one_hot_labels):
    """
    Adapt data from ``mnist_loader`` (built for the flat, 784-pixel-column
    world of ``Network``) into the shape ``ConvNetwork`` expects: each
    image as a ``(1, 28, 28)`` single-channel grid instead of a flat
    ``(784, 1)`` column, and (for the training set) each one-hot label
    reshaped from a ``(10, 1)`` column into a plain ``(10,)`` vector.

    Parameters
    ----------
    data : list of (x, y) pairs
        As produced by ``mnist_loader.load_data_wrapper``.
    one_hot_labels : bool
        True for the training set (labels are already one-hot columns);
        False for validation/test data (labels are plain integers, left
        untouched here since ``ConvNetwork.accuracy`` compares those
        directly against its own predictions).
    """
    prepared = []
    for x, y in data:
        image = x.reshape(1, 28, 28).astype(np.float32)
        label = y.reshape(10).astype(np.float32) if one_hot_labels else y
        prepared.append((image, label))
    return prepared


class ConvNetwork:
    """
    A convolutional neural network for classifying 28x28 MNIST digit
    images into one of 10 classes.

    Parameters
    ----------
    num_filters : int
        How many convolutional filters the first (and only) conv layer
        learns.
    filter_size : int
        The width/height of each filter, in pixels.
    hidden_units : int
        How many neurons the fully-connected hidden layer has, after the
        convolution and pooling stages.
    """

    def __init__(self, num_filters=20, filter_size=5, hidden_units=100):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_units = hidden_units

        pooled_size = (28 - filter_size + 1) // 2  # after conv (valid) + 2x2 pool
        flattened_size = num_filters * pooled_size * pooled_size

        # The network is just a plain Python list of layers, each one
        # responsible for its own forward/backward math (see
        # conv_layers.py). Running the whole network forward or backward
        # is simply looping over this list, forwards or in reverse.
        self.layers = [
            Conv2D(in_channels=1, out_channels=num_filters, filter_size=filter_size),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(in_features=flattened_size, out_features=hidden_units),
            ReLU(),
            Dense(in_features=hidden_units, out_features=10),
        ]

    def forward(self, x_batch):
        """
        Run a batch of images through every layer and return the raw,
        un-normalized output scores ("logits") for each of the 10
        digits. Call ``conv_layers.softmax`` on the result to turn these
        into actual probabilities.
        """
        out = x_batch
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_logits):
        """
        Walk backward through every layer in reverse order, exactly like
        ``Network.backprop`` walks backward through the fully-connected
        network's layers -- starting from how wrong the final output
        was, and working out each layer's share of the blame in turn.
        """
        grad = grad_logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, x_batch):
        """Run a batch of images through the network and return actual
        probabilities (summing to 1 per image), using softmax."""
        return softmax(self.forward(x_batch))

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=0.0,
        eta_decrease_factor=2.0,
        no_improvement_limit=5,
        min_eta=0.0001,
        evaluation_data=None,
        monitor_evaluation_accuracy=True,
        monitor_training_accuracy=False,
    ):
        """
        Train with mini-batch SGD. This mirrors ``Network.SGD`` closely
        on purpose -- same mini-batching, same L2 regularization, same
        "shrink the learning rate once progress stalls" schedule -- so
        reading the two side by side highlights what's genuinely
        different between the two architectures (the layers themselves)
        versus what's just shared training machinery.

        The one real difference: where ``Network.update_mini_batch``
        loops over a mini-batch one image at a time, this version builds
        the whole mini-batch into a single 4D array and runs it through
        the network in one batched forward/backward pass (see the module
        docstring in ``conv_layers.py`` for why that matters for a CNN).
        """
        import random
        import time

        n = len(training_data)
        best_accuracy = 0
        epochs_since_improvement = 0
        log = {"evaluation_accuracy": [], "training_accuracy": [], "eta": []}

        for epoch in range(epochs):
            start_time = time.time()
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, eta, lmbda, n)

            epoch_time = time.time() - start_time

            training_accuracy = None
            if monitor_training_accuracy:
                training_accuracy = self.accuracy(training_data, convert=True)
                log["training_accuracy"].append(training_accuracy)

            evaluation_accuracy = None
            if evaluation_data is not None and monitor_evaluation_accuracy:
                evaluation_accuracy = self.accuracy(evaluation_data)
                log["evaluation_accuracy"].append(evaluation_accuracy)

            log["eta"].append(eta)

            n_eval = len(evaluation_data) if evaluation_data is not None else n
            print(
                f"Epoch {epoch + 1:3d}/{epochs} "
                f"- eta={eta:.5f} "
                + (
                    f"- train_acc={training_accuracy / n * 100:6.2f}% "
                    if training_accuracy is not None
                    else ""
                )
                + (
                    f"- val_acc={evaluation_accuracy / n_eval * 100:6.2f}% "
                    if evaluation_accuracy is not None
                    else ""
                )
                + f"- {epoch_time:.1f}s"
            )

            if evaluation_data is not None and monitor_evaluation_accuracy:
                if evaluation_accuracy > best_accuracy:
                    best_accuracy = evaluation_accuracy
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= no_improvement_limit:
                    eta /= eta_decrease_factor
                    epochs_since_improvement = 0
                    print(
                        f"  -> no improvement for {no_improvement_limit} epochs, "
                        f"reducing eta to {eta:.5f}"
                    )
                    if eta < min_eta:
                        print("  -> eta below minimum threshold, stopping early.")
                        break

        return log

    def _update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        One gradient-descent update using a single mini-batch, built as
        one batched array instead of ``Network.update_mini_batch``'s
        per-image Python loop (see the module docstring for why).
        """
        x_batch = np.stack([x for x, _ in mini_batch])          # (batch, 1, 28, 28)
        y_batch = np.stack([y for _, y in mini_batch])           # (batch, 10) one-hot

        logits = self.forward(x_batch)
        _, grad_logits, _ = softmax_cross_entropy(logits, y_batch)
        self.backward(grad_logits)

        # Same L2-regularized update rule as Network.update_mini_batch:
        # shrink every weight slightly toward zero, then step it
        # opposite its gradient (already computed into layer.dW/layer.db
        # by the backward() call above). Biases are left out of the
        # shrinkage, same as in the dense network.
        for layer in self.layers:
            if layer.has_params:
                layer.W = (1 - eta * (lmbda / n)) * layer.W - eta * layer.dW
                layer.b = layer.b - eta * layer.db

    def accuracy(self, data, convert=False, batch_size=500):
        """
        Count how many examples in ``data`` the network currently
        classifies correctly, processing it in chunks (``batch_size`` at
        a time) so evaluating on a large dataset doesn't require holding
        every image's full forward pass in memory at once.

        Parameters
        ----------
        convert : bool
            Set True when ``data``'s labels are one-hot vectors (as in
            the training set) rather than plain integers (as in the
            validation/test sets) -- same meaning as in
            ``Network.accuracy``.
        """
        correct = 0
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            x_batch = np.stack([x for x, _ in chunk])
            probs = self.predict_proba(x_batch)
            predictions = np.argmax(probs, axis=1)
            if convert:
                truths = np.array([np.argmax(y) for _, y in chunk])
            else:
                truths = np.array([y for _, y in chunk])
            correct += int(np.sum(predictions == truths))
        return correct

    def save(self, json_path):
        """
        Save every layer's learnable numbers to a human-readable JSON
        file, in the same spirit as ``Network.save`` -- this is what the
        TypeScript web app loads to run this exact trained network in
        the browser.
        """
        data = {
            "architecture": "cnn",
            "num_filters": self.num_filters,
            "filter_size": self.filter_size,
            "hidden_units": self.hidden_units,
            "layers": [],
        }
        for layer in self.layers:
            if layer.has_params:
                data["layers"].append(
                    {"type": type(layer).__name__, "W": layer.W.tolist(), "b": layer.b.tolist()}
                )
            else:
                data["layers"].append({"type": type(layer).__name__})
        with open(json_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(json_path):
        """Rebuild a ``ConvNetwork`` from a JSON file written by ``save``."""
        with open(json_path, "r") as f:
            data = json.load(f)
        net = ConvNetwork(
            num_filters=data["num_filters"],
            filter_size=data["filter_size"],
            hidden_units=data["hidden_units"],
        )
        param_layers = [layer for layer in net.layers if layer.has_params]
        saved_param_layers = [entry for entry in data["layers"] if "W" in entry]
        for layer, saved in zip(param_layers, saved_param_layers):
            layer.W = np.array(saved["W"])
            layer.b = np.array(saved["b"])
        return net
