"""
network.py
===========

This is the heart of the project: a neural network, built from scratch
using nothing but NumPy for the math. There is no machine-learning
library doing the heavy lifting behind the scenes -- every piece of
arithmetic a real neural network needs (making a guess, measuring how
wrong it was, and figuring out how to adjust itself) is written out
explicitly below, so you can read exactly what is going on.

A quick mental picture of the network
---------------------------------------
Picture the network as a series of columns of little circles ("neurons"),
with every circle in one column connected to every circle in the next
column:

    784 input neurons  ->  100 neurons  ->  100 neurons  ->  100 neurons  ->  10 output neurons
    (one per pixel)        (hidden)          (hidden)          (hidden)        (one per digit 0-9)

- The 784 input neurons simply hold the brightness of each of the image's
  784 pixels -- they don't compute anything.
- Each of the three "hidden" layers of 100 neurons looks at everything the
  previous layer produced, combines it, and passes a new, more abstract
  signal forward (early layers might learn to detect edges and curves;
  later layers combine those into whole digit shapes).
- The final 10 output neurons each represent one digit (0 through 9). The
  network's "answer" is whichever of those 10 neurons has the highest
  output.

Each connection between two neurons has an adjustable "weight" (how much
one neuron's output should influence the next neuron), and each neuron
has an adjustable "bias" (a baseline nudge, independent of its inputs).
Training a network means searching for the combination of weights and
biases that makes the network's guesses as accurate as possible. There
are tens of thousands of these numbers in this network, and no human
designs them by hand -- they start out random, and training gradually
improves them via an algorithm called **backpropagation**, described
further down in this file.
"""

import json
import random
import time

import numpy as np

from activations import sigmoid, sigmoid_prime
from cost_functions import CrossEntropyCost
from mnist_loader import vectorized_result


class Network:
    """
    A fully-connected, feed-forward neural network of sigmoid neurons,
    trained with mini-batch stochastic gradient descent, cross-entropy
    cost, and L2 regularization.

    Parameters
    ----------
    sizes : list of int
        The number of neurons in each layer, from input to output. For
        this project, ``sizes = [784, 100, 100, 100, 10]``: a 784-pixel
        input, three hidden layers of 100 neurons each, and 10 output
        neurons (one per digit).
    cost : class, optional
        The cost function class to use (see ``cost_functions.py``).
        Defaults to ``CrossEntropyCost``.
    """

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initializer()

    def default_weight_initializer(self):
        """
        Give every weight and bias in the network a small random starting
        value, using a scheme that is known to help sigmoid networks
        learn faster (compared to plain, unscaled random numbers).

        Biases: one per neuron in every layer *except* the input layer
        (input neurons just hold pixel values -- they have nothing to
        bias). Each bias starts as a random sample from the "standard
        normal distribution" -- a classic bell curve centered on 0, where
        most samples land between roughly -2 and 2.

        ``np.random.randn(y, 1)`` draws ``y`` random numbers from that
        bell curve and arranges them in a column (``y`` rows, 1 column) --
        matching the "neurons are columns of numbers" convention used
        throughout this file.

        Weights: one per connection between a layer and the layer before
        it. These are also drawn from the standard normal bell curve, but
        then *scaled down* by dividing by the square root of the number
        of neurons feeding into that layer (``np.sqrt(x)``). Why scale
        them down? If a neuron has, say, 784 incoming connections, and
        each incoming weight is a "typical" random number, then the sum
        of all 784 weighted inputs can end up enormous -- which pushes
        the sigmoid neuron's input ``z`` far into the flat, slow-to-learn
        tails of the sigmoid curve (see ``activations.py``) right from
        the very start of training, before it has even had a chance to
        learn anything. Shrinking each individual weight in proportion to
        how many of them are being summed keeps that total sum in a
        sensible range, so training starts off quickly instead of stuck
        on a flat slope.

        ``zip(sizes[:-1], sizes[1:])`` pairs up consecutive layer sizes --
        for ``sizes = [784, 100, 100, 100, 10]`` this produces the pairs
        ``(784, 100), (100, 100), (100, 100), (100, 10)``, i.e. exactly
        the four weight matrices connecting one layer to the next.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """
        Run one image through the whole network and return its final
        output: a column of 10 numbers between 0 and 1, one per digit.

        ``a`` starts out as the input image (a column of 784 pixel
        values) and is overwritten, layer by layer, with that layer's
        output -- which immediately becomes the *next* layer's input.

        For each layer, every neuron's input is a weighted sum of every
        neuron's output from the layer before, plus that neuron's own
        bias. In matrix form, this weighted sum for an *entire layer at
        once* is simply ``np.dot(w, a) + b``:

        - ``w`` is that layer's matrix of weights (one row per neuron in
          this layer, one column per neuron in the previous layer).
        - ``np.dot(w, a)`` is matrix multiplication: it computes, for
          every neuron in this layer simultaneously, the sum of
          "previous layer's output times the connecting weight" --
          exactly the weighted sum each neuron needs, for the whole
          layer, in a single fast operation instead of a manual loop.
        - adding ``b`` (that layer's column of biases) then adds each
          neuron's baseline nudge.
        - ``sigmoid(...)`` then squashes every one of those sums into the
          neuron's final 0-to-1 output, ready to feed the next layer.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

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
        Train the network using **mini-batch Stochastic Gradient Descent**
        (SGD) -- the main training loop of the whole project.

        The big idea of SGD
        ---------------------
        We want to nudge every weight and bias in a direction that makes
        the cost (how wrong the network is) a little smaller. Calculus
        tells us that direction: the *negative gradient* -- the direction
        of steepest decrease. Computing the exact gradient using every one
        of the 50,000 training images for every single nudge would be
        extremely slow. Instead, SGD estimates a "good enough" gradient
        using just a small random handful of images at a time -- a
        **mini-batch** -- and takes a small step based on that estimate.
        Over many mini-batches, these slightly-noisy-but-cheap steps add
        up to steady overall progress, much faster than computing exact
        gradients would allow.

        Parameters
        ----------
        training_data : list of (x, y) pairs
            The training images and their one-hot answer vectors.
        epochs : int
            The maximum number of times to sweep through the entire
            training set. One full sweep is called an "epoch".
        mini_batch_size : int
            How many images go into each mini-batch (50, for this
            project -- small enough to update frequently, large enough
            that each update's gradient estimate isn't too noisy).
        eta : float
            The initial "learning rate": how big a step to take in the
            gradient's direction. This project starts with a
            comparatively *large* eta so the network makes fast progress
            early on, then automatically shrinks eta as training
            progresses (see below) so that later, finer adjustments don't
            overshoot and bounce around a good solution.
        lmbda : float
            The L2 regularization strength (see ``update_mini_batch``
            below for what this does and why).
        eta_decrease_factor : float
            Whenever the learning rate is reduced, it is divided by this
            factor (so a factor of 2.0 means "cut the learning rate in
            half").
        no_improvement_limit : int
            If the accuracy measured on ``evaluation_data`` hasn't set a
            new best score for this many epochs in a row, the learning
            rate is decreased. This is the "variable learning rate"
            strategy: start fast, and automatically slow down once
            progress stalls, to allow the network to settle into a more
            precise minimum.
        min_eta : float
            Training stops early once the learning rate has been
            decreased below this value -- by that point, further epochs
            tend to buy very little extra accuracy for the time spent.
        evaluation_data : list of (x, y) pairs, optional
            Usually the *validation* set. Used after every epoch purely
            to measure progress and decide when to shrink eta -- the
            network never learns from this data directly.
        monitor_evaluation_accuracy, monitor_training_accuracy : bool
            Whether to print accuracy figures after each epoch (handy for
            watching progress, and for the benchmark numbers in the
            project's README).

        Returns
        -------
        dict
            A small log of per-epoch statistics, used later to print a
            benchmark summary and/or save a training curve.
        """
        n = len(training_data)
        best_accuracy = 0
        epochs_since_improvement = 0
        log = {"evaluation_accuracy": [], "training_accuracy": [], "eta": []}

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle the training data into a brand new random order
            # every epoch. This is the "Stochastic" in Stochastic
            # Gradient Descent: by always drawing mini-batches from a
            # freshly shuffled deck, no two epochs see the exact same
            # sequence of mini-batches, which helps the network avoid
            # getting stuck repeating the same noisy pattern of updates.
            random.shuffle(training_data)

            # Slice the shuffled training data into consecutive chunks of
            # `mini_batch_size` images each. Python's slice notation
            # `list[k : k + size]` grabs `size` items starting at
            # position `k`, so this one line produces every mini-batch
            # for the epoch in a single pass over the data.
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            epoch_time = time.time() - start_time

            # --- Measure progress and report it ---
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

            # --- The "variable learning rate" strategy ---
            # If validation accuracy improved, remember this as the new
            # best score and reset the "stalled" counter. Otherwise, note
            # that another epoch has passed without improvement.
            if evaluation_data is not None and monitor_evaluation_accuracy:
                if evaluation_accuracy > best_accuracy:
                    best_accuracy = evaluation_accuracy
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                # Once progress has stalled for long enough, shrink the
                # learning rate so training can settle into finer, more
                # careful adjustments -- and if eta has already shrunk
                # past our minimum threshold, further epochs aren't
                # worth running, so stop training early.
                if epochs_since_improvement >= no_improvement_limit:
                    eta /= eta_decrease_factor
                    epochs_since_improvement = 0
                    print(f"  -> no improvement for {no_improvement_limit} epochs, "
                          f"reducing eta to {eta:.5f}")
                    if eta < min_eta:
                        print("  -> eta below minimum threshold, stopping early.")
                        break

        return log

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Perform one gradient-descent update using a single mini-batch of
        training images.

        The steps are:

        1. For every image in the mini-batch, run backpropagation (see
           ``backprop`` below) to find out exactly how much each
           individual weight and bias contributed to that image's error.
        2. Add up those contributions across the whole mini-batch.
        3. Nudge every weight and bias a small step opposite to its
           accumulated contribution -- i.e. in the direction that would
           have reduced the error, scaled by the learning rate ``eta``
           and averaged over the mini-batch size.

        Parameters
        ----------
        mini_batch : list of (x, y) pairs
            The small handful of training images for this update.
        eta : float
            The current learning rate.
        lmbda : float
            The L2 regularization strength.
        n : int
            The total size of the full training set (needed for the
            regularization formula below, not just this mini-batch).

        L2 regularization, in plain terms
        -----------------------------------
        A network can "overfit": instead of learning the general shape of
        each digit, it can partly memorize quirks of the specific
        training images it has seen, which hurts its accuracy on new,
        unseen images. One sign of overfitting is individual weights
        growing very large, as the network contorts itself to fit
        training-set noise.

        L2 regularization discourages this by adding a tiny extra pull on
        every weight, shrinking it slightly toward zero on every single
        update (see the ``(1 - eta * lmbda / n)`` factor below) *before*
        the normal gradient-based nudge is applied. This acts like a
        standing preference for simpler, smaller weights, and the network
        only lets weights grow large when the training data gives it
        strong, consistent evidence to do so -- which tends to produce a
        network that generalizes better to images it has never seen
        before. Biases are deliberately left out of this shrinkage: only
        weights (the things more directly responsible for overfitting)
        are regularized.
        """
        # Start with "zero contribution so far" accumulators, shaped
        # exactly like the real weights and biases. `np.zeros_like(x)`
        # creates a new array of zeros with the same shape as `x`.
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Accumulate this image's contribution on top of the running
            # totals for the mini-batch.
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Apply the update: shrink each weight slightly (regularization),
        # then step it opposite the averaged accumulated gradient.
        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        **Backpropagation**: the algorithm that makes training a deep
        network practical at all.

        Given one training image ``x`` and its correct answer ``y``, this
        works out exactly how much *every single weight and bias in the
        entire network* contributed to the mistake the network made --
        efficiently, in just one forward pass and one backward pass,
        no matter how many layers the network has.

        The two passes
        ----------------
        1. **Forward pass**: run the image through the network exactly
           like ``feedforward`` does, but this time remember every
           intermediate number along the way (every layer's raw weighted
           sum ``z``, and every layer's squashed output ``activation``).
           We need those intermediate numbers for step 2.

        2. **Backward pass**: start at the output layer, where we can
           directly measure the error (the cost function's ``delta``, see
           ``cost_functions.py``). Then walk *backwards* through the
           network, one layer at a time, using each layer's error to
           compute: (a) exactly how much each weight and bias in that
           layer should be blamed for the final mistake, and (b) the
           error "one layer further back", so the same calculation can
           repeat for the previous layer. This backward chain is an
           application of the calculus chain rule, and it's dramatically
           cheaper than computing each weight's contribution separately
           from scratch.

        Returns
        -------
        (nabla_b, nabla_w) : tuple of lists of NumPy arrays
            The gradient of the cost with respect to every bias and every
            weight in the network (for this one image), in the same
            per-layer list structure as ``self.biases`` and
            ``self.weights``.
        """
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        # --- Forward pass, remembering everything ---
        activation = x
        activations = [x]  # every layer's output, starting with the input image
        zs = []  # every layer's raw weighted sum, before the sigmoid squashes it

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # --- Backward pass ---
        # Start with the output layer's error. Thanks to pairing
        # cross-entropy cost with sigmoid outputs, this is simply
        # "guess minus correct answer" (see cost_functions.CrossEntropyCost.delta).
        delta = self.cost.delta(zs[-1], activations[-1], y)

        # The bias gradient for a layer is just its error signal...
        nabla_b[-1] = delta
        # ...and the weight gradient pairs that error with whatever the
        # *previous* layer's activation was feeding into it.
        # `.transpose()` flips a column into a row (or vice versa) so the
        # shapes line up correctly for this matrix multiplication.
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Walk backwards through the remaining layers. `range(2, num_layers)`
        # with a negative index (`-l`) is a compact way of saying "the 2nd
        # layer from the end, then the 3rd from the end, and so on" --
        # i.e. walking from the last hidden layer back to the first one.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # Propagate the error one layer backward: take next layer's
            # error, "unmix" it through that layer's weights (transposed,
            # since we're now going backward through the same
            # connections), and scale by this layer's own sigmoid slope.
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        Count how many examples in ``data`` the network currently gets
        right.

        Parameters
        ----------
        data : list of (x, y) pairs
        convert : bool
            The training set stores its answers ``y`` as one-hot vectors
            (see ``mnist_loader.vectorized_result``), while the
            validation and test sets store ``y`` as plain digits. Set
            ``convert=True`` when checking accuracy on the training set,
            so both sides of the comparison below end up as plain digits.

        For every example, ``feedforward`` is run to get the network's 10
        output numbers, and ``np.argmax`` finds the *position* of the
        largest one -- i.e. which digit the network is most confident
        about. That's compared against the true digit, and the total
        count of matches is returned.
        """
        if convert:
            results = [
                (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data
            ]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(guess == truth) for (guess, truth) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Compute the average cross-entropy cost (plus the L2 regularization
        penalty) across an entire dataset -- useful for monitoring whether
        the network is still improving, beyond just its accuracy score.

        Parameters
        ----------
        convert : bool
            The cost function needs ``y`` as a one-hot vector (see
            ``cost_functions.CrossEntropyCost.fn``). The training set
            already stores ``y`` that way, but the validation and test
            sets store ``y`` as a plain digit -- set ``convert=True`` for
            those so each digit is converted to a one-hot vector first.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        # Add the L2 regularization penalty: a tiny fraction of the sum
        # of every weight squared (`np.linalg.norm(w) ** 2` computes "the
        # sum of all of w's entries, each squared" in one call).
        cost += (
            0.5
            * (lmbda / len(data))
            * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        )
        return cost

    def save(self, json_path):
        """
        Save every weight and bias to a human-readable JSON file.

        This is the bridge between the Python training code and the
        TypeScript web app: the app re-implements ``feedforward`` in
        TypeScript and loads these exact numbers to make predictions
        in the browser, without needing Python or NumPy at all.

        ``.tolist()`` converts a NumPy array into a plain nested Python
        list (NumPy arrays themselves aren't directly understood by the
        standard ``json`` module, but plain lists and numbers are).
        """
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(json_path):
        """
        Rebuild a ``Network`` from a JSON file previously written by
        ``save``, restoring its exact trained weights and biases.
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        net = Network(data["sizes"], cost=CrossEntropyCost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
