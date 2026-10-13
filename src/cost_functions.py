"""
cost_functions.py
==================

A "cost function" (also called a "loss function") is a single number that
answers the question: **"how wrong was the network's guess?"** Training a
neural network is, at its heart, just a search for the weights and biases
that make this number as small as possible across the whole training set.

This project uses the **cross-entropy cost** function instead of the
simpler "mean squared error" used in introductory examples. The reason is
a subtle but important one, explained below.

Why cross-entropy instead of plain squared error?
---------------------------------------------------
A sigmoid neuron that is *very* wrong (e.g. it outputs 0.01 when the
answer should be 1) sits on the flat, nearly-horizontal part of the
sigmoid curve. On the flat part of that curve, the slope
(``sigmoid_prime``, see ``activations.py``) is tiny. With plain squared
error, the size of each learning step is proportional to that slope --
so exactly when the network is most wrong, and most needs a big
correction, squared error gives it the smallest, slowest nudge. This is
called the "learning slowdown" problem.

Cross-entropy cost is defined so that this slope term cancels out
mathematically. The result is a learning step whose size depends only on
how wrong the guess was (``output - desired_output``), with no small-slope
penalty. In practice this means a network trained with cross-entropy
starts correcting its biggest mistakes immediately, instead of limping
along slowly before it "wakes up" -- which is part of why this project
reaches high accuracy faster than a squared-error version would.
"""

import numpy as np


class CrossEntropyCost:
    """
    Implements the cross-entropy cost function, paired with the sigmoid
    output layer used throughout this project.

    This class has no state of its own (no ``__init__``) -- it's simply a
    convenient place to group the two related pieces of math together as
    "static methods" (functions that belong to the class but don't need
    an instance of it to be called).
    """

    @staticmethod
    def fn(a, y):
        """
        Compute the cross-entropy cost for one training example, given:

        - ``a``: the network's actual output (a column of 10 numbers,
          each between 0 and 1 -- one per possible digit).
        - ``y``: the desired output (the one-hot "answer key" column from
          ``mnist_loader.vectorized_result``).

        The formula is::

            cost = -sum( y * ln(a) + (1-y) * ln(1-a) )

        Intuitively: for the one output neuron where ``y`` is 1 (the
        correct digit), this penalizes the network for any output less
        than 1. For the nine output neurons where ``y`` is 0, it
        penalizes the network for any output greater than 0. The further
        off the output is, the larger (worse) the cost.

        ``np.nan_to_num`` is a small safety net: if ``a`` ever becomes
        exactly 0 or exactly 1 due to rounding, ``ln(0)`` is negative
        infinity, which would otherwise poison the calculation. This
        function quietly converts any resulting "not a number" or
        infinite values back into large-but-finite numbers instead.
        """
        # np.log computes the natural logarithm (base e) of every entry
        # in an array at once. np.nan_to_num cleans up any 0 * -inf = nan
        # or ln(0) = -inf edge cases that can appear from floating-point
        # rounding when "a" is extremely close to 0 or 1.
        return np.sum(
            np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))
        )

    @staticmethod
    def delta(z, a, y):
        """
        Compute "delta": the error signal for the output layer, which is
        the very first thing backpropagation needs in order to work
        backwards through the network and figure out how to adjust every
        weight and bias.

        For plain squared-error cost, this would be
        ``(a - y) * sigmoid_prime(z)`` -- and that ``sigmoid_prime(z)``
        term is exactly the "learning slowdown" culprit described above.

        For cross-entropy cost paired with a sigmoid output layer, the
        math works out so that the ``sigmoid_prime(z)`` term cancels
        out entirely, leaving simply::

            delta = a - y

        This is the whole payoff of choosing cross-entropy: a clean,
        slowdown-free error signal that is just "how far off was the
        guess", in the same units as the output itself. Note that ``z``
        (the output layer's input, before the sigmoid is applied) isn't
        actually used in this formula -- it's kept as a parameter only so
        this class has the exact same shape as a squared-error cost class
        would, making the two interchangeable in the training code.
        """
        return a - y
