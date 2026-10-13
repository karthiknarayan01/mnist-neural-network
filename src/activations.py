"""
activations.py
===============

An "activation function" is the little piece of math each neuron runs its
input through before passing a number along to the next layer. Without an
activation function, a neural network could only ever learn straight-line
(linear) patterns -- stacking layers would be pointless, because stacking
straight lines just gives you another straight line. Activation functions
bend that line, which is what lets the network learn the curvy, intricate
patterns needed to tell a "3" apart from an "8".

This project uses the **sigmoid** neuron, the classic choice for a "first
neural network" and the one used throughout the book this project is
based on. A sigmoid neuron squashes any input number -- no matter how
large or small -- into an output between 0 and 1. That makes its output
easy to interpret as "how confident is this neuron that its pattern is
present?", with 0 meaning "not at all" and 1 meaning "completely".
"""

import numpy as np


def sigmoid(z):
    """
    The sigmoid ("S-shaped") function: squashes any real number into the
    open interval (0, 1).

    - Very negative numbers come out close to 0.
    - Very positive numbers come out close to 1.
    - Zero comes out as exactly 0.5.

    ``z`` can be a single number, or (much more commonly in this project)
    a whole NumPy array of numbers -- a grid of however many neuron
    inputs we're computing at once. When you hand NumPy an array, it
    applies the formula to *every entry independently* and hands back a
    new array of the same shape, all in one fast operation. This is
    called "vectorization", and it's the main reason NumPy code can
    process a whole batch of 50 images' worth of neurons without writing
    a single explicit loop.

    ``np.exp(x)`` computes Euler's number ``e`` (~2.718) raised to the
    power of ``x``, for every entry in ``x`` at once.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    The derivative (slope) of the sigmoid function at point ``z``.

    During training, the network needs to know not just "what did this
    neuron output?" but "how sensitive is that output to small changes in
    its input?" -- that sensitivity is exactly what a derivative measures,
    and it's a key ingredient of the backpropagation algorithm (see
    ``network.py``), which uses it to work out how much each weight and
    bias contributed to a mistake.

    Conveniently, the sigmoid function's slope can be written in terms of
    the sigmoid function itself: ``sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))``.
    """
    s = sigmoid(z)
    return s * (1 - s)
