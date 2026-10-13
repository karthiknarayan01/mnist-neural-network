"""
mnist_loader.py
================

This file is only concerned with one job: turning the raw MNIST data file
into a shape that is convenient for the neural network to learn from.

What is MNIST?
---------------
MNIST is a famous dataset of 70,000 small, grayscale pictures of
handwritten digits (0 through 9), each one 28x28 pixels. It is split into:

- 50,000 **training** images  -- the network studies these and adjusts
  itself based on them.
- 10,000 **validation** images -- used only to *check* how the network is
  doing during training, so we can decide things like when to slow down
  the learning rate. The network never learns from these directly.
- 10,000 **test** images -- used only once, at the very end, to report a
  final, unbiased accuracy score.

The raw file (``data/mnist.pkl.gz``) stores each image as a flat list of
784 numbers (28 x 28 = 784 pixel brightness values, each between 0 and 1,
where 0 is black and 1 is white) and a single integer label (0-9) saying
which digit the image shows.
"""

import gzip
import pickle

import numpy as np


def load_data(path="data/mnist.pkl.gz"):
    """
    Open the MNIST data file and return the three raw data sets.

    Returns a tuple ``(training_data, validation_data, test_data)``.
    Each of those three items is itself a tuple ``(images, labels)`` where:

    - ``images`` is a NumPy array with one row per picture and 784 columns
      (one column per pixel).
    - ``labels`` is a NumPy array with one number per picture: the digit
      (0-9) that picture actually shows.

    This "raw" format is compact, but it is not quite what our network
    wants to train on, so :func:`load_data_wrapper` below reshapes it.
    """
    # "gzip.open" lets us read a compressed (zipped) file without manually
    # unzipping it first -- Python decompresses it on the fly as we read.
    with gzip.open(path, "rb") as f:
        # The file was originally saved with Python 2, so we tell pickle
        # (Python's tool for saving/loading arbitrary objects) to read it
        # using the "latin1" text encoding. Without this, loading would
        # crash under modern Python 3.
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


def vectorized_result(digit):
    """
    Turn a single digit label (e.g. the number 3) into a "one-hot" vector:
    a column of 10 numbers that are all 0 except for a single 1 in the
    position of the correct digit.

    Example: the digit 3 becomes::

        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   (as a 10x1 column)

    Why do this? Our network's final layer has 10 output neurons, one per
    possible digit. During training we need something of the same shape
    (10 numbers) to compare the network's guess against, so we can measure
    how wrong it was. A one-hot vector is that "perfect answer key".
    """
    # np.zeros((10, 1)) creates a column of 10 zeros -- a 10-row,
    # 1-column grid of numbers, all starting at 0.0.
    one_hot = np.zeros((10, 1))
    one_hot[digit] = 1.0
    return one_hot


def load_data_wrapper(path="data/mnist.pkl.gz"):
    """
    Load MNIST and reshape it into the exact format our ``Network`` class
    expects to train and evaluate on.

    Returns ``(training_data, validation_data, test_data)`` where each one
    is a Python list of ``(x, y)`` pairs:

    - ``x`` is a column of 784 numbers: the picture, flattened out of its
      28x28 grid into one long strip of pixel brightness values.
    - ``y`` is the "correct answer":
        * for the training set, ``y`` is the one-hot vector described in
          :func:`vectorized_result` above, because that is the shape the
          cost function needs to compare against the network's output.
        * for the validation and test sets, ``y`` is kept as a plain
          integer (0-9), because for those sets we only ever need to ask
          "did the network's top guess match the true digit?", and a
          plain integer is simpler to compare against.
    """
    training_data, validation_data, test_data = load_data(path)

    # "reshape(784, 1)" takes a flat row of 784 pixel values and stands it
    # up into a column (784 rows, 1 column). Our network always works
    # with columns of numbers, so every image is reshaped this way.
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    # "zip" pairs up the two lists item-by-item: image 1 with its answer,
    # image 2 with its answer, and so on.
    training = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test = list(zip(test_inputs, test_data[1]))

    return training, validation, test
