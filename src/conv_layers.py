"""
conv_layers.py
===============

The building blocks of a **convolutional neural network (CNN)**: a
different (and, for images, usually better) way of structuring a
network than the plain, fully-connected ``Network`` class in
``network.py``.

If you haven't already, read the "convolutional layer" explanation in
this project's docs before diving into this file -- it explains the
whole idea (small, reusable "flashlights" sliding across an image) in
plain language. This file is the literal, from-scratch NumPy math behind
that explanation.

Why a separate file, with its own separate classes?
------------------------------------------------------
``network.py``'s ``Network`` class is a *fully-connected* network: every
input pixel is wired directly to every neuron. It works well (98%+ test
accuracy), but it treats an image as nothing more than a flat list of
784 unrelated numbers -- it has no built-in sense that pixel (5, 5) is
right next to pixel (5, 6).

A CNN is built instead from distinct kinds of layers -- a convolutional
layer, a pooling layer, a fully-connected layer -- stacked together.
That's different enough from the single monolithic ``Network`` class
that it's clearer to give each kind of layer its own small, focused
class here, and compose them together in ``conv_network.py``, rather
than contorting the original class to do both jobs. Reading this file
side-by-side with ``network.py`` is a good way to see how the same
underlying ideas (forward pass, backpropagation, mini-batch SGD, L2
regularization) show up differently depending on the architecture.

A note on batching
---------------------
``network.py`` processes one training image at a time inside a Python
loop, then averages the results over a mini-batch. That's simple and
easy to follow, but convolution is a much more expensive operation, and
looping over images one at a time in pure Python would make training a
CNN painfully slow. Every layer in this file instead processes a whole
mini-batch of images at once, as a single 4D NumPy array of shape
``(batch_size, channels, height, width)`` -- the same trick NumPy uses
everywhere to replace slow Python loops with fast, vectorized math (see
the "vectorization" note in ``activations.py`` for the simpler,
single-number version of this same idea).
"""

import numpy as np


class Layer:
    """
    The common interface every layer in this file follows: take a batch
    of inputs and produce a batch of outputs (``forward``), and later,
    given how wrong the final answer turned out to be, work out how much
    this layer's inputs (and, if it has any, this layer's own learnable
    numbers) are to blame (``backward``).

    This is the same forward-pass-then-backward-pass idea as
    ``Network.feedforward`` and ``Network.backprop``, just split up so
    each layer only needs to know about its own small piece of the
    calculation. ``ConvNetwork`` (see ``conv_network.py``) chains layers
    together, feeding each one's output into the next.
    """

    #: Layers with learnable numbers (weights/biases) set this to True
    #: and expose ``.params()`` / ``.grads()`` so the training loop knows
    #: to update them. Layers with nothing to learn (activations,
    #: pooling, reshaping) leave it False.
    has_params = False

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class Conv2D(Layer):
    """
    A convolutional layer: slides a set of small, reusable filters
    ("flashlights") across every input image in the batch, looking for
    the same simple pattern everywhere it appears.

    Parameters
    ----------
    in_channels : int
        How many channels the input has (1 for our grayscale MNIST
        digits; a color photo would have 3).
    out_channels : int
        How many different filters to learn, i.e. how many different
        patterns this layer can look for at once.
    filter_size : int
        The filter's width and height, in pixels (this project uses 5,
        giving each filter a 5x5 window).

    Shapes
    ------
    Input:  ``(batch_size, in_channels, height, width)``
    Output: ``(batch_size, out_channels, height - filter_size + 1, width - filter_size + 1)``

    The output is smaller than the input because this is a "valid"
    convolution: a filter only ever slides to positions where it fully
    overlaps the image, so it never looks at padding that isn't really
    there. A 28x28 image with a 5x5 filter produces a 24x24 output.
    """

    has_params = True

    def __init__(self, in_channels, out_channels, filter_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        # Each filter looks at every input channel, across a
        # filter_size x filter_size window. Scaling by the square root
        # of the number of inputs each output pixel sums over is the
        # same "keep the starting numbers sensible" trick used in
        # Network.default_weight_initializer -- see that docstring for
        # the full explanation of why.
        fan_in = in_channels * filter_size * filter_size
        self.W = np.random.randn(out_channels, in_channels, filter_size, filter_size) / np.sqrt(fan_in)
        self.b = np.zeros(out_channels)

        self.dW = None
        self.db = None
        self._cached_input = None

    def forward(self, x):
        """
        Slide every filter across every image in the batch.

        Rather than building one giant table of every little image patch
        (the "im2col" trick most frameworks use), this loop takes a
        simpler path: for every possible (row, column) offset inside the
        filter window, it shifts the whole input by that offset, scales
        it by the one weight number each filter has at that offset, and
        adds the result into a running total. After looping over every
        offset in the filter (25 of them, for a 5x5 filter), the running
        total *is* the full convolution -- it's mathematically the exact
        same sum a filter computes by sliding across the image, just
        computed offset-by-offset instead of position-by-position, which
        lets each step be one large, fast, vectorized NumPy operation
        instead of a slow Python loop over every output pixel.
        """
        self._cached_input = x
        batch_size, _, in_h, in_w = x.shape
        out_h = in_h - self.filter_size + 1
        out_w = in_w - self.filter_size + 1

        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        for di in range(self.filter_size):
            for dj in range(self.filter_size):
                # Every pixel this offset contributes to, across the
                # whole batch and every input channel, in one slice.
                x_slice = x[:, :, di:di + out_h, dj:dj + out_w]
                # "ncij,oc->noij": for each batch item (n) and output
                # position (i, j), multiply this offset's input channels
                # (c) by this offset's weights and sum over channels (c)
                # to produce every output filter's (o) contribution --
                # np.einsum is NumPy's tool for expressing exactly this
                # kind of "multiply these, then sum over that dimension"
                # calculation in one vectorized call.
                out += np.einsum("ncij,oc->noij", x_slice, self.W[:, :, di, dj])

        out += self.b[None, :, None, None]
        return out

    def backward(self, grad_output):
        """
        Given how wrong the output of this layer turned out to be
        (``grad_output``, the same shape as this layer's output), work
        out:

        - ``self.dW`` / ``self.db``: how much to blame each filter
          weight and bias for that error (used to update them).
        - the returned value: how much to blame this layer's *input*,
          so the layer before this one can continue the same process
          (exactly what ``Network.backprop`` does when it walks
          backward through the fully-connected network's layers).

        This mirrors ``forward`` almost exactly, looping over the same
        filter offsets -- which makes sense, since backpropagation is
        always "run the forward calculation's steps in reverse".
        """
        x = self._cached_input
        batch_size, _, in_h, in_w = x.shape
        out_h = in_h - self.filter_size + 1
        out_w = in_w - self.filter_size + 1

        self.dW = np.zeros_like(self.W)
        self.db = grad_output.sum(axis=(0, 2, 3))
        grad_input = np.zeros_like(x)

        for di in range(self.filter_size):
            for dj in range(self.filter_size):
                x_slice = x[:, :, di:di + out_h, dj:dj + out_w]
                # How much each weight at this offset contributed to the
                # error, summed over every batch item and output pixel
                # it touched.
                self.dW[:, :, di, dj] = np.einsum("noij,ncij->oc", grad_output, x_slice)
                # How much each input pixel at this offset contributed,
                # scattered back into the right place in grad_input.
                grad_input[:, :, di:di + out_h, dj:dj + out_w] += np.einsum(
                    "noij,oc->ncij", grad_output, self.W[:, :, di, dj]
                )

        return grad_input

    def params_and_grads(self):
        return [(self.W, self.dW, "W"), (self.b, self.db, "b")]


class ReLU(Layer):
    """
    The Rectified Linear Unit activation: the simplest possible
    "bend" in the data, and the standard choice inside modern CNNs.

    Unlike the sigmoid neuron used throughout ``network.py`` (which
    squashes everything into a smooth 0-to-1 curve), ReLU does something
    almost embarrassingly simple: keep positive numbers exactly as they
    are, and replace every negative number with zero.

        ReLU(z) = max(0, z)

    Why use a different activation here? Sigmoid's curve goes nearly
    flat for very positive or very negative inputs (see
    ``activations.py``'s "learning slowdown" discussion), which becomes
    a bigger problem the more layers a signal has to pass through. ReLU
    stays perfectly flat (slope 1) for every positive input, no matter
    how large, which keeps gradients flowing cleanly through a deeper
    network like this one -- part of why ReLU, not sigmoid, became the
    default choice once CNNs started stacking many layers together.
    """

    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_output):
        # Wherever the input was positive, the gradient passes through
        # unchanged (ReLU's slope there is exactly 1). Wherever the
        # input was zero or negative, no gradient flows at all (the
        # slope there is 0) -- so we simply zero out those spots.
        return grad_output * self._mask


class MaxPool2D(Layer):
    """
    A pooling layer: shrinks each feature map by keeping only the single
    strongest signal in each small zone, and throwing the rest away.

    Parameters
    ----------
    pool_size : int
        The size of each zone to shrink down (2 means every 2x2 block of
        the input becomes a single number in the output).

    This is the "zoom out" step. It does two useful things at once:
    it makes every later layer's job cheaper (there's simply less data
    to process), and it makes the network less fussy about a pattern
    landing one pixel to the left or right of exactly where it did in
    training, since a small shift usually still lands in the same zone
    and produces the same "strongest signal".
    """

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self._input_shape = None
        self._mask = None

    def forward(self, x):
        p = self.pool_size
        batch_size, channels, h, w = x.shape
        self._input_shape = x.shape

        # Reshape so every pool_size x pool_size zone becomes its own
        # pair of axes, then take the maximum across just those axes --
        # a reshape-and-reduce trick that replaces what would otherwise
        # be a slow Python loop over every zone with one vectorized call.
        x_reshaped = x.reshape(batch_size, channels, h // p, p, w // p, p)
        out = x_reshaped.max(axis=(3, 5))

        # Remember *which* position inside each zone had the maximum
        # value, so backward() knows exactly where to send the gradient.
        out_broadcast = out[:, :, :, None, :, None]
        self._mask = (x_reshaped == out_broadcast)
        return out

    def backward(self, grad_output):
        p = self.pool_size
        batch_size, channels, out_h, out_w = grad_output.shape

        # Every zone's incoming gradient belongs entirely to whichever
        # single position was the maximum during forward() -- every
        # other position in the zone gets zero, since changing a
        # non-maximum value by a tiny amount wouldn't have changed the
        # zone's output at all.
        grad_broadcast = grad_output[:, :, :, None, :, None]
        grad_reshaped = self._mask * grad_broadcast
        return grad_reshaped.reshape(self._input_shape)


class Flatten(Layer):
    """
    Reshapes a batch of multi-channel images into a batch of plain
    vectors, so a ``Dense`` layer can work with them.

    After a few rounds of convolution and pooling, each image in the
    batch is no longer a single 28x28 grid -- it's a stack of several
    smaller feature maps (one per filter). Before handing this off to
    the same kind of fully-connected "voting" layers used in
    ``network.py``, this layer flattens each image's whole stack of
    feature maps into one long list of numbers, exactly the shape a
    ``Dense`` layer expects.
    """

    def __init__(self):
        self._input_shape = None

    def forward(self, x):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self._input_shape)


class Dense(Layer):
    """
    A batched fully-connected layer: every input number is connected to
    every output neuron, exactly like every layer in ``network.py``'s
    ``Network`` class -- except here, an entire mini-batch of examples
    is processed in one matrix multiplication instead of looping over
    one example at a time.

    Parameters
    ----------
    in_features : int
        How many numbers each input example has.
    out_features : int
        How many neurons this layer has.
    """

    has_params = True

    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None
        self._cached_input = None

    def forward(self, x):
        """
        ``x`` has shape ``(batch_size, in_features)``. For the whole
        batch at once, ``x @ self.W.T`` computes every example's
        weighted sum for every neuron in a single matrix multiplication
        -- the batched equivalent of ``np.dot(w, a)`` in
        ``Network.feedforward``.
        """
        self._cached_input = x
        return x @ self.W.T + self.b

    def backward(self, grad_output):
        self.dW = grad_output.T @ self._cached_input
        self.db = grad_output.sum(axis=0)
        return grad_output @ self.W

    def params_and_grads(self):
        return [(self.W, self.dW, "W"), (self.b, self.db, "b")]


def softmax(logits):
    """
    Turn a batch of raw output scores ("logits") into proper
    probabilities that add up to exactly 1 across each example's 10
    digits.

    Unlike the sigmoid output layer in ``network.py`` (10 *independent*
    0-to-1 scores that don't necessarily add up to anything in
    particular), softmax is specifically built for "exactly one of these
    choices is correct" problems like digit classification: it
    guarantees the 10 numbers it produces are all positive and sum to 1,
    so they can be read directly as "there's an 87% chance this is a 3".

        softmax(z)_i = exp(z_i) / sum_j(exp(z_j))

    Subtracting each row's maximum value before exponentiating doesn't
    change the final answer (it cancels out in the division), but it
    keeps the numbers involved from overflowing to infinity when a logit
    is large -- a standard numerical-stability trick.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y_onehot):
    """
    Combine ``softmax`` with the cross-entropy cost function (see
    ``cost_functions.py`` for the plain-language explanation of what
    cross-entropy cost measures) and return both the cost and the
    gradient needed to start backpropagation.

    Just like pairing sigmoid output neurons with cross-entropy cost
    made the output layer's error signal collapse to the simple
    ``a - y`` (see ``CrossEntropyCost.delta``), pairing softmax with
    cross-entropy cost here produces the exact same kind of
    simplification: the gradient of the cost with respect to the raw
    logits (before softmax) is just ``probabilities - y_onehot``. Same
    underlying mathematical trick, same payoff (a clean, slowdown-free
    error signal), just for a different pair of activation and cost
    function.
    """
    batch_size = logits.shape[0]
    probs = softmax(logits)
    # Avoid log(0) for the (extremely unlikely) case of total certainty.
    clipped = np.clip(probs, 1e-12, 1.0)
    cost = -np.sum(y_onehot * np.log(clipped)) / batch_size
    grad_logits = (probs - y_onehot) / batch_size
    return cost, grad_logits, probs
