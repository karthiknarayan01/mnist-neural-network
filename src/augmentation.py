"""
augmentation.py
================

A technique for artificially growing the training set: take every digit
the network would normally learn from, and also show it several rotated
copies of that same digit. The label doesn't change (a "3" rotated
slightly is still a "3") -- only the pixels do.

Why would this help?
----------------------
A network can only ever be as good as the variety of examples it has
seen. The 50,000 original MNIST images were all written by a limited
group of people, so the network's sense of "what a 3 looks like" is
shaped entirely by their particular handwriting habits. If every real
person who ever uses this app happens to write their digits perfectly
upright, the network would be fine -- but real handwriting leans and
slants in all sorts of small ways. Showing the network some slightly
rotated versions of each training digit teaches it, directly, that
tilt shouldn't change the answer -- which is a form of regularization:
it reduces the network's tendency to memorize incidental quirks of the
original 50,000 images (like "this particular 7 happens to lean this
exact amount") in favor of learning the more fundamental shape of each
digit.

This is one of several "expand the training data" tricks suggested in
Michael Nielsen's book (alongside translations and elastic distortions);
this project implements small rotations only, capped at +/-15 degrees --
large enough to meaningfully vary each digit's slant, small enough that
a rotated "6" never starts looking like a "9" (which a larger rotation
risks).
"""

import numpy as np
from PIL import Image


def rotate_image(x, angle_degrees):
    """
    Rotate a single MNIST image by the given angle and return it in the
    exact same format it started in.

    Parameters
    ----------
    x : np.ndarray
        A single training image, as a 784x1 column of pixel brightness
        values between 0 and 1 (the same format `mnist_loader` produces).
    angle_degrees : float
        How far to rotate, in degrees. Positive values rotate
        counterclockwise, negative values rotate clockwise -- Pillow's
        usual convention.

    Returns
    -------
    np.ndarray
        A new 784x1 column: the rotated image, in the same 0-1 pixel
        format as the input.

    How it works
    --------------
    NumPy doesn't have a built-in "rotate an image" operation, so this
    function temporarily hands the image to Pillow (a general-purpose
    image library) which knows how to do that, then converts the result
    straight back into the plain NumPy array format the rest of this
    project uses everywhere else.

    1. Reshape the flat 784-number column back into its natural 28x28
       grid, and scale it from this project's 0-1 brightness range up to
       the standard 0-255 range image libraries expect.
    2. Rotate it with Pillow, using bilinear resampling (which blends
       neighboring pixels smoothly, avoiding jagged, blocky edges) and a
       black fill color for any new corners the rotation exposes --
       matching MNIST's plain black background.
    3. Scale back down to 0-1 and flatten back into a 784x1 column.
    """
    image_28x28 = (x.reshape(28, 28) * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_28x28, mode="L")
    rotated = pil_image.rotate(
        angle_degrees,
        resample=Image.BILINEAR,
        fillcolor=0,
        expand=False,
    )
    rotated_array = np.asarray(rotated, dtype=np.float32) / 255.0
    return rotated_array.reshape(784, 1)


def expand_training_data(training_data, max_angle=15.0, copies_per_image=10):
    """
    Build an enlarged training set by adding several rotated copies of
    every image in ``training_data``.

    Parameters
    ----------
    training_data : list of (x, y) pairs
        The original training set, in the format `mnist_loader` produces
        (``x`` a 784x1 pixel column, ``y`` a 10x1 one-hot answer column).
    max_angle : float
        The largest rotation to use in either direction, in degrees.
    copies_per_image : int
        How many extra rotated copies to generate per original image.
        The rotation angles are spaced evenly across the full
        ``[-max_angle, +max_angle]`` range (skipping 0 degrees, since
        the unrotated original is already included).

    Returns
    -------
    list of (x, y) pairs
        The original images, followed by all of their rotated copies --
        ``len(training_data) * (copies_per_image + 1)`` examples in
        total. Each rotated copy keeps the exact same label ``y`` as the
        original it was generated from, since rotating a digit doesn't
        change what digit it is.
    """
    # np.linspace(-15, 15, 11) places 11 evenly-spaced angles across the
    # range, including both endpoints and 0 right in the middle; slicing
    # out every angle except the middle one leaves exactly
    # `copies_per_image` non-zero angles, symmetric around 0.
    angles = np.linspace(-max_angle, max_angle, copies_per_image + 1)
    angles = [a for a in angles if not np.isclose(a, 0.0)]

    expanded = list(training_data)
    total = len(training_data)
    for i, (x, y) in enumerate(training_data):
        for angle in angles:
            expanded.append((rotate_image(x, angle), y))
        if (i + 1) % 10000 == 0:
            print(f"  augmented {i + 1}/{total} images...")

    return expanded
