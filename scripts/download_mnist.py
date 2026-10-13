"""
download_mnist.py
==================

A tiny helper script that fetches the raw MNIST dataset file so
`scripts/train.py` has something to load. The file itself isn't checked
into this repository (it's ~17 MB of binary data, a poor fit for git),
so this script downloads it the first time you need it.
"""

import os
import urllib.request

URL = (
    "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
    "master/data/mnist.pkl.gz"
)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEST_PATH = os.path.join(PROJECT_ROOT, "data", "mnist.pkl.gz")


def main():
    os.makedirs(os.path.dirname(DEST_PATH), exist_ok=True)
    if os.path.exists(DEST_PATH):
        print(f"Already have {DEST_PATH}, skipping download.")
        return
    print(f"Downloading MNIST dataset from {URL} ...")
    urllib.request.urlretrieve(URL, DEST_PATH)
    print(f"Saved to {DEST_PATH}")


if __name__ == "__main__":
    main()
