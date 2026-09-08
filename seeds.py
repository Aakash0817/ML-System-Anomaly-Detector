"""
seeds.py
========
Central random-seed control so training and benchmark runs are reproducible.

Call set_global_seeds() once at the start of any entry point that trains a
model or splits data (comparison.py, monitor.py, train_rl.py). Estimators
that expose their own seed parameter also set it explicitly, so a detector
stays reproducible even when constructed directly.

Note: OneClassSVM and LocalOutlierFactor accept no random_state — both are
deterministic for a given training set, so there is nothing to seed.
"""

import os
import random

import numpy as np

SEED = 42


def set_global_seeds(seed: int = SEED) -> None:
    """
    Seed Python's `random`, NumPy, and TensorFlow/Keras.

    TensorFlow is seeded only if it is importable, so the non-GUI scripts
    still work in an environment without it.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        return

    # Seeds Python, NumPy and TensorFlow's own RNGs in one call, and drives
    # weight initialisation, dropout masks and fit() shuffling.
    tf.keras.utils.set_random_seed(seed)

    # Makes GPU/multi-threaded kernels deterministic. Unsupported on some
    # op/hardware combinations, so failure here is not fatal.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
