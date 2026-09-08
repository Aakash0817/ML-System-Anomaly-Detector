"""
neural_detector.py
==================
Detector wrapper for the trained Keras binary classifier.
"""

import time                     # ← new import
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'neural_detector.keras'
SCALER_PATH = BASE_DIR / 'neural_detector_scaler.pkl'
FEATURES = ['cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
            'gpu_percent', 'gpu_memory', 'gpu_temp']


class NeuralDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_order = FEATURES

    def train(self, X=None, y=None):
        """Pre-trained model is already loaded – nothing to do."""
        pass

    def predict(self, features: dict):
        """
        Returns:
            pred:  1 = normal, -1 = anomaly
            score: higher = more normal, in [-1, 1]
            latency: inference time in milliseconds
        """
        start = time.perf_counter()          # ← start timer

        # Build feature vector in correct order
        X = np.array([[features[f] for f in self.feature_order]])
        X_scaled = self.scaler.transform(X)

        # The sigmoid head outputs P(anomaly), because train_neural.py keeps the
        # collect_labeled convention of 0 = normal, 1 = anomaly. Map it onto the
        # project-wide score convention used by every other detector:
        # higher = more normal, on [-1, 1].
        proba = self.model.predict(X_scaled, verbose=0)[0, 0]
        score = float(1.0 - 2.0 * proba)

        pred = 1 if score > 0 else -1

        latency = (time.perf_counter() - start) * 1000   # ← elapsed ms

        return pred, score, latency