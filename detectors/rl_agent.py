"""
rl_agent.py
===========
Detector wrapper for the trained Keras classifier (RL agent).
"""

import time                     # ← new import
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'rl_agent.keras'
SCALER_PATH = BASE_DIR / 'rl_agent_scaler.pkl'
FEATURES = ['cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
            'gpu_percent', 'gpu_memory', 'gpu_temp']


class RLAgentDetector:
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
            score: probability of anomaly (0..1)
            latency: inference time in milliseconds
        """
        start = time.perf_counter()          # ← start timer

        # Build feature vector in correct order
        X = np.array([[features[f] for f in self.feature_order]])
        X_scaled = self.scaler.transform(X)

        # Probability of class 1 (anomaly)
        proba = self.model.predict(X_scaled, verbose=0)[0, 0]
        score = float(proba)

        # Convert to -1/1 prediction (threshold at 0.5)
        pred = -1 if score > 0.5 else 1

        latency = (time.perf_counter() - start) * 1000   # ← elapsed ms

        return pred, score, latency