"""
neural_detector.py
==================
Detector wrapper for the trained Keras binary classifier.

Implements the full BaseDetector contract. It previously did not subclass
BaseDetector and had no health_check(), save() or load(), so
EnsembleDetector — which calls det.health_check() before every sub-predict —
raised AttributeError on it, caught the exception, and silently dropped the
neural model from every ensemble vote.
"""

import joblib
import numpy as np
from pathlib import Path

from .base import BaseDetector, FEATURE_ORDER

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'neural_detector.keras'
SCALER_PATH = BASE_DIR / 'neural_detector_scaler.pkl'


def _paths_for(path) -> tuple:
    """Map a single path onto the (model, scaler) pair it stands for.

    The model is a Keras file and the scaler a joblib pickle, so one path
    cannot hold both. Callers that pass '<dir>/neural.pkl' — EnsembleDetector
    does exactly that — get '<dir>/neural.keras' and '<dir>/neural_scaler.pkl'.
    """
    p = Path(path)
    return p.with_suffix('.keras'), p.with_name(p.stem + '_scaler.pkl')


class NeuralDetector(BaseDetector):
    # The network ships pre-trained; train() is a no-op and needs no labels.
    NEEDS_LABELS = False

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model = None
        self.scaler = None
        self.feature_order = FEATURE_ORDER
        # Load eagerly when the artefacts are present, but do not explode when
        # they are not: health_check() then reports the standard "no trained
        # model" error instead of a stack trace at construction time.
        if Path(model_path).exists() and Path(scaler_path).exists():
            self._load_pair(model_path, scaler_path)
        else:
            print(f"⚠ NeuralDetector: '{model_path}' or '{scaler_path}' "
                  f"missing — run train_neural.py.")

    # ------------------------------------------------------------------ #
    def _load_pair(self, model_path, scaler_path) -> None:
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def train(self, X_train_df=None, y_train=None):
        """Pre-trained model is already loaded – nothing to do."""
        return self

    def health_check(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError(
                "NeuralDetector has no trained model. Run train_neural.py, "
                "or call load() first."
            )

    def predict(self, features_dict: dict) -> tuple:
        """
        Returns:
            pred:  1 = normal, -1 = anomaly
            score: higher = more normal, in [-1, 1]
            latency: inference time in milliseconds
        """
        self.health_check()
        X = np.array([[features_dict[f] for f in self.feature_order]],
                     dtype=float)

        def _infer(x):
            x_scaled = self.scaler.transform(x)
            # The sigmoid head outputs P(anomaly), because train_neural.py
            # keeps the collect_labeled convention of 0 = normal, 1 = anomaly.
            # Map it onto the project-wide score convention used by every
            # other detector: higher = more normal, on [-1, 1].
            return float(self.model.predict(x_scaled, verbose=0)[0, 0])

        # Scaling is inside the timed section: it is part of the per-sample
        # cost the dashboard reports, and it was previously timed anyway.
        proba, latency = self._timed_predict(_infer, X)
        score = 1.0 - 2.0 * proba
        pred = 1 if score > 0 else -1
        return pred, float(score), latency

    def save(self, path: str) -> None:
        self.health_check()
        model_path, scaler_path = _paths_for(path)
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, path: str) -> None:
        model_path, scaler_path = _paths_for(path)
        for p in (model_path, scaler_path):
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
        self._load_pair(model_path, scaler_path)
