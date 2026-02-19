"""
pca_reconstruction.py
=====================
PCA‑based anomaly detector using reconstruction error.
Now uses a fixed number of components to guarantee imperfect reconstruction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .base import BaseDetector, FEATURE_ORDER


class PCADetector(BaseDetector):
    """
    Anomaly detector based on PCA reconstruction error.
    Uses a fixed number of principal components.
    """
    NEEDS_LABELS = False

    def __init__(self, n_components=5, threshold_percentile=95):
        """
        Parameters
        ----------
        n_components : int
            Number of principal components to keep (must be < number of features).
        threshold_percentile : float
            Percentile of training reconstruction errors used as threshold.
        """
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.pca = None
        self.scaler = None
        self.threshold = None
        self.model = None
        self.feature_order = FEATURE_ORDER

    def train(self, X, y=None):
        """
        Fit PCA on normal data (X) and compute reconstruction error threshold.
        X: DataFrame or numpy array with shape (n_samples, n_features)
        """
        if isinstance(X, np.ndarray):
            X_arr = X
        else:
            X_arr = X[self.feature_order].values

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        # Ensure n_components is less than number of features
        n_features = X_scaled.shape[1]
        if self.n_components >= n_features:
            raise ValueError(
                f"n_components ({self.n_components}) must be less than number "
                f"of features ({n_features}) to create reconstruction error."
            )

        # Fit PCA with fixed number of components
        self.pca = PCA(n_components=self.n_components, whiten=False)
        self.pca.fit(X_scaled)

        # Compute reconstruction errors on training data
        X_proj = self.pca.transform(X_scaled)
        X_recon = self.pca.inverse_transform(X_proj)
        errors = np.mean((X_scaled - X_recon) ** 2, axis=1)

        # Set threshold as given percentile of errors
        self.threshold = np.percentile(errors, self.threshold_percentile)

        # Mark model as ready
        self.model = self.pca

    def _reconstruct_error(self, X_scaled: np.ndarray) -> float:
        """Compute MSE reconstruction error for a single scaled sample."""
        X_proj = self.pca.transform(X_scaled)
        X_recon = self.pca.inverse_transform(X_proj)
        return float(np.mean((X_scaled - X_recon) ** 2))

    def predict(self, features_dict: dict) -> tuple:
        
        X = np.array([[features_dict[f] for f in self.feature_order]])
        X_scaled = self.scaler.transform(X)

        error, latency = self._timed_predict(self._reconstruct_error, X_scaled)

        # Normalize by the training threshold
        relative_error = error / self.threshold
        score = -relative_error

        pred = 1 if error <= self.threshold else -1
        return pred, float(score), latency

    def health_check(self) -> None:
        if self.pca is None or self.scaler is None or self.threshold is None:
            raise RuntimeError(
                "PCADetector is not ready. Call train() or load() first."
            )

    def save(self, path: str) -> None:
        self.health_check()
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'n_components': self.n_components,
            'threshold_percentile': self.threshold_percentile,
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model_data = joblib.load(path)
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.n_components = model_data['n_components']
        self.threshold_percentile = model_data['threshold_percentile']
        self.model = self.pca