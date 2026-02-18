import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseDetector, FEATURE_ORDER


class LocalOutlierFactorDetector(BaseDetector):
    NEEDS_LABELS = False

    def __init__(self, n_neighbors: int = 20):
        self.model = None
        self.n_neighbors = n_neighbors
        self.feature_order = FEATURE_ORDER

    def train(self, X_train_df, y_train=None):
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, novelty=True, n_jobs=-1
        )
        self.model.fit(X_train_df[self.feature_order].values)

    def predict(self, features_dict: dict) -> tuple:
        self.health_check()
        X = np.array([[features_dict[f] for f in self.feature_order]])
        (score,), latency = self._timed_predict(
            self.model.decision_function, X
        )
        pred = 1 if score > 0 else -1
        return pred, float(score), latency

    def save(self, path: str) -> None:
        self.health_check()
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
