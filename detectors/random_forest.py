import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base import BaseDetector, FEATURE_ORDER


class RandomForestDetector(BaseDetector):
    NEEDS_LABELS = True

    def __init__(self, n_estimators: int = 100):
        self.model = None
        self.n_estimators = n_estimators
        self.feature_order = FEATURE_ORDER

    def train(self, X_train_df, y_train=None):
        if y_train is None:
            raise ValueError("RandomForestDetector requires y_train labels.")
        # Convention: 1 = normal, -1 = anomaly → convert to 1 / 0 for sklearn
        y_bin = (np.asarray(y_train) == 1).astype(int)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=10,
            random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_df[self.feature_order], y_bin)

    def predict(self, features_dict: dict) -> tuple:
        self.health_check()
        X = pd.DataFrame(
            [[features_dict[f] for f in self.feature_order]],
            columns=self.feature_order,
        )
        def _infer(x):
            return self.model.predict_proba(x)[0]   # [p_anomaly, p_normal]

        proba, latency = self._timed_predict(_infer, X)
        # Map to [-1, 1]: positive → normal
        score = float(proba[1] - proba[0])
        pred = 1 if score > 0 else -1
        return pred, score, latency

    def save(self, path: str) -> None:
        self.health_check()
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
