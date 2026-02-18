import joblib
import pandas as pd
from .base import BaseDetector, FEATURE_ORDER


class IsolationForestDetector(BaseDetector):
    NEEDS_LABELS = False

    def __init__(self, contamination: float = 0.05):
        self.model = None
        self.contamination = contamination
        self.feature_order = FEATURE_ORDER

    def train(self, X_train_df, y_train=None):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train_df[self.feature_order])

    def predict(self, features_dict: dict) -> tuple:
        self.health_check()
        X = pd.DataFrame(
            [[features_dict[f] for f in self.feature_order]],
            columns=self.feature_order,
        )
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
