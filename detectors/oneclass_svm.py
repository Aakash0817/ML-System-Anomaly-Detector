import joblib
import pandas as pd
from sklearn.svm import OneClassSVM
from .base import BaseDetector, FEATURE_ORDER


class OneClassSVMDetector(BaseDetector):
    NEEDS_LABELS = False

    def __init__(self, nu: float = 0.05):
        self.model = None
        self.nu = nu
        self.feature_order = FEATURE_ORDER

    def train(self, X_train_df, y_train=None):
        self.model = OneClassSVM(nu=self.nu, kernel='rbf', gamma='scale')
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
