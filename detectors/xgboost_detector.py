import joblib
import numpy as np
import pandas as pd
from .base import BaseDetector, FEATURE_ORDER


class XGBoostDetector(BaseDetector):
    NEEDS_LABELS = True

    def __init__(self):
        self.model = None
        self.feature_order = FEATURE_ORDER

    def train(self, X_train_df, y_train=None):
        if y_train is None:
            raise ValueError("XGBoostDetector requires y_train labels.")
        import xgboost as xgb
        y_bin = (np.asarray(y_train) == 1).astype(int)
        dtrain = xgb.DMatrix(X_train_df[self.feature_order], label=y_bin)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'eta': 0.1,
            'eval_metric': 'logloss',
            'seed': 42,
            'nthread': -1,
        }
        self.model = xgb.train(params, dtrain, num_boost_round=100,
                               verbose_eval=False)

    def predict(self, features_dict: dict) -> tuple:
        self.health_check()
        import xgboost as xgb
        X = pd.DataFrame(
            [[features_dict[f] for f in self.feature_order]],
            columns=self.feature_order,
        )
        dtest = xgb.DMatrix(X)

        (proba,), latency = self._timed_predict(self.model.predict, dtest)
        score = float(2 * proba - 1)   # [0,1] → [-1,1]
        pred = 1 if score > 0 else -1
        return pred, score, latency

    def save(self, path: str) -> None:
        self.health_check()
        self.model.save_model(path)

    def load(self, path: str) -> None:
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(path)
