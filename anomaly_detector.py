"""
anomaly_detector.py
===================
Standalone single-model anomaly detector used by monitor.py.
Includes:
  • RunningStats  – online mean/std with configurable window
  • AnomalyExplainer – per-feature z-score attribution
  • DriftDetector – Page-Hinkley change-point detector; fires when
                    the score distribution shifts significantly, signalling
                    that the model may need retraining.
  • AnomalyDetector – wraps a joblib-serialised IsolationForest with
                      graceful dummy fallback and all of the above.
"""

import joblib
import time
import os
import atexit
import numpy as np
import pandas as pd
from collections import deque

# Minimum samples before z-score is meaningful
_MIN_STATS_SAMPLES = 30

FEATURE_ORDER = [
    'cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
    'gpu_percent', 'gpu_memory', 'gpu_temp',
]


# ─────────────────────────────────────────────────────────────────────────────
# RunningStats
# ─────────────────────────────────────────────────────────────────────────────

class RunningStats:
    def __init__(self, window: int = 200):
        self.window = window
        self.values: deque = deque(maxlen=window)
        self.mean = 0.0
        self.std = 1.0

    def update(self, value: float) -> None:
        self.values.append(value)
        arr = np.asarray(self.values)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr)) if len(arr) > 1 else 1.0

    def zscore(self, value: float) -> float:
        """Return z-score; safe before window fills."""
        if len(self.values) < _MIN_STATS_SAMPLES:
            return 0.0   # not enough data yet – treat as normal
        denom = self.std if self.std > 1e-9 else 1e-9
        return (value - self.mean) / denom

    @property
    def ready(self) -> bool:
        return len(self.values) >= _MIN_STATS_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyExplainer
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyExplainer:
    def __init__(self, feature_names: list, window: int = 200):
        self.feature_names = feature_names
        self.stats = {name: RunningStats(window) for name in feature_names}

    def update_normal(self, features_dict: dict) -> None:
        for name in self.feature_names:
            if name in features_dict:
                self.stats[name].update(features_dict[name])

    def explain(self, features_dict: dict, top_n: int = 3) -> list:
        """Return list of human-readable strings for the top deviating features."""
        scored = []
        for name in self.feature_names:
            if name in features_dict and self.stats[name].ready:
                val = features_dict[name]
                z = abs(self.stats[name].zscore(val))
                scored.append((name, z, val, self.stats[name].mean))
        scored.sort(key=lambda x: x[1], reverse=True)
        result = []
        for name, z, val, mean in scored[:top_n]:
            direction = "high" if val > mean else "low"
            result.append(f"{name}: {val:.1f} ({direction}, expected ≈{mean:.1f}, z={z:.1f})")
        return result or ["Insufficient baseline data for explanation."]


# ─────────────────────────────────────────────────────────────────────────────
# DriftDetector  (Page-Hinkley)
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Page-Hinkley test on the anomaly score stream.
    Raises drift_flag when the running minimum of cumulative deviations
    exceeds a threshold, suggesting a regime change.
    """
    def __init__(self, delta: float = 0.005, threshold: float = 50.0,
                 warmup: int = 100):
        self.delta = delta
        self.threshold = threshold
        self.warmup = warmup
        self._n = 0
        self._cum_sum = 0.0
        self._min_cum = 0.0
        self._mean = 0.0
        self.drift_flag = False

    def update(self, value: float) -> bool:
        """Update and return True if drift detected."""
        self._n += 1
        # Online mean
        self._mean += (value - self._mean) / self._n
        self._cum_sum += value - self._mean - self.delta
        self._min_cum = min(self._min_cum, self._cum_sum)
        if self._n > self.warmup and (self._cum_sum - self._min_cum) > self.threshold:
            self.drift_flag = True
        return self.drift_flag

    def reset(self) -> None:
        self._n = 0
        self._cum_sum = 0.0
        self._min_cum = 0.0
        self._mean = 0.0
        self.drift_flag = False


# ─────────────────────────────────────────────────────────────────────────────
# AnomalyDetector
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    def __init__(self, model_path: str = 'isolation_forest.pkl'):
        if not os.path.exists(model_path):
            print(f"⚠ Model file '{model_path}' not found — using dummy detector.")
            self.model = None
        else:
            self.model = joblib.load(model_path)
            print(f"✓ Loaded model from '{model_path}'")

        self.feature_order = FEATURE_ORDER
        self.explainer = AnomalyExplainer(self.feature_order, window=200)
        self.drift = DriftDetector()
        self.latencies: list = []
        self._anomaly_count = 0
        self._total_count = 0

    # ------------------------------------------------------------------
    def predict(self, features: dict) -> tuple:
        """
        Returns (pred, latency_ms, explanation, score, drift_detected)
        pred        : 1 = normal, -1 = anomaly
        latency_ms  : inference time
        explanation : list[str] or None
        score       : raw decision-function value
        drift       : bool – True when score distribution seems to have shifted
        """
        self._total_count += 1

        if self.model is None:
            return self._dummy_predict(features)

        X = pd.DataFrame(
            [[features[k] for k in self.feature_order]],
            columns=self.feature_order,
        )
        start = time.perf_counter()
        score = float(self.model.decision_function(X)[0])
        latency = (time.perf_counter() - start) * 1000
        self.latencies.append(latency)

        pred = 1 if score > 0 else -1
        drift = self.drift.update(score)

        if pred == 1:
            self.explainer.update_normal(features)
            explanation = None
        else:
            self._anomaly_count += 1
            explanation = self.explainer.explain(features, top_n=3)

        return pred, latency, explanation, score, drift

    def _dummy_predict(self, features: dict) -> tuple:
        time.sleep(0.001)
        score = float(np.random.uniform(-0.5, 0.5))
        pred = 1 if score > 0 else -1
        if pred == 1:
            self.explainer.update_normal(features)
            explanation = None
        else:
            self._anomaly_count += 1
            explanation = self.explainer.explain(features, top_n=3)
        return pred, 1.0, explanation, score, False

    # ------------------------------------------------------------------
    @property
    def anomaly_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._anomaly_count / self._total_count

    def summary(self) -> str:
        lines = [
            f"Total samples  : {self._total_count}",
            f"Anomalies      : {self._anomaly_count} ({self.anomaly_rate:.1%})",
        ]
        if self.latencies:
            lines += [
                f"Avg latency    : {np.mean(self.latencies):.2f} ms",
                f"Max latency    : {np.max(self.latencies):.2f} ms",
                f"P99 latency    : {np.percentile(self.latencies, 99):.2f} ms",
            ]
        if self.drift.drift_flag:
            lines.append("⚠  DRIFT DETECTED – consider retraining the model.")
        return "\n".join(lines)
