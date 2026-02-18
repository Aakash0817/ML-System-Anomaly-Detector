"""
EnsembleDetector
================
Combines predictions from multiple BaseDetector instances using
soft-vote (score averaging) with optional per-detector weights.

Key properties
--------------
• Gracefully skips any detector that is not ready (no crash).
• Returns an aggregate score in [-1, 1] so it fits the same dashboard.
• Exposes individual sub-detector scores for debugging.
• Implements save/load by delegating to each sub-detector.
"""

import time
import numpy as np
from .base import BaseDetector, FEATURE_ORDER


class EnsembleDetector(BaseDetector):
    NEEDS_LABELS = False   # determined per sub-detector at training time

    def __init__(self, detectors: list, weights: list = None):
        """
        Parameters
        ----------
        detectors : list of (name: str, detector: BaseDetector) tuples
        weights   : optional list of floats (same length as detectors).
                    Defaults to uniform weighting.
        """
        if not detectors:
            raise ValueError("At least one sub-detector is required.")
        self.detectors = detectors   # [(name, detector), ...]
        n = len(detectors)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError("weights length must match detectors length.")
            total = sum(weights)
            self.weights = [w / total for w in weights]   # normalise

        self.feature_order = FEATURE_ORDER
        # model sentinel – set to True once at least one sub-detector is trained
        self.model = None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, X_train_df, y_train=None):
        """
        Train each sub-detector. Supervised detectors receive y_train;
        unsupervised ones receive only X_train_df.
        """
        trained = 0
        for name, det in self.detectors:
            try:
                if det.NEEDS_LABELS:
                    if y_train is None:
                        print(f"[Ensemble] Skipping {name}: needs labels but none provided.")
                        continue
                    det.train(X_train_df, y_train)
                else:
                    det.train(X_train_df)
                trained += 1
                print(f"[Ensemble] Trained {name}")
            except Exception as exc:
                print(f"[Ensemble] WARNING – {name} failed to train: {exc}")

        if trained == 0:
            raise RuntimeError("Ensemble: no sub-detectors trained successfully.")
        self.model = True   # mark as ready

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, features_dict: dict) -> tuple:
        """
        Returns (pred, score, latency_ms) where score is the
        weighted average of individual scores.

        Also populates self.last_breakdown with per-detector details
        for inspection / GUI display.
        """
        self.health_check()
        start = time.perf_counter()

        weighted_scores = []
        active_weights = []
        self.last_breakdown = {}

        for (name, det), w in zip(self.detectors, self.weights):
            try:
                det.health_check()
                pred_i, score_i, lat_i = det.predict(features_dict)
                weighted_scores.append(score_i * w)
                active_weights.append(w)
                self.last_breakdown[name] = {
                    'pred': pred_i, 'score': score_i, 'latency_ms': lat_i
                }
            except Exception as exc:
                print(f"[Ensemble] {name} predict error: {exc}")
                self.last_breakdown[name] = {'pred': 0, 'score': 0.0, 'latency_ms': 0.0}

        latency = (time.perf_counter() - start) * 1000

        if not active_weights:
            return 0, 0.0, latency

        # Re-normalise in case some detectors were skipped
        total_w = sum(active_weights)
        agg_score = sum(weighted_scores) / total_w if total_w > 0 else 0.0
        pred = 1 if agg_score > 0 else -1
        return pred, float(agg_score), latency

    def vote_breakdown(self) -> dict:
        """
        After calling predict(), returns per-detector results.
        Useful for GUI display of agreement / disagreement.
        """
        return getattr(self, 'last_breakdown', {})

    def agreement_rate(self) -> float:
        """
        Fraction of sub-detectors that agree with the ensemble decision.
        Call after predict().
        """
        bd = self.vote_breakdown()
        if not bd:
            return 0.0
        preds = [v['pred'] for v in bd.values() if v['pred'] != 0]
        if not preds:
            return 0.0
        majority = 1 if sum(p == 1 for p in preds) > len(preds) / 2 else -1
        return sum(p == majority for p in preds) / len(preds)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        """
        Save each sub-detector to <directory>/<name>.pkl (or .model).
        """
        import os, joblib
        os.makedirs(directory, exist_ok=True)
        for name, det in self.detectors:
            safe_name = name.replace(' ', '_').lower()
            path = os.path.join(directory, f"{safe_name}.pkl")
            try:
                det.save(path)
                print(f"[Ensemble] Saved {name} → {path}")
            except Exception as exc:
                print(f"[Ensemble] Could not save {name}: {exc}")
        # Save weights
        import json
        meta = {
            'weights': self.weights,
            'names': [n for n, _ in self.detectors],
        }
        with open(os.path.join(directory, 'ensemble_meta.json'), 'w') as f:
            import json; json.dump(meta, f, indent=2)

    def load(self, directory: str) -> None:
        """
        Load each sub-detector from <directory>/<name>.pkl.
        """
        import os
        for name, det in self.detectors:
            safe_name = name.replace(' ', '_').lower()
            path = os.path.join(directory, f"{safe_name}.pkl")
            if os.path.exists(path):
                try:
                    det.load(path)
                    print(f"[Ensemble] Loaded {name} from {path}")
                except Exception as exc:
                    print(f"[Ensemble] Could not load {name}: {exc}")
            else:
                print(f"[Ensemble] No saved model for {name} at {path}")
        self.model = True
