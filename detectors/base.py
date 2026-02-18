import abc
import time

# Canonical feature set used across the entire project.
FEATURE_ORDER = [
    'cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
    'gpu_percent', 'gpu_memory', 'gpu_temp',
]


class BaseDetector(abc.ABC):
    """
    Abstract base for every anomaly detector.

    Contract
    --------
    • train()   – fit (or load) the model
    • predict() – return (pred, score, latency_ms)
                  pred   : -1 = anomaly, 1 = normal
                  score  : higher → more normal (normalised to [-1, 1] where possible)
                  latency: wall-clock inference time in milliseconds
    • save()    – persist the model to disk
    • load()    – restore the model from disk
    • health_check() – raise RuntimeError if the model is not ready
    """

    # Subclasses may override to declare whether they need labelled data.
    NEEDS_LABELS: bool = False

    # ------------------------------------------------------------------ #
    #  Required interface                                                  #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def train(self, X_train_df, y_train=None):
        """
        Train on a DataFrame whose columns include FEATURE_ORDER.
        y_train is required when NEEDS_LABELS is True.
        """

    @abc.abstractmethod
    def predict(self, features_dict: dict) -> tuple:
        """
        Predict on a single sample.

        Parameters
        ----------
        features_dict : dict with at least the keys in FEATURE_ORDER.

        Returns
        -------
        (pred, score, latency_ms)
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path*."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Restore a trained model from *path*."""

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    def health_check(self) -> None:
        """
        Raise RuntimeError if the detector is not ready to predict.
        Subclasses may override for additional checks.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has no trained model. "
                "Call train() or load() first."
            )

    def _timed_predict(self, fn, *args, **kwargs):
        """
        Utility wrapper: calls *fn* and returns (result, latency_ms).
        Use inside predict() implementations to keep timing consistent.
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        latency = (time.perf_counter() - start) * 1000
        return result, latency

    def __repr__(self):
        ready = hasattr(self, 'model') and self.model is not None
        return f"<{self.__class__.__name__} ready={ready}>"
