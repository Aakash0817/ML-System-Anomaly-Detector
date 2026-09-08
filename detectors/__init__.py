"""
detectors
=========
Detector registry.

XGBoost and TensorFlow are optional at import time: the modules that need
them are resolved lazily through PEP 562's module __getattr__, so importing
this package (or any sklearn-only detector) still works in an environment
where those two heavyweight wheels are not installed. Accessing
XGBoostDetector or NeuralDetector without its backend raises ImportError
with an actionable message, at the point of use rather than at import.
"""

from .base import BaseDetector, FEATURE_ORDER
from .isolation_forest import IsolationForestDetector
from .local_outlier import LocalOutlierFactorDetector
from .oneclass_svm import OneClassSVMDetector
from .pca_reconstruction import PCADetector
from .random_forest import RandomForestDetector
from .ensemble_detector import EnsembleDetector

# name -> (module, backend distribution needed by that module)
_LAZY = {
    'XGBoostDetector': ('.xgboost_detector', 'xgboost'),
    'NeuralDetector':  ('.neural_detector', 'tensorflow'),
}

__all__ = [
    'BaseDetector', 'FEATURE_ORDER',
    'IsolationForestDetector', 'LocalOutlierFactorDetector',
    'OneClassSVMDetector', 'PCADetector',
    'RandomForestDetector', 'XGBoostDetector',
    'NeuralDetector', 'EnsembleDetector',
]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, backend = _LAZY[name]
    from importlib import import_module
    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the optional '{backend}' dependency "
            f"(pip install {backend}): {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value        # cache, so this runs once
    return value


def __dir__():
    return sorted(__all__)
