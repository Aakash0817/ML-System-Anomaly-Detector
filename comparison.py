"""
comparison.py
=============
Trains and benchmarks every detector on the same test set.

Improvements vs original
-------------------------
• Feature columns are sliced BEFORE passing to train/predict, so
  extra columns (per_cpu, avg_p_core, etc.) never reach the model.
• EnsembleDetector is included in the comparison.
• Adds ROC-AUC to the metrics table.
• Bar charts replaced with a cleaner radar + grouped-bar layout.
• Results are printed as a ranked table sorted by F1-score.
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import psutil
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from detectors.isolation_forest import IsolationForestDetector
from detectors.oneclass_svm import OneClassSVMDetector
from detectors.local_outlier import LocalOutlierFactorDetector
from detectors.pca_reconstruction import PCADetector
from detectors.random_forest import RandomForestDetector
from detectors.ensemble_detector import EnsembleDetector
from detectors.base import FEATURE_ORDER
from seeds import set_global_seeds
from console import enable_unicode_output

enable_unicode_output()


def _optional(label: str, factory):
    """Build a detector, or report why it is unavailable and skip it.

    XGBoost and TensorFlow are heavy optional wheels, and the neural detector
    also needs artefacts that only exist once train_neural.py has run. Both
    were constructed unconditionally at import, so a missing backend aborted
    the whole benchmark instead of costing it one row.
    """
    try:
        return factory()
    except Exception as exc:
        print(f"  SKIP {label}: {exc}")
        return None

# Fix all RNGs before any detector is constructed or trained.
set_global_seeds()

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading datasets …")
normal_df       = pd.read_csv('data/normal_training.csv')
labeled_train_df = pd.read_csv('data/labeled_training.csv')
test_df         = pd.read_csv('data/labeled_test.csv')

# Normalise labels:  collect_labeled uses 0=normal, 1=anomaly
# Detectors use convention 1=normal, -1=anomaly
def to_detector_label(x):
    return 1 if x == 0 else -1

labeled_train_df['label'] = labeled_train_df['label'].map(to_detector_label)
test_df['label']          = test_df['label'].map(to_detector_label)

X_normal       = normal_df[FEATURE_ORDER]
X_labeled_train = labeled_train_df[FEATURE_ORDER]
y_labeled_train  = labeled_train_df['label']
X_test         = test_df[FEATURE_ORDER]
y_test         = test_df['label'].values

# ─── Detector registry ────────────────────────────────────────────────────────
print("Building detectors …")


def _xgboost():
    from detectors.xgboost_detector import XGBoostDetector
    return XGBoostDetector()


def _neural():
    from detectors.neural_detector import NeuralDetector
    det = NeuralDetector()
    det.health_check()      # refuse to benchmark an unloaded model
    return det


# Ensemble sub-detectors are separate instances from the standalone rows.
_ens_members = [
    ('IF',  IsolationForestDetector()),
    ('LOF', LocalOutlierFactorDetector()),
    ('PCA', PCADetector()),
    ('RF',  RandomForestDetector()),
]
_ens_xgb = _optional('XGB (ensemble member)', _xgboost)
if _ens_xgb is not None:
    _ens_members.append(('XGB', _ens_xgb))
# The neural model is pre-trained, so it joins the ensemble as-is. It used to
# be left out entirely, which is why the ensemble never saw its vote.
_ens_neural = _optional('Neural (ensemble member)', _neural)
if _ens_neural is not None:
    _ens_members.append(('NN', _ens_neural))

DETECTORS = [
    ('Isolation Forest',    IsolationForestDetector(),       False),
    ('One-Class SVM',       OneClassSVMDetector(),           False),
    ('Local Outlier Factor',LocalOutlierFactorDetector(),    False),
    ('PCA Reconstruction',  PCADetector(),                   False),
    ('Random Forest',       RandomForestDetector(),          True),
    ('XGBoost',             _optional('XGBoost', _xgboost),  True),
    ('Neural Detector',     _optional('Neural Detector', _neural), False),
    ('Ensemble',            EnsembleDetector(_ens_members),  True),   # needs labels for RF/XGB sub-members
]
DETECTORS = [row for row in DETECTORS if row[1] is not None]

results = {}

# ─── Evaluate ─────────────────────────────────────────────────────────────────
for name, detector, needs_labels in DETECTORS:
    print(f"\n── {name} ──")
    # Training
    t0 = time.perf_counter()
    try:
        if needs_labels:
            detector.train(X_labeled_train, y_labeled_train)
        else:
            detector.train(X_normal)
    except Exception as exc:
        print(f"  TRAIN ERROR: {exc} — skipping.")
        continue
    train_time = time.perf_counter() - t0
    print(f"  train: {train_time:.2f}s")

    # Model size
    tmp = f"_tmp_{name.replace(' ','_')}.pkl"
    try:
        obj = detector.model if (hasattr(detector, 'model') and detector.model is not None) else detector
        joblib.dump(obj, tmp)
        model_kb = os.path.getsize(tmp) / 1024
    except Exception:
        model_kb = 0
    finally:
        # Leave no scratch file behind when the dump raises part-way.
        if os.path.exists(tmp):
            os.remove(tmp)

    # Memory
    proc = psutil.Process()
    mem0 = proc.memory_info().rss / 1024 / 1024
    _ = detector.predict(X_test.iloc[0].to_dict())
    mem_delta = proc.memory_info().rss / 1024 / 1024 - mem0

    # Inference
    preds, scores, lats = [], [], []
    for _, row in X_test.iterrows():
        p, s, l = detector.predict(row.to_dict())
        preds.append(p); scores.append(s); lats.append(l)

    avg_lat = np.mean(lats)
    f1   = f1_score(y_test, preds, pos_label=-1, zero_division=0)
    prec = precision_score(y_test, preds, pos_label=-1, zero_division=0)
    rec  = recall_score(y_test, preds, pos_label=-1, zero_division=0)
    # ROC-AUC: flip sign so that more-negative score = more anomalous
    try:
        auc = roc_auc_score((y_test == -1).astype(int), [-s for s in scores])
    except ValueError:
        auc = float('nan')

    results[name] = dict(
        train_time_s=round(train_time, 3),
        model_size_kb=round(model_kb, 1),
        runtime_mem_mb=round(mem_delta, 2),
        latency_ms=round(avg_lat, 3),
        throughput_ips=round(1000/avg_lat if avg_lat > 0 else 0, 1),
        f1_score=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        roc_auc=round(auc, 4),
    )
    print(f"  F1={f1:.3f}  AUC={auc:.3f}  latency={avg_lat:.3f}ms")

# ─── Table ────────────────────────────────────────────────────────────────────
if not results:
    print("\nNo detector completed the benchmark — nothing to report.")
    raise SystemExit(1)

df_results = pd.DataFrame(results).T.sort_values('f1_score', ascending=False)
print("\n" + "=" * 60)
print("RESULTS (sorted by F1)")
print("=" * 60)
print(df_results.to_string())
df_results.to_csv('comparison_results.csv')
print("\nSaved → comparison_results.csv")

# ─── Plots ────────────────────────────────────────────────────────────────────
names = df_results.index.tolist()
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

PLOT_METRICS = [
    ('f1_score',       'F1 Score',          'steelblue'),
    ('roc_auc',        'ROC-AUC',           'seagreen'),
    ('precision',      'Precision',         'darkorange'),
    ('recall',         'Recall',            'tomato'),
    ('latency_ms',     'Latency (ms)',      'slategray'),
    ('model_size_kb',  'Model Size (KB)',   'mediumpurple'),
]

for idx, (metric, title, color) in enumerate(PLOT_METRICS):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    vals = df_results[metric].values.astype(float)
    bars = ax.barh(names, vals, color=color, alpha=0.8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(title, fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    # Annotate bars. Offset by a fraction of the axis range, not by a
    # multiple of the bar width: a zero-width bar put its label at x=0,
    # on top of the axis.
    span = max(vals.max(), 0.0) - min(vals.min(), 0.0)
    pad = (span or 1.0) * 0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                f'{v:.3g}', va='center', fontsize=6)

fig.suptitle("Anomaly Detector Comparison", fontsize=14, fontweight='bold')
plt.savefig('comparison_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → comparison_plots.png")
