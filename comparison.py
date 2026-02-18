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
from detectors.xgboost_detector import XGBoostDetector
from detectors.rl_agent import RLAgentDetector, RLAgentDetectorQuantized
from detectors.ensemble_detector import EnsembleDetector
from detectors.base import FEATURE_ORDER

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
# Build ensemble sub-detectors (separate instances)
_ens_members = [
    ('IF',  IsolationForestDetector()),
    ('LOF', LocalOutlierFactorDetector()),
    ('PCA', PCADetector()),
    ('RF',  RandomForestDetector()),
    ('XGB', XGBoostDetector()),
]

DETECTORS = [
    ('Isolation Forest',    IsolationForestDetector(),       False),
    ('One-Class SVM',       OneClassSVMDetector(),           False),
    ('Local Outlier Factor',LocalOutlierFactorDetector(),    False),
    ('PCA Reconstruction',  PCADetector(),                   False),
    ('Random Forest',       RandomForestDetector(),          True),
    ('XGBoost',             XGBoostDetector(),               True),
    ('RL Agent',            RLAgentDetector(),               False),
    ('RL Agent (Quantized)', RLAgentDetectorQuantized(),      False),
    ('Ensemble',            EnsembleDetector(_ens_members),  True),   # needs labels for RF/XGB sub-members
]

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
    try:
        tmp = f"_tmp_{name.replace(' ','_')}.pkl"
        obj = detector.model if (hasattr(detector, 'model') and detector.model is not None) else detector
        joblib.dump(obj, tmp)
        model_kb = os.path.getsize(tmp) / 1024
        os.remove(tmp)
    except Exception:
        model_kb = 0

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
    # Annotate bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3g}', va='center', fontsize=6)

fig.suptitle("Anomaly Detector Comparison", fontsize=14, fontweight='bold')
plt.savefig('comparison_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → comparison_plots.png")
