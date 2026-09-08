"""
collect_normal.py
=================
Collect a baseline of normal system behaviour into data/normal_training.csv.
"""

import time
from pathlib import Path

import pandas as pd

from console import enable_unicode_output
from data_collector import collect_all_metrics
from detectors.base import FEATURE_ORDER

enable_unicode_output()

DURATION = 1200  # 10 minutes – you can increase to 30 minutes for better baseline
SAMPLE_INTERVAL = 1
OUTPUT_PATH = Path('data/normal_training.csv')

# The unsupervised detectors train on FEATURE_ORDER only. Keep the per-core
# breakdown out of the file: it is a list of dicts, so it lands in the CSV as
# a quoted repr that nothing reads back, and it dominates the file size.
EXTRA_COLUMNS = ['avg_p_core', 'avg_e_core']

print(f"Collecting normal data for {DURATION} seconds. "
      "Please use your computer normally (no heavy stress).")

data = []
next_sample_at = time.perf_counter()
for i in range(DURATION):
    metrics = collect_all_metrics()
    data.append({col: metrics.get(col)
                 for col in FEATURE_ORDER + EXTRA_COLUMNS})

    if (i + 1) % 60 == 0:
        print(f"Collected {i+1} samples")

    # Sample on a fixed grid: collect_all_metrics() itself blocks for ~0.2s,
    # so sleeping a flat interval stretched the run well past DURATION.
    next_sample_at += SAMPLE_INTERVAL
    remaining = next_sample_at - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    else:
        next_sample_at = time.perf_counter()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(data)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} samples → {OUTPUT_PATH}")
