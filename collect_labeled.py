"""
collect_labeled.py
==================
Collect labeled training data with keyboard toggles.

Label convention (fixed)
------------------------
  0 = normal   (consistent with train_rl.py and split_data.py)
  1 = anomaly

Output: data/labeled_raw.csv  (feed into split_data.py next)

Controls
--------
  Press  a  → start anomaly period
  Press  n  → return to normal period
  Press  q  → quit early and save what has been collected
"""

import time
import signal
import sys
from pathlib import Path
import pandas as pd

try:
    import keyboard
except ImportError:
    print("Install the 'keyboard' library: pip install keyboard")
    sys.exit(1)

from data_collector import collect_all_metrics

DURATION = 300          # seconds (adjust upward for more data)
SAMPLE_INTERVAL = 1
OUTPUT_PATH = Path('data/labeled_raw.csv')
FEATURE_COLS = [
    'cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
    'gpu_percent', 'gpu_memory', 'gpu_temp',
]

print("=" * 55)
print("Labeled data collection")
print("  Press 'a' to START an anomaly (e.g. run stress test)")
print("  Press 'n' to STOP  the anomaly (back to normal)")
print("  Press 'q' to quit early and save")
print("=" * 55)

state = 0
data = []
labels = []
quit_flag = False

def on_a(e):
    global state
    if state != 1:
        state = 1
        print(f"\n  🔴 Anomaly mode ON")

def on_n(e):
    global state
    if state != 0:
        state = 0
        print(f"\n  🟢 Normal mode ON")

def on_q(e):
    global quit_flag
    quit_flag = True
    print(f"\n  Early exit requested.")

keyboard.on_press_key('a', on_a)
keyboard.on_press_key('n', on_n)
keyboard.on_press_key('q', on_q)

for i in range(DURATION):
    if quit_flag:
        break

    metrics = collect_all_metrics()
    row = {col: metrics.get(col, 0) for col in FEATURE_COLS}
    data.append(row)
    labels.append(state)
    time.sleep(SAMPLE_INTERVAL)

keyboard.unhook_all()
# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(data)
df['label'] = labels   # 0 = normal, 1 = anomaly
df.to_csv(OUTPUT_PATH, index=False)

n_anom = sum(labels)
n_norm = len(labels) - n_anom
print(f"\nSaved {len(df)} samples → {OUTPUT_PATH}")
print(f"  Normal : {n_norm}  |  Anomaly : {n_anom}")
print("Run split_data.py next to create train/test splits.")
