"""
split_data.py
=============
Splits data/labeled_raw.csv into labeled_training.csv and labeled_test.csv.

FIX vs original
---------------
Original code read labeled_test.csv, split it, and wrote the TEST portion
back to labeled_test.csv — destroying the training split on every run.

This version reads labeled_raw.csv (the full collected dataset) and writes
two non-overlapping files: labeled_training.csv and labeled_test.csv.
If labeled_raw.csv does not exist but labeled_test.csv does, it falls back
to that as the source (for backward compatibility) and issues a warning.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path('data')
RAW_PATH = DATA_DIR / 'labeled_raw.csv'
TRAIN_PATH = DATA_DIR / 'labeled_training.csv'
TEST_PATH = DATA_DIR / 'labeled_test.csv'
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ── Source selection ──────────────────────────────────────────────────────────
if RAW_PATH.exists():
    source = RAW_PATH
elif TEST_PATH.exists():
    print(
        f"WARNING: '{RAW_PATH}' not found. "
        f"Falling back to '{TEST_PATH}' as source. "
        "Rename collect_labeled.py output to 'labeled_raw.csv' to avoid this."
    )
    source = TEST_PATH
else:
    print(f"ERROR: No source data found. Run collect_labeled.py first.")
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(source)
print(f"Loaded {len(df)} rows from '{source}'")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# ── Stratified split ──────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['label'],          # preserve anomaly ratio in both splits
)

# ── Save ──────────────────────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print(f"Train : {len(train_df)} rows → {TRAIN_PATH}")
print(f"Test  : {len(test_df)} rows  → {TEST_PATH}")
print("Done.")
