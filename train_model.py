import time
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from data_collector import collect_all_metrics

# ===== CONFIGURATION =====
DURATION = 600          # seconds (10 minutes) – adjust as needed
SAMPLE_INTERVAL = 1     # seconds
CONTAMINATION = 0.05    # expected proportion of anomalies (0.05 = 5%)
# =========================

print(f"Collecting training data for {DURATION} seconds ({DURATION/60:.1f} minutes)...")
print("Please use your computer normally (browse, watch videos, etc.) during this time.")
data = []
for i in range(int(DURATION / SAMPLE_INTERVAL)):
    metrics = collect_all_metrics()
    data.append(metrics)
    time.sleep(SAMPLE_INTERVAL)
    if (i+1) % 60 == 0:
        print(f"Collected {i+1} samples...")

# Convert to DataFrame
df = pd.DataFrame(data)
df = df.drop(columns=['timestamp'])   # remove timestamp column

# Train Isolation Forest
model = IsolationForest(contamination=CONTAMINATION, random_state=42)
model.fit(df)

# Save model
joblib.dump(model, 'isolation_forest.pkl')
print("Model saved as isolation_forest.pkl")