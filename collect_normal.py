import time
import pandas as pd
from data_collector import collect_all_metrics

DURATION = 1200  # 10 minutes – you can increase to 30 minutes for better baseline
SAMPLE_INTERVAL = 1

print(f"Collecting normal data for {DURATION} seconds. Please use your computer normally (no heavy stress).")
data = []
for i in range(DURATION):
    metrics = collect_all_metrics()
    data.append(metrics)
    time.sleep(SAMPLE_INTERVAL)
    if (i+1) % 60 == 0:
        print(f"Collected {i+1} samples")

df = pd.DataFrame(data)
df = df.drop(columns=['timestamp'])
df.to_csv('data/normal_training.csv', index=False)
print("Saved normal_training.csv to data/ folder.")