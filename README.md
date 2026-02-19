# 🖥️ ML System Anomaly Detector

A real-time system monitoring tool that collects CPU/GPU hardware metrics and detects anomalies using an ensemble of machine learning models, with a live PyQt5 dashboard, desktop alerting, and detailed CSV logging.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-f7931e?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📊 Benchmark Results

All eight detectors were evaluated on the same labeled test set. Supervised models (XGBoost, Random Forest) achieved the highest F1 scores, while One-Class SVM led among unsupervised methods on ROC-AUC.

| Detector | F1 Score | ROC-AUC | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|
| **XGBoost** | **0.966** | 0.974 | 0.992 | 0.914 | ~5 |
| **Random Forest** | 0.937 | 0.975 | 0.981 | 0.897 | 41.4 |
| **RL Agent** | 0.929 | 0.056* | 0.963 | 0.897 | 63.1 |
| **Ensemble** | 0.898 | 0.876 | 0.883 | 0.914 | 78.7 |
| Isolation Forest | 0.857 | 0.882 | 0.836 | 0.879 | 4.98 |
| One-Class SVM | 0.841 | **0.930** | 0.779 | 0.914 | 1.01 |
| PCA Reconstruction | 0.812 | 0.635 | 0.700 | 0.966 | 0.143 |
| Local Outlier Factor | 0.733 | 0.500 | 0.657 | 0.828 | 37.5 |

> \* RL Agent ROC-AUC reflects the binary classification output format rather than a continuous anomaly score.

![Comparison Plot](comparison_plots.png)

---

## ✨ Features

- **Live hardware metrics** — CPU usage, frequency, memory, and temperature (via a non-blocking background WMI thread on Windows); GPU load, memory, and temperature via `GPUtil`
- **8 anomaly detectors** — Isolation Forest, One-Class SVM, Local Outlier Factor, PCA Reconstruction, Random Forest, XGBoost, a neural-network RL agent, and a voting Ensemble
- **PyQt5 dashboard** — Tabbed UI with system overview, per-core usage charts, model score timelines, latency stats, and a live anomaly event log
- **Smart alerting** — Desktop notifications via `plyer` with configurable cooldowns; falls back to terminal bell; all alerts persisted to `alerts.jsonl`
- **Online drift detection** — Page-Hinkley test on the score stream signals when the model distribution has shifted and retraining may be needed
- **Thread-safe CSV logging** — Every sample, prediction, score, latency, jitter, and drift flag is written to `performance_log.csv` with a summary on close
- **Modular detector design** — Add a new detector by inheriting from `BaseDetector` and registering it in `comparison.py` / `monitor.py`

---

## 🗂️ Project Structure

```
.
├── detectors/                  # All detector implementations
│   ├── base.py                 # BaseDetector + FEATURE_ORDER
│   ├── isolation_forest.py
│   ├── oneclass_svm.py
│   ├── local_outlier.py
│   ├── pca_reconstruction.py
│   ├── random_forest.py
│   ├── xgboost_detector.py
│   ├── rl_agent.py
│   └── ensemble_detector.py
│
├── data/                       # Training / test data (created by collection scripts)
│   ├── normal_training.csv
│   ├── labeled_raw.csv
│   ├── labeled_training.csv
│   └── labeled_test.csv
│
├── monitor.py                  # Main GUI application (entry point)
├── alerting.py                 # Non-blocking alert system
├── anomaly_detector.py         # Standalone IsolationForest wrapper with drift detection
├── logger.py                   # Thread-safe CSV logger
├── data_collector.py           # Hardware metric collection
├── process_tracker.py          # Top-process monitoring
│
├── collect_normal.py           # Step 1 – collect normal baseline data
├── collect_labeled.py          # Step 2 – collect labeled data (keyboard-toggled)
├── train_rl.py                 # Train the RL agent (neural network)
├── comparison.py               # Benchmark all detectors
│
├── rl_agent.keras              # Pre-trained neural network weights
├── rl_agent_scaler.pkl         # StandardScaler for RL agent
├── isolation_forest.pkl        # Pre-trained IsolationForest (optional)
│
├── comparison_results.csv      # Latest benchmark results
├── comparison_plots.png        # Benchmark visualisation
├── alerts.jsonl                # Runtime alert history
├── performance_log.csv         # Runtime sample log
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows only:** CPU temperature reading requires [OpenHardwareMonitor](https://openhardwaremonitor.org/) to be running and the `wmi` package: `pip install wmi`.

### 2. Collect normal baseline data (≥10 min recommended)

```bash
python collect_normal.py
```

Use your computer normally during collection — no stress tests. Saves to `data/normal_training.csv`.

### 3. Collect labeled data for supervised models

```bash
python collect_labeled.py
```

| Key | Action |
|-----|--------|
| `a` | Switch to **anomaly** mode (run a stress test now) |
| `n` | Return to **normal** mode |
| `q` | Stop early and save |

Saves to `data/labeled_raw.csv`. Then split into train/test:

```bash
python split_data.py
```

### 4. Train models

```bash
# Train IsolationForest on normal data only
python train_model.py

# Train the neural-network RL agent on labeled data
python train_rl.py
```

All other detectors (RF, XGBoost, LOF, PCA, SVM) are trained automatically at runtime.

### 5. Benchmark all detectors

```bash
python comparison.py
```

Produces `comparison_results.csv` and `comparison_plots.png`.

### 6. Launch the live monitor

```bash
python monitor.py
```

---

## 🧠 How It Works

### Data Collection

`data_collector.py` gathers seven core features every second:

| Feature | Source |
|---|---|
| `cpu_percent` | `psutil.cpu_percent()` |
| `cpu_freq` | `psutil.cpu_freq()` |
| `cpu_memory` | `psutil.virtual_memory().percent` |
| `cpu_temp` | WMI (background thread) / `psutil.sensors_temperatures()` |
| `gpu_percent` | `GPUtil` |
| `gpu_memory` | `GPUtil` |
| `gpu_temp` | `GPUtil` |

CPU temperature is read in a dedicated background thread (with its own `CoInitialize` on Windows) so it never blocks the main producer loop.

### Anomaly Detection Pipeline

```
Raw metrics → Feature vector (7 values)
           → Each detector's predict()      → pred ∈ {1, -1}, score, latency
           → AnomalyExplainer (z-scores)    → top-3 deviating features
           → DriftDetector (Page-Hinkley)   → drift flag
           → Alerter                        → desktop notification / log
           → CSVLogger                      → performance_log.csv
```

### Drift Detection

The `DriftDetector` runs a Page-Hinkley test on the anomaly score stream. When the cumulative deviation from the running mean exceeds a configurable threshold (default 50), a drift flag is raised and the alerting system fires a drift alert, suggesting retraining.

---

## ⚙️ Configuration

Key parameters are defined at the top of each module:

| File | Parameter | Default | Description |
|---|---|---|---|
| `alerting.py` | `cooldown_s` | `10` | Min seconds between anomaly alerts |
| `anomaly_detector.py` | `DriftDetector.threshold` | `50.0` | Page-Hinkley drift sensitivity |
| `logger.py` | `flush_every` | `10` | Rows between disk flushes |
| `collect_normal.py` | `DURATION` | `1200` | Seconds of normal data to collect |
| `collect_labeled.py` | `DURATION` | `300` | Seconds of labeled data to collect |
| `process_tracker.py` | `top_n` | `5` | Processes shown in the tracker |

---

## 📦 Requirements

```
psutil
GPUtil
numpy
pandas
scikit-learn
xgboost
tensorflow>=2.0
joblib
matplotlib
PyQt5
plyer          # optional – desktop notifications
keyboard       # optional – for collect_labeled.py
wmi            # optional – Windows CPU temperature
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-detector`
3. Add your detector in `detectors/` by subclassing `BaseDetector`
4. Register it in `comparison.py` and `monitor.py`
5. Open a pull request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
