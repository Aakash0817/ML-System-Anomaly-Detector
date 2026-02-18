import sys
import threading
import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QTabWidget, QLabel, QPushButton, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt   # <-- Qt added here
from PyQt5.QtGui import QColor

# Import all detector classes
from detectors.isolation_forest import IsolationForestDetector
from detectors.oneclass_svm import OneClassSVMDetector
from detectors.local_outlier import LocalOutlierFactorDetector
from detectors.pca_reconstruction import PCADetector
from detectors.random_forest import RandomForestDetector
from detectors.xgboost_detector import XGBoostDetector
from detectors.rl_agent import RLAgentDetector
from data_collector import collect_all_metrics

# ========== CONFIGURATION ==========
SAMPLE_INTERVAL = 1.0          # seconds
BUFFER_SIZE = 100               # samples kept for plotting
NORMAL_DATA_PATH = 'data/normal_training.csv'
LABELED_TRAIN_PATH = 'data/labeled_training.csv'

# ========== WORKER THREAD FOR DATA COLLECTION ==========
class DataCollector(QObject):
    """Runs in a separate thread to collect metrics and run inference."""
    new_data = pyqtSignal(object)  # emits (timestamp, metrics, results)

    def __init__(self, detectors):
        super().__init__()
        self.detectors = detectors  # list of (name, detector_instance, needs_labels)
        self.running = True

    def run(self):
        while self.running:
            metrics = collect_all_metrics()
            results = []
            for name, det, _ in self.detectors:
                try:
                    pred, score, lat = det.predict(metrics)
                except Exception as e:
                    print(f"Error in {name}: {e}")
                    score = 0
                    lat = 0
                results.append({'name': name, 'score': score, 'latency': lat})
            self.new_data.emit((time.time(), metrics, results))
            time.sleep(SAMPLE_INTERVAL)

    def stop(self):
        self.running = False

# ========== MAIN GUI WINDOW ==========
class MainWindow(QMainWindow):
    def __init__(self, detectors):
        super().__init__()
        self.detectors = detectors
        self.setWindowTitle("ML System Monitor - Small Multiples")
        self.setGeometry(100, 100, 1400, 900)

        # Data buffer (thread-safe via pyqtSignal)
        self.buffer = deque(maxlen=BUFFER_SIZE)

        # Central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Control bar (start/stop button)
        control_layout = QHBoxLayout()
        self.start_stop_btn = QPushButton("Pause")
        self.start_stop_btn.clicked.connect(self.toggle_collection)
        control_layout.addWidget(self.start_stop_btn)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: System Overview (CPU/GPU)
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "System")
        self.setup_system_tab()

        # Tab 2: Core Utilization (P/E cores)
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "Cores")
        self.setup_cores_tab()

        # Tab 3: Model Scores (small multiples grid) + Latency Table
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "Models")
        self.setup_models_tab()

        # Start data collection thread
        self.collector = DataCollector(detectors)
        self.thread = threading.Thread(target=self.collector.run, daemon=True)
        self.collector.new_data.connect(self.on_new_data)
        self.thread.start()
        self.collecting = True

        # Timer to refresh plots (every 50 ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)

    def toggle_collection(self):
        """Pause or resume data collection."""
        if self.collecting:
            self.collector.stop()
            self.start_stop_btn.setText("Resume")
            self.collecting = False
        else:
            self.collector = DataCollector(self.detectors)
            self.thread = threading.Thread(target=self.collector.run, daemon=True)
            self.collector.new_data.connect(self.on_new_data)
            self.thread.start()
            self.start_stop_btn.setText("Pause")
            self.collecting = True

    def setup_system_tab(self):
        layout = QVBoxLayout(self.tab1)
        self.fig_system = Figure(figsize=(8, 4))
        self.canvas_system = FigureCanvas(self.fig_system)
        layout.addWidget(self.canvas_system)
        self.ax_system = self.fig_system.add_subplot(111)
        self.ax_system.set_ylim(0, 100)
        self.ax_system.set_xlabel("Time (s)")
        self.ax_system.set_ylabel("Usage / Temp (°C)")
        self.line_cpu = self.ax_system.plot([], [], 'b-', label='CPU %')[0]
        self.line_cpu_temp = self.ax_system.plot([], [], 'r-', label='CPU Temp')[0]
        self.line_gpu = self.ax_system.plot([], [], 'g-', label='GPU %')[0]
        self.line_gpu_temp = self.ax_system.plot([], [], 'm-', label='GPU Temp')[0]
        self.ax_system.legend(fontsize=8)

    def setup_cores_tab(self):
        layout = QVBoxLayout(self.tab2)
        self.fig_cores = Figure(figsize=(8, 4))
        self.canvas_cores = FigureCanvas(self.fig_cores)
        layout.addWidget(self.canvas_cores)
        self.ax_cores = self.fig_cores.add_subplot(111)
        self.ax_cores.set_ylim(0, 100)
        self.ax_cores.set_xlabel("Time (s)")
        self.ax_cores.set_ylabel("Per‑Core CPU %")
        self.ax_cores.set_title("P‑cores (red shades), E‑cores (blue shades)")
        self.core_lines = []  # will be created when data arrives

    def setup_models_tab(self):
        layout = QVBoxLayout(self.tab3)
        
        # Create a figure with a grid of small subplots: 4 rows, 2 columns
        n_models = len(self.detectors)
        self.fig_models = Figure(figsize=(8, 6))
        self.canvas_models = FigureCanvas(self.fig_models)
        layout.addWidget(self.canvas_models)

        self.model_axes = []
        self.model_lines = []
        for i in range(n_models):
            ax = self.fig_models.add_subplot(4, 2, i+1)  # 4 rows, 2 cols
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            # Y-label is the model name, rotated for readability
            ax.set_ylabel(self.detectors[i][0], fontsize=5, rotation=0, labelpad=8, ha='right')
            ax.tick_params(labelsize=4)
            # Hide x-axis labels for all but bottom row (indices 5,6,7 in 0-based? Let's hide for all)
            if i < n_models - 2:  # bottom two rows have x-labels? Actually easier: set_xlabel only on bottom
                ax.set_xticklabels([])
            line, = ax.plot([], [], 'b-', linewidth=0.8)
            self.model_axes.append(ax)
            self.model_lines.append(line)

        # Share x-axis among all
        for ax in self.model_axes:
            ax.sharex(self.model_axes[0])
        self.model_axes[-1].set_xlabel('Time (s)', fontsize=6)  # bottom-most axis gets label
        self.fig_models.tight_layout()

        # ----- Latency Table (below the plot) -----
        table_label = QLabel("Latest Inference Latency (ms)")
        layout.addWidget(table_label)

        self.latency_table = QTableWidget()
        self.latency_table.setRowCount(n_models)
        self.latency_table.setColumnCount(2)
        self.latency_table.setHorizontalHeaderLabels(["Model", "Latency (ms)"])
        self.latency_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.latency_table.setMaximumHeight(150)  # Limit table height

        # Populate with initial data (model names, placeholder latency)
        for i, (name, _, _) in enumerate(self.detectors):
            self.latency_table.setItem(i, 0, QTableWidgetItem(name))
            self.latency_table.setItem(i, 1, QTableWidgetItem("0.0"))
            # Make model name non-editable
            self.latency_table.item(i, 0).setFlags(self.latency_table.item(i, 0).flags() & ~Qt.ItemIsEditable)

        layout.addWidget(self.latency_table)

    def on_new_data(self, data):
        """Receive data from collector thread and store in buffer."""
        self.buffer.append(data)

    def update_plots(self):
        """Called by QTimer – redraws all plots from buffer."""
        if not self.buffer:
            return

        data_list = list(self.buffer)
        times = [d[0] for d in data_list]
        t0 = times[0]
        rel_times = [t - t0 for t in times]

        # ----- System tab -----
        cpu_percents = [d[1]['cpu_percent'] for d in data_list]
        cpu_temps = [d[1]['cpu_temp'] for d in data_list]
        gpu_percents = [d[1]['gpu_percent'] for d in data_list]
        gpu_temps = [d[1]['gpu_temp'] for d in data_list]

        self.line_cpu.set_data(rel_times, cpu_percents)
        self.line_cpu_temp.set_data(rel_times, cpu_temps)
        self.line_gpu.set_data(rel_times, gpu_percents)
        self.line_gpu_temp.set_data(rel_times, gpu_temps)
        self.ax_system.set_xlim(rel_times[0], rel_times[-1])
        self.canvas_system.draw_idle()

        # ----- Cores tab -----
        n_cores = len(data_list[0][1]['per_cpu'])
        if len(self.core_lines) == 0:
            # Create core lines based on first sample
            p_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 4))
            e_colors = plt.cm.Blues(np.linspace(0.3, 0.8, 8))
            p_idx, e_idx = 0, 0
            for core_idx in range(n_cores):
                core_type = data_list[0][1]['per_cpu'][core_idx]['type']
                if core_type == 'P':
                    color = p_colors[p_idx % len(p_colors)]
                    label = f'P-core {p_idx//2}.{p_idx%2}'
                    p_idx += 1
                elif core_type == 'E':
                    color = e_colors[e_idx % len(e_colors)]
                    label = f'E-core {e_idx}'
                    e_idx += 1
                else:
                    color = 'gray'
                    label = f'Core {core_idx}'
                line, = self.ax_cores.plot([], [], color=color, linewidth=1, label=label)
                self.core_lines.append(line)
            self.ax_cores.legend(loc='upper right', fontsize=5, ncol=min(4, n_cores))

        # Update each core line
        for core_idx in range(n_cores):
            core_vals = [d[1]['per_cpu'][core_idx]['usage'] for d in data_list]
            self.core_lines[core_idx].set_data(rel_times, core_vals)

        # Show average P/E core utilization
        latest = data_list[-1][1]
        avg_p = latest.get('avg_p_core', 0)
        avg_e = latest.get('avg_e_core', 0)
        for txt in self.ax_cores.texts:
            txt.remove()
        self.ax_cores.text(0.02, 0.95, f'P‑core avg: {avg_p:.1f}% | E‑core avg: {avg_e:.1f}%',
                           transform=self.ax_cores.transAxes, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.ax_cores.set_xlim(rel_times[0], rel_times[-1])
        self.canvas_cores.draw_idle()

        # ----- Models tab (small multiples) -----
        # Update each model's line
        for i, (name, _, _) in enumerate(self.detectors):
            scores = []
            for d in data_list:
                for res in d[2]:
                    if res['name'] == name:
                        scores.append(res['score'])
                        break
            self.model_lines[i].set_data(rel_times, scores)
            # Auto-scale y individually
            if scores:
                min_s, max_s = min(scores), max(scores)
                margin = (max_s - min_s) * 0.1 if max_s > min_s else 0.5
                self.model_axes[i].set_ylim(min_s - margin, max_s + margin)
        # Set x-limits (shared)
        self.model_axes[0].set_xlim(rel_times[0], rel_times[-1])
        self.canvas_models.draw_idle()

        # ----- Update Latency Table (latest latencies) -----
        if data_list:
            latest_results = data_list[-1][2]  # list of dicts with 'name' and 'latency'
            # Build a dict for quick lookup
            lat_dict = {res['name']: res['latency'] for res in latest_results}
            for i, (name, _, _) in enumerate(self.detectors):
                lat = lat_dict.get(name, 0)
                item = QTableWidgetItem(f"{lat:.2f}")
                # Optional: color code based on latency (e.g., red if > some threshold)
                if lat > 50:
                    item.setForeground(QColor("red"))
                self.latency_table.setItem(i, 1, item)

    def closeEvent(self, event):
        self.collector.stop()
        event.accept()

# ========== MAIN ==========
if __name__ == "__main__":
    # Load training data
    print("Loading datasets...")
    normal_df = pd.read_csv(NORMAL_DATA_PATH)
    labeled_train_df = pd.read_csv(LABELED_TRAIN_PATH)
    labeled_train_df['label'] = labeled_train_df['label'].apply(lambda x: 1 if x == 0 else -1)
    features = ['cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
                'gpu_percent', 'gpu_memory', 'gpu_temp']

    # Define detectors (same as in live_comparison.py)
    DETECTORS = [
        ('Isolation Forest', IsolationForestDetector(), False),
        ('One-Class SVM', OneClassSVMDetector(), False),
        ('Local Outlier Factor', LocalOutlierFactorDetector(), False),
        ('PCA Reconstruction', PCADetector(), False),
        ('Random Forest', RandomForestDetector(), True),
        ('XGBoost', XGBoostDetector(), True),
        ('RL Agent', RLAgentDetector(), False),
    ]

    # Train all detectors
    print("Training detectors...")
    for name, det, needs_labels in DETECTORS:
        print(f"  Training {name}...")
        try:
            if needs_labels:
                det.train(labeled_train_df[features], labeled_train_df['label'])
            else:
                det.train(normal_df[features])
        except Exception as e:
            print(f"    Error training {name}: {e}")
    print("All detectors ready.\n")

    # Start GUI
    app = QApplication(sys.argv)
    window = MainWindow(DETECTORS)
    window.show()
    sys.exit(app.exec_())