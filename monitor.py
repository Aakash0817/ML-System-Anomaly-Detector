import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QTabWidget, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QFrame, QSplitter, QAbstractItemView
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QColor, QFont, QBrush

# ── Detector imports ──────────────────────────────────────────────────────────
from detectors.isolation_forest import IsolationForestDetector
from detectors.oneclass_svm import OneClassSVMDetector
from detectors.local_outlier import LocalOutlierFactorDetector
from detectors.pca_reconstruction import PCADetector
from detectors.random_forest import RandomForestDetector
from detectors.xgboost_detector import XGBoostDetector
from detectors.rl_agent import RLAgentDetector
from data_collector import collect_all_metrics

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL    = 1.0
BUFFER_SIZE        = 100
MAX_ANOMALY_ROWS   = 200     # max rows kept in the anomaly log table
NORMAL_DATA_PATH   = 'data/normal_training.csv'
LABELED_TRAIN_PATH = 'data/labeled_training.csv'

# Latency thresholds for colour coding (ms)
LAT_WARN  = 10.0
LAT_CRIT  = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# DataCollector worker
# ─────────────────────────────────────────────────────────────────────────────
class DataCollector(QObject):
    """
    Runs in a background thread.
    Emits (timestamp, metrics, results) where results is a list of dicts:
      {'name', 'pred', 'score', 'latency'}
    """
    new_data = pyqtSignal(object)

    def __init__(self, detectors):
        super().__init__()
        self.detectors = detectors
        self.running   = True

    def run(self):
        while self.running:
            metrics = collect_all_metrics()
            results = []
            for name, det, _ in self.detectors:
                try:
                    pred, score, lat = det.predict(metrics)
                except Exception as exc:
                    print(f"[{name}] predict error: {exc}")
                    pred, score, lat = 1, 0.0, 0.0
                results.append({
                    'name':    name,
                    'pred':    pred,
                    'score':   score,
                    'latency': lat,
                })
            self.new_data.emit((time.time(), metrics, results))
            time.sleep(SAMPLE_INTERVAL)

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: styled section label
# ─────────────────────────────────────────────────────────────────────────────
def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont('Segoe UI', 9, QFont.Bold))
    lbl.setStyleSheet("color: #444; padding: 2px 0 2px 0;")
    return lbl


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #ddd;")
    return line


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, detectors):
        super().__init__()
        self.detectors     = detectors          # [(name, det, needs_labels), ...]
        self.buffer        = deque(maxlen=BUFFER_SIZE)
        self.anomaly_log   = []                 # list of dicts for Anomaly tab
        self._latency_hist = {                  # rolling history per model
            name: deque(maxlen=BUFFER_SIZE)
            for name, _, _ in detectors
        }

        self.setWindowTitle("ML System Monitor")
        self.setGeometry(80, 80, 1500, 950)
        self.setStyleSheet("""
            QMainWindow { background: #f5f5f5; }
            QTabWidget::pane { border: 1px solid #ccc; background: #fff; }
            QTabBar::tab {
                background: #e8e8e8; border: 1px solid #ccc;
                padding: 6px 16px; font: 9pt 'Segoe UI';
            }
            QTabBar::tab:selected { background: #fff; border-bottom: none; font-weight: bold; }
            QPushButton {
                background: #2563eb; color: white; border: none;
                padding: 6px 18px; border-radius: 4px; font: 9pt 'Segoe UI';
            }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton#danger { background: #dc2626; }
            QPushButton#danger:hover { background: #b91c1c; }
            QTableWidget {
                gridline-color: #e5e7eb; font: 8pt 'Segoe UI';
                alternate-background-color: #f9fafb;
            }
            QHeaderView::section {
                background: #f3f4f6; font: 8pt 'Segoe UI';
                font-weight: bold; border: 1px solid #e5e7eb; padding: 4px;
            }
        """)

        # ── Central layout ────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        # ── Top control bar ───────────────────────────────────────────────
        bar = QHBoxLayout()
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self.toggle_collection)
        self.btn_clear = QPushButton("🗑  Clear Log")
        self.btn_clear.setObjectName("danger")
        self.btn_clear.clicked.connect(self.clear_anomaly_log)

        self.lbl_status = QLabel("● Collecting")
        self.lbl_status.setStyleSheet("color: #16a34a; font: 9pt 'Segoe UI'; font-weight: bold;")
        self.lbl_anomaly_count = QLabel("Anomalies: 0")
        self.lbl_anomaly_count.setStyleSheet("color: #dc2626; font: 9pt 'Segoe UI';")

        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_clear)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_status)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_anomaly_count)
        bar.addStretch()
        root.addLayout(bar)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tab_system  = QWidget(); self.tabs.addTab(self.tab_system,  "📊  System")
        self.tab_cores   = QWidget(); self.tabs.addTab(self.tab_cores,   "🔲  Cores")
        self.tab_models  = QWidget(); self.tabs.addTab(self.tab_models,  "🤖  Models")
        self.tab_latency = QWidget(); self.tabs.addTab(self.tab_latency, "⚡  Latency")
        self.tab_anomaly = QWidget(); self.tabs.addTab(self.tab_anomaly, "🚨  Anomaly Log")

        self._setup_system_tab()
        self._setup_cores_tab()
        self._setup_models_tab()
        self._setup_latency_tab()
        self._setup_anomaly_tab()

        # ── Start collector ───────────────────────────────────────────────
        self.collecting = True
        self._start_collector()

        # ── Refresh timer ─────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(200)

    # ─────────────────────────────────────────────────────────────────────
    # Tab setup
    # ─────────────────────────────────────────────────────────────────────

    def _setup_system_tab(self):
        layout = QVBoxLayout(self.tab_system)
        self.fig_system  = Figure(figsize=(10, 4), tight_layout=True)
        self.canvas_sys  = FigureCanvas(self.fig_system)
        layout.addWidget(self.canvas_sys)

        axes = self.fig_system.subplots(1, 2)
        self.ax_cpu, self.ax_gpu = axes

        for ax in axes:
            ax.set_ylim(0, 100)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)

        self.ax_cpu.set_ylabel("CPU % / Temp (°C)", fontsize=8)
        self.ax_gpu.set_ylabel("GPU % / Temp (°C)", fontsize=8)

        self.line_cpu_pct,  = self.ax_cpu.plot([], [], 'b-',  lw=1.2, label='CPU %')
        self.line_cpu_temp, = self.ax_cpu.plot([], [], 'r-',  lw=1.2, label='CPU Temp')
        self.line_gpu_pct,  = self.ax_gpu.plot([], [], 'g-',  lw=1.2, label='GPU %')
        self.line_gpu_temp, = self.ax_gpu.plot([], [], 'm-',  lw=1.2, label='GPU Temp')

        self.ax_cpu.legend(fontsize=7, loc='upper right')
        self.ax_gpu.legend(fontsize=7, loc='upper right')

        self.cpu_temp_txt = self.ax_cpu.text(
            0.02, 0.95, '', transform=self.ax_cpu.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='#fff3cd', alpha=0.8)
        )
        self.gpu_temp_txt = self.ax_gpu.text(
            0.02, 0.95, '', transform=self.ax_gpu.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='#d1fae5', alpha=0.8)
        )

    def _setup_cores_tab(self):
        layout = QVBoxLayout(self.tab_cores)
        self.fig_cores  = Figure(figsize=(10, 4), tight_layout=True)
        self.canvas_cores = FigureCanvas(self.fig_cores)
        layout.addWidget(self.canvas_cores)
        self.ax_cores = self.fig_cores.add_subplot(111)
        self.ax_cores.set_ylim(0, 100)
        self.ax_cores.set_xlabel("Time (s)", fontsize=8)
        self.ax_cores.set_ylabel("Per-Core CPU %", fontsize=8)
        self.ax_cores.set_title("P-cores (reds)  |  E-cores (blues)", fontsize=9)
        self.ax_cores.grid(True, alpha=0.3, linewidth=0.5)
        self.core_lines = []

    def _setup_models_tab(self):
        layout = QVBoxLayout(self.tab_models)
        n = len(self.detectors)
        cols = 2
        rows = (n + 1) // cols

        self.fig_models   = Figure(figsize=(10, rows * 1.4), tight_layout=True)
        self.canvas_models = FigureCanvas(self.fig_models)
        layout.addWidget(self.canvas_models)

        self.model_axes  = []
        self.model_lines = []
        for i in range(n):
            ax = self.fig_models.add_subplot(rows, cols, i + 1)
            ax.axhline(y=0, color='#9ca3af', linestyle='--', lw=0.6)
            ax.set_ylabel(self.detectors[i][0], fontsize=6,
                          rotation=0, labelpad=60, ha='right', va='center')
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.2, linewidth=0.4)
            if i < n - cols:
                ax.set_xticklabels([])
            line, = ax.plot([], [], lw=0.9,
                            color=plt.cm.tab10(i / max(n - 1, 1)))
            self.model_axes.append(ax)
            self.model_lines.append(line)

        for ax in self.model_axes:
            ax.sharex(self.model_axes[0])
        if self.model_axes:
            self.model_axes[-1].set_xlabel('Time (s)', fontsize=7)

    def _setup_latency_tab(self):
        """
        Latency tab: live bar chart of latest latency per model
        + rolling history line chart + detailed stats table.
        """
        layout = QVBoxLayout(self.tab_latency)
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # ── Top: bar + line charts side by side ──────────────────────────
        chart_widget = QWidget()
        chart_layout = QHBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        self.fig_lat_bar  = Figure(figsize=(5, 3), tight_layout=True)
        self.fig_lat_line = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas_lat_bar  = FigureCanvas(self.fig_lat_bar)
        self.canvas_lat_line = FigureCanvas(self.fig_lat_line)

        chart_layout.addWidget(self.canvas_lat_bar)
        chart_layout.addWidget(self.canvas_lat_line)
        splitter.addWidget(chart_widget)

        # Bar chart — latest latency
        self.ax_lat_bar = self.fig_lat_bar.add_subplot(111)
        self.ax_lat_bar.set_title("Latest Inference Latency (ms)", fontsize=9)
        self.ax_lat_bar.set_xlabel("Latency (ms)", fontsize=8)
        self.ax_lat_bar.tick_params(labelsize=7)
        self.ax_lat_bar.grid(True, axis='x', alpha=0.3)

        names = [d[0] for d in self.detectors]
        y_pos = range(len(names))
        self._lat_bars = self.ax_lat_bar.barh(
            y_pos, [0] * len(names),
            color=[plt.cm.tab10(i / max(len(names) - 1, 1))
                   for i in range(len(names))],
            alpha=0.85
        )
        self.ax_lat_bar.set_yticks(list(y_pos))
        self.ax_lat_bar.set_yticklabels(names, fontsize=7)

        # Line chart — rolling latency history per model
        self.ax_lat_line = self.fig_lat_line.add_subplot(111)
        self.ax_lat_line.set_title("Latency History (ms)", fontsize=9)
        self.ax_lat_line.set_xlabel("Sample", fontsize=8)
        self.ax_lat_line.set_ylabel("ms", fontsize=8)
        self.ax_lat_line.tick_params(labelsize=7)
        self.ax_lat_line.grid(True, alpha=0.3)
        self._lat_history_lines = {}
        for i, (name, _, _) in enumerate(self.detectors):
            line, = self.ax_lat_line.plot(
                [], [], lw=1.0, label=name,
                color=plt.cm.tab10(i / max(len(self.detectors) - 1, 1))
            )
            self._lat_history_lines[name] = line
        self.ax_lat_line.legend(fontsize=5, loc='upper right',
                                 ncol=2, framealpha=0.7)

        # ── Bottom: stats table ───────────────────────────────────────────
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.addWidget(_section_label("Per-Model Latency Statistics"))

        self.lat_stats_table = QTableWidget()
        self.lat_stats_table.setRowCount(len(self.detectors))
        self.lat_stats_table.setColumnCount(6)
        self.lat_stats_table.setHorizontalHeaderLabels(
            ["Model", "Latest (ms)", "Avg (ms)", "Min (ms)", "Max (ms)", "Status"]
        )
        self.lat_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lat_stats_table.setAlternatingRowColors(True)
        self.lat_stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lat_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        for i, (name, _, _) in enumerate(self.detectors):
            self.lat_stats_table.setItem(i, 0, QTableWidgetItem(name))
            for j in range(1, 6):
                self.lat_stats_table.setItem(i, j, QTableWidgetItem("—"))

        table_layout.addWidget(self.lat_stats_table)
        splitter.addWidget(table_widget)
        splitter.setSizes([400, 250])

    def _setup_anomaly_tab(self):
        """
        Anomaly Log tab: scrollable table of every anomaly event,
        with timestamp, scores, top explanation, and per-model votes.
        """
        layout = QVBoxLayout(self.tab_anomaly)

        # ── Summary bar ───────────────────────────────────────────────────
        summary_frame = QFrame()
        summary_frame.setStyleSheet(
            "background:#fef2f2; border:1px solid #fca5a5; border-radius:6px; padding:4px;"
        )
        summary_layout = QHBoxLayout(summary_frame)

        self.lbl_total_anomalies = QLabel("Total anomalies: 0")
        self.lbl_last_anomaly    = QLabel("Last anomaly: —")
        self.lbl_anomaly_rate    = QLabel("Rate: 0.0%")

        for lbl in (self.lbl_total_anomalies, self.lbl_last_anomaly, self.lbl_anomaly_rate):
            lbl.setStyleSheet("color:#991b1b; font: 9pt 'Segoe UI'; font-weight: bold;")
            summary_layout.addWidget(lbl)
        summary_layout.addStretch()
        layout.addWidget(summary_frame)

        layout.addWidget(_separator())
        layout.addWidget(_section_label("Anomaly Event Log  (newest at top)"))

        # ── Anomaly table ─────────────────────────────────────────────────
        n_models = len(self.detectors)
        model_names = [d[0] for d in self.detectors]

        # Fixed columns: Time | CPU% | CPU°C | GPU% | GPU°C | Score | Top Cause
        # Then one column per model showing its individual score
        fixed_cols  = ["Time", "CPU %", "CPU °C", "GPU %", "GPU °C",
                        "Agg Score", "Top Cause"]
        model_cols  = [f"{n[:6]}…" if len(n) > 8 else n for n in model_names]
        all_cols    = fixed_cols + model_cols

        self.anomaly_table = QTableWidget()
        self.anomaly_table.setColumnCount(len(all_cols))
        self.anomaly_table.setHorizontalHeaderLabels(all_cols)
        self.anomaly_table.setRowCount(0)

        # --- FIX 1: Use Stretch for equal column widths ---
        self.anomaly_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # --- Other table properties ---
        self.anomaly_table.setAlternatingRowColors(True)
        self.anomaly_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.anomaly_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.anomaly_table.setSortingEnabled(True)
        self.anomaly_table.setWordWrap(False)

        layout.addWidget(self.anomaly_table)

        # ── Export button ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_export = QPushButton("💾  Export to CSV")
        self.btn_export.clicked.connect(self._export_anomaly_csv)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    # ─────────────────────────────────────────────────────────────────────
    # Collector management
    # ─────────────────────────────────────────────────────────────────────

    def _start_collector(self):
        self.collector = DataCollector(self.detectors)
        self.collector.new_data.connect(self._on_new_data)
        self._coll_thread = threading.Thread(
            target=self.collector.run, daemon=True
        )
        self._coll_thread.start()

    def toggle_collection(self):
        if self.collecting:
            self.collector.stop()
            self.btn_pause.setText("▶  Resume")
            self.lbl_status.setText("● Paused")
            self.lbl_status.setStyleSheet(
                "color:#d97706; font:9pt 'Segoe UI'; font-weight:bold;"
            )
            self.collecting = False
        else:
            self._start_collector()
            self.btn_pause.setText("⏸  Pause")
            self.lbl_status.setText("● Collecting")
            self.lbl_status.setStyleSheet(
                "color:#16a34a; font:9pt 'Segoe UI'; font-weight:bold;"
            )
            self.collecting = True

    def clear_anomaly_log(self):
        self.anomaly_log.clear()
        self.anomaly_table.setRowCount(0)
        self.lbl_total_anomalies.setText("Total anomalies: 0")
        self.lbl_last_anomaly.setText("Last anomaly: —")
        self.lbl_anomaly_rate.setText("Rate: 0.0%")
        self.lbl_anomaly_count.setText("Anomalies: 0")

    # ─────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ─────────────────────────────────────────────────────────────────────

    def _on_new_data(self, data):
        """Receive (timestamp, metrics, results) from collector thread."""
        ts, metrics, results = data
        self.buffer.append(data)

        # Update latency history
        for r in results:
            self._latency_hist[r['name']].append(r['latency'])

        # Check if any model flagged anomaly — use supermajority vote (>=60%)
        # to reduce false positives from over-sensitive models like LOF
        anomaly_votes = sum(1 for r in results if r['pred'] == -1)
        is_anomaly    = anomaly_votes >= max(4, round(len(results) * 0.6))

        if is_anomaly:
            # Build aggregate score (mean of individual scores)
            agg_score = np.mean([r['score'] for r in results])
            record = {
                'time':        datetime.fromtimestamp(ts).strftime('%H:%M:%S'),
                'timestamp':   ts,
                'cpu_percent': metrics.get('cpu_percent', 0),
                'cpu_freq':    metrics.get('cpu_freq', 0),
                'cpu_memory':  metrics.get('cpu_memory', 0),
                'cpu_temp':    metrics.get('cpu_temp', 0),
                'gpu_percent': metrics.get('gpu_percent', 0),
                'gpu_memory':  metrics.get('gpu_memory', 0),
                'gpu_temp':    metrics.get('gpu_temp', 0),
                'agg_score':   agg_score,
                'results':     results,   # per-model scores
            }
            self.anomaly_log.insert(0, record)   # newest first
            if len(self.anomaly_log) > MAX_ANOMALY_ROWS:
                self.anomaly_log.pop()

    # ─────────────────────────────────────────────────────────────────────
    # Master refresh (called by QTimer every 200 ms)
    # ─────────────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self.buffer:
            return
        data_list  = list(self.buffer)
        times      = [d[0] for d in data_list]
        t0         = times[0]
        rel        = [t - t0 for t in times]
        active_tab = self.tabs.currentIndex()

        # Always update system tab (low cost)
        self._update_system(data_list, rel)

        if active_tab == 1:
            self._update_cores(data_list, rel)
        elif active_tab == 2:
            self._update_models(data_list, rel)
        elif active_tab == 3:
            self._update_latency(data_list)
        elif active_tab == 4:
            self._update_anomaly_table()

        # Control bar counters (always)
        total = len(self.anomaly_log)
        self.lbl_anomaly_count.setText(f"Anomalies: {total}")

    # ─────────────────────────────────────────────────────────────────────
    # Tab update helpers
    # ─────────────────────────────────────────────────────────────────────

    def _update_system(self, data_list, rel):
        cpu_pcts  = [d[1]['cpu_percent'] for d in data_list]
        cpu_tmps  = [d[1]['cpu_temp']    for d in data_list]
        gpu_pcts  = [d[1]['gpu_percent'] for d in data_list]
        gpu_tmps  = [d[1]['gpu_temp']    for d in data_list]

        self.line_cpu_pct.set_data(rel, cpu_pcts)
        self.line_cpu_temp.set_data(rel, cpu_tmps)
        self.line_gpu_pct.set_data(rel, gpu_pcts)
        self.line_gpu_temp.set_data(rel, gpu_tmps)

        if rel:
            self.ax_cpu.set_xlim(rel[0], rel[-1])
            self.ax_gpu.set_xlim(rel[0], rel[-1])
            self.cpu_temp_txt.set_text(f'CPU: {cpu_tmps[-1]:.1f}°C')
            self.gpu_temp_txt.set_text(f'GPU: {gpu_tmps[-1]:.1f}°C')

        self.canvas_sys.draw_idle()

    def _update_cores(self, data_list, rel):
        n_cores = len(data_list[0][1].get('per_cpu', []))
        if n_cores == 0:
            return

        if not self.core_lines:
            p_colors = plt.cm.Reds(np.linspace(0.45, 0.9, 8))
            e_colors = plt.cm.Blues(np.linspace(0.35, 0.85, 8))
            p_idx = e_idx = 0
            for ci in range(n_cores):
                ctype = data_list[0][1]['per_cpu'][ci]['type']
                if ctype == 'P':
                    color = p_colors[p_idx % len(p_colors)]
                    label = f'P{p_idx}'
                    p_idx += 1
                elif ctype == 'E':
                    color = e_colors[e_idx % len(e_colors)]
                    label = f'E{e_idx}'
                    e_idx += 1
                else:
                    color = 'gray'
                    label = f'C{ci}'
                ln, = self.ax_cores.plot([], [], color=color, lw=0.9, label=label)
                self.core_lines.append(ln)
            self.ax_cores.legend(loc='upper right', fontsize=5,
                                  ncol=min(4, n_cores), framealpha=0.7)

        for ci in range(n_cores):
            vals = [d[1]['per_cpu'][ci]['usage'] for d in data_list]
            self.core_lines[ci].set_data(rel, vals)

        latest  = data_list[-1][1]
        avg_p   = latest.get('avg_p_core', 0)
        avg_e   = latest.get('avg_e_core', 0)
        for txt in self.ax_cores.texts:
            txt.remove()
        self.ax_cores.text(
            0.02, 0.96,
            f'P-core avg: {avg_p:.1f}%  |  E-core avg: {avg_e:.1f}%',
            transform=self.ax_cores.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.6)
        )
        if rel:
            self.ax_cores.set_xlim(rel[0], rel[-1])
        self.canvas_cores.draw_idle()

    def _update_models(self, data_list, rel):
        for i, (name, _, _) in enumerate(self.detectors):
            scores = [
                r['score']
                for d in data_list
                for r in d[2]
                if r['name'] == name
            ]
            if not scores:
                continue
            self.model_lines[i].set_data(rel[:len(scores)], scores)
            mn, mx = min(scores), max(scores)
            margin = (mx - mn) * 0.12 if mx > mn else 0.5
            self.model_axes[i].set_ylim(mn - margin, mx + margin)

        if rel and self.model_axes:
            self.model_axes[0].set_xlim(rel[0], rel[-1])
        self.canvas_models.draw_idle()

    def _update_latency(self, data_list):
        if not data_list:
            return

        latest_results = data_list[-1][2]
        lat_dict = {r['name']: r['latency'] for r in latest_results}
        names    = [d[0] for d in self.detectors]

        # ── Bar chart ─────────────────────────────────────────────────────
        max_lat = 0.001
        for i, name in enumerate(names):
            lat = lat_dict.get(name, 0)
            self._lat_bars[i].set_width(lat)
            max_lat = max(max_lat, lat)
            # Colour by severity
            if lat > LAT_CRIT:
                self._lat_bars[i].set_color('#dc2626')
            elif lat > LAT_WARN:
                self._lat_bars[i].set_color('#f59e0b')
            else:
                self._lat_bars[i].set_color('#16a34a')

        self.ax_lat_bar.set_xlim(0, max_lat * 1.2)
        self.canvas_lat_bar.draw_idle()

        # ── Line chart (rolling history) ──────────────────────────────────
        for name, line in self._lat_history_lines.items():
            hist = list(self._latency_hist[name])
            if hist:
                line.set_data(range(len(hist)), hist)
        all_vals = [v for h in self._latency_hist.values() for v in h if v > 0]
        if all_vals:
            self.ax_lat_line.set_xlim(0, BUFFER_SIZE)
            self.ax_lat_line.set_ylim(0, max(all_vals) * 1.15)
        self.canvas_lat_line.draw_idle()

        # ── Stats table ───────────────────────────────────────────────────
        for i, name in enumerate(names):
            hist = list(self._latency_hist[name])
            lat  = lat_dict.get(name, 0)

            def _item(txt, align=Qt.AlignCenter):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(align)
                return it

            self.lat_stats_table.setItem(i, 0, _item(name, Qt.AlignLeft))
            self.lat_stats_table.setItem(i, 1, _item(f"{lat:.3f}"))
            self.lat_stats_table.setItem(i, 2, _item(f"{np.mean(hist):.3f}" if hist else "—"))
            self.lat_stats_table.setItem(i, 3, _item(f"{np.min(hist):.3f}"  if hist else "—"))
            self.lat_stats_table.setItem(i, 4, _item(f"{np.max(hist):.3f}"  if hist else "—"))

            # Status cell with colour
            if lat > LAT_CRIT:
                status, fg, bg = "SLOW",   "#7f1d1d", "#fee2e2"
            elif lat > LAT_WARN:
                status, fg, bg = "WARN",   "#78350f", "#fef3c7"
            else:
                status, fg, bg = "OK",     "#14532d", "#dcfce7"
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            status_item.setForeground(QBrush(QColor(fg)))
            status_item.setBackground(QBrush(QColor(bg)))
            status_item.setFont(QFont('Segoe UI', 8, QFont.Bold))
            self.lat_stats_table.setItem(i, 5, status_item)

    def _update_anomaly_table(self):
        """Rebuild anomaly table rows from self.anomaly_log."""
        n_models = len(self.detectors)
        model_names = [d[0] for d in self.detectors]

        self.anomaly_table.setSortingEnabled(False)
        self.anomaly_table.setRowCount(len(self.anomaly_log))

        for row, rec in enumerate(self.anomaly_log):
            results_map = {r['name']: r for r in rec['results']}

            def _c(txt, align=Qt.AlignCenter, bold=False):
                it = QTableWidgetItem(str(txt))
                it.setTextAlignment(align)
                if bold:
                    it.setFont(QFont('Segoe UI', 8, QFont.Bold))
                return it

            # Fixed columns: indices 0 to 6
            self.anomaly_table.setItem(row, 0, _c(rec['time'], Qt.AlignCenter, bold=True))
            self.anomaly_table.setItem(row, 1, _c(f"{rec['cpu_percent']:.1f}%"))
            self.anomaly_table.setItem(row, 2, _c(f"{rec['cpu_temp']:.1f}°C"))
            self.anomaly_table.setItem(row, 3, _c(f"{rec['gpu_percent']:.1f}%"))
            self.anomaly_table.setItem(row, 4, _c(f"{rec['gpu_temp']:.1f}°C"))

            # Aggregate score (column 5)
            score = rec['agg_score']
            score_item = _c(f"{score:.3f}", bold=True)
            if score < -0.5:
                score_item.setForeground(QBrush(QColor("#7f1d1d")))
                score_item.setBackground(QBrush(QColor("#fee2e2")))
            elif score < -0.2:
                score_item.setForeground(QBrush(QColor("#78350f")))
                score_item.setBackground(QBrush(QColor("#fef3c7")))
            else:
                score_item.setForeground(QBrush(QColor("#1e3a5f")))
                score_item.setBackground(QBrush(QColor("#dbeafe")))
            self.anomaly_table.setItem(row, 5, score_item)

            # Top cause (column 6) – hardware thresholds + model agreement fallback
            causes = []

            # CPU usage (flag if notably elevated)
            if rec['cpu_percent'] > 70:
                causes.append(f"cpu:{rec['cpu_percent']:.0f}%")
            elif rec['cpu_percent'] > 40:
                causes.append(f"cpu:{rec['cpu_percent']:.0f}%")
            # CPU frequency throttling
            if rec.get('cpu_freq', 9999) < 1500:
                causes.append(f"freq:{rec['cpu_freq']:.0f}MHz")
            # RAM pressure
            if rec.get('cpu_memory', 0) > 75:
                causes.append(f"ram:{rec['cpu_memory']:.0f}%")
            # CPU temperature
            if rec['cpu_temp'] > 65:
                causes.append(f"cpu_temp:{rec['cpu_temp']:.0f}°C")
            # GPU usage
            if rec['gpu_percent'] > 60:
                causes.append(f"gpu:{rec['gpu_percent']:.0f}%")
            # GPU memory
            if rec.get('gpu_memory', 0) > 70:
                causes.append(f"gpu_mem:{rec['gpu_memory']:.0f}%")
            # GPU temperature
            if rec['gpu_temp'] > 65:
                causes.append(f"gpu_temp:{rec['gpu_temp']:.0f}°C")

            # Fallback: if no hardware threshold breached, show which models voted anomaly
            if not causes:
                flagging = [
                    r['name'].split()[0]   # first word of model name for brevity
                    for r in rec['results']
                    if r['pred'] == -1
                ]
                if flagging:
                    causes.append(f"models: {', '.join(flagging)}")

            cause_txt = "  |  ".join(causes) if causes else "—"
            self.anomaly_table.setItem(row, 6, _c(cause_txt, Qt.AlignLeft))

            # Per-model score columns (starting at column 7)
            for mi, mname in enumerate(model_names):
                r = results_map.get(mname)
                col = 7 + mi
                if r:
                    cell = _c(f"{r['score']:.3f}")
                    if r['pred'] == -1:
                        cell.setBackground(QBrush(QColor("#fee2e2")))
                        cell.setForeground(QBrush(QColor("#991b1b")))
                    else:
                        cell.setBackground(QBrush(QColor("#f0fdf4")))
                        cell.setForeground(QBrush(QColor("#166534")))
                    self.anomaly_table.setItem(row, col, cell)

            # Optional: highlight entire row if critical
            if score < -0.5:
                for col in range(self.anomaly_table.columnCount()):
                    it = self.anomaly_table.item(row, col)
                    if it and col not in (5, 6) and not it.background().color().isValid():
                        it.setBackground(QBrush(QColor("#fff1f2")))

        self.anomaly_table.setSortingEnabled(True)

        # Update summary labels
        total = len(self.anomaly_log)
        total_samples = len(self.buffer)
        rate = (total / total_samples * 100) if total_samples > 0 else 0.0
        last_time = self.anomaly_log[0]['time'] if self.anomaly_log else "—"

        self.lbl_total_anomalies.setText(f"Total anomalies: {total}")
        self.lbl_last_anomaly.setText(f"Last anomaly: {last_time}")
        self.lbl_anomaly_rate.setText(f"Rate: {rate:.1f}%")
    # ─────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────

    def _export_anomaly_csv(self):
        if not self.anomaly_log:
            return
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Anomaly Log", "anomaly_log.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        rows = []
        for rec in self.anomaly_log:
            row = {
                'time':        rec['time'],
                'cpu_percent': rec['cpu_percent'],
                'cpu_temp':    rec['cpu_temp'],
                'gpu_percent': rec['gpu_percent'],
                'gpu_temp':    rec['gpu_temp'],
                'agg_score':   rec['agg_score'],
            }
            for r in rec['results']:
                row[f"{r['name']}_score"] = r['score']
                row[f"{r['name']}_pred"]  = r['pred']
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[Export] Saved {len(rows)} anomaly records → {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.timer.stop()
        self.collector.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("Loading datasets …")
    normal_df        = pd.read_csv(NORMAL_DATA_PATH)
    labeled_train_df = pd.read_csv(LABELED_TRAIN_PATH)
    labeled_train_df['label'] = labeled_train_df['label'].apply(
        lambda x: 1 if x == 0 else -1
    )
    features = [
        'cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
        'gpu_percent', 'gpu_memory', 'gpu_temp',
    ]

    DETECTORS = [
        ('Isolation Forest',     IsolationForestDetector(),    False),
        ('One-Class SVM',        OneClassSVMDetector(),        False),
        ('Local Outlier Factor', LocalOutlierFactorDetector(), False),
        ('PCA Reconstruction',   PCADetector(),                False),
        ('Random Forest',        RandomForestDetector(),       True),
        ('XGBoost',              XGBoostDetector(),            True),
        ('RL Agent',             RLAgentDetector(),            True),
    ]

    print("Training detectors …")
    for name, det, needs_labels in DETECTORS:
        print(f"  {name} …", end=' ', flush=True)
        try:
            if needs_labels:
                det.train(labeled_train_df[features], labeled_train_df['label'])
            else:
                det.train(normal_df[features])
            print("✓")
        except Exception as exc:
            print(f"✗  {exc}")
    print("Ready.\n")

    app    = QApplication(sys.argv)
    window = MainWindow(DETECTORS)
    window.show()
    sys.exit(app.exec_())