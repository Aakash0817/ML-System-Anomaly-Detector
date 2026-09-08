import sys
import threading
import time
from collections import deque
from datetime import datetime

# ── Detector imports ──────────────────────────────────────────────────────────
# These must precede anything that loads Qt (PyQt5 itself and matplotlib's
# qt5agg backend): importing Qt first loads MSVC runtime DLLs that make
# TensorFlow's native library fail to initialise on Windows.
from detectors.isolation_forest import IsolationForestDetector
from detectors.oneclass_svm import OneClassSVMDetector
from detectors.local_outlier import LocalOutlierFactorDetector
from detectors.pca_reconstruction import PCADetector
from detectors.random_forest import RandomForestDetector
from detectors.xgboost_detector import XGBoostDetector
from detectors.rl_agent import RLAgentDetector

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

from anomaly_detector import DriftDetector
from data_collector import collect_all_metrics
from logger import CSVLogger
from seeds import set_global_seeds

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL    = 1.0
BUFFER_SIZE        = 100
MAX_ANOMALY_ROWS   = 200     # max rows kept in the anomaly log table
LOG_PATH           = 'logs/performance_log.csv'
NORMAL_DATA_PATH   = 'data/normal_training.csv'
LABELED_TRAIN_PATH = 'data/labeled_training.csv'

# Latency thresholds for colour coding (ms)
LAT_WARN  = 10.0
LAT_CRIT  = 50.0

# A detector that raises this many times in a row is taken out of service.
MAX_CONSECUTIVE_FAILURES = 3

# The verdict needs at least this many healthy detectors to mean anything.
MIN_HEALTHY_FOR_VOTE = 4


class DetectorState:
    """Health of one detector, preserved across pause/resume cycles."""

    def __init__(self, name):
        self.name = name
        self.consecutive_failures = 0
        self.disabled = False
        self.last_error = None

    def record_success(self):
        self.consecutive_failures = 0

    def record_failure(self, exc):
        self.consecutive_failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.disabled = True
        return self.disabled


# ─────────────────────────────────────────────────────────────────────────────
# DataCollector worker
# ─────────────────────────────────────────────────────────────────────────────
class DataCollector(QObject):
    """
    Runs in a background thread.

    Emits (timestamp, metrics, results, summary):
      results — one dict per detector:
        {'name', 'pred', 'score', 'latency', 'ok', 'disabled', 'error'}
        A detector that produced no reading has pred/score/latency None
        and ok False, so consumers can skip it rather than read a zero.
      summary — the per-sample verdict, computed once here:
        {'n_healthy', 'anomaly_votes', 'is_anomaly', 'agg_score',
         'agg_latency', 'valid', 'jitter_ms'}
        valid is False when fewer than MIN_HEALTHY_FOR_VOTE detectors
        reported; is_anomaly and agg_score are then None.
    """
    new_data = pyqtSignal(object)

    def __init__(self, detectors, state, logger=None, drift=None):
        super().__init__()
        self.logger    = logger
        self.drift     = drift or {}    # {name: DriftDetector}, owned by MainWindow
        self.detectors = detectors
        self.state     = state          # {name: DetectorState}, owned by MainWindow
        self.running   = True
        self._prev_sample_ts = None

    def run(self):
        while self.running:
            metrics = collect_all_metrics()
            results = []
            for name, det, _ in self.detectors:
                st = self.state[name]

                if st.disabled:
                    results.append(self._failed(name, st, disabled=True))
                    continue

                try:
                    pred, score, lat = det.predict(metrics)
                except Exception as exc:
                    now_disabled = st.record_failure(exc)
                    if now_disabled:
                        print(f"[{name}] DISABLED after "
                              f"{MAX_CONSECUTIVE_FAILURES} consecutive failures: "
                              f"{st.last_error}", flush=True)
                    else:
                        print(f"[{name}] predict failed "
                              f"({st.consecutive_failures}/"
                              f"{MAX_CONSECUTIVE_FAILURES}): {st.last_error}",
                              flush=True)
                    results.append(self._failed(name, st, disabled=now_disabled))
                    continue

                st.record_success()

                # One Page-Hinkley tracker per detector, on that detector's own
                # score stream. Raw statistic only: no threshold, no alert.
                ph = self.drift.get(name)
                ph_stat = None
                if ph is not None:
                    ph.update(score)
                    ph_stat = ph.statistic

                results.append({
                    'name':    name,
                    'pred':    pred,
                    'score':   score,
                    'latency': lat,
                    'ok':      True,
                    'disabled': False,
                    'error':   None,
                    'ph_stat': ph_stat,
                })
            now = time.time()
            if self._prev_sample_ts is None:
                jitter_ms = 0.0
            else:
                jitter_ms = (now - self._prev_sample_ts - SAMPLE_INTERVAL) * 1000.0
            self._prev_sample_ts = now

            summary = self._summarise(results, jitter_ms)
            self._write_log(now, metrics, results, summary)
            self.new_data.emit((now, metrics, results, summary))
            time.sleep(SAMPLE_INTERVAL)

    def _write_log(self, ts, metrics, results, summary):
        """Persist one row. Runs on the worker thread so disk I/O stays off
        the GUI thread; CSVLogger has its own lock."""
        if self.logger is None:
            return
        per_detector = {
            r['name']: {'score': r['score'], 'latency': r['latency'],
                        'ok': r['ok'], 'ph_stat': r.get('ph_stat')}
            for r in results
        }
        anomaly = None
        if summary['valid']:
            anomaly = -1 if summary['is_anomaly'] else 1
        try:
            self.logger.log(
                timestamp=ts, metrics=metrics, anomaly=anomaly,
                score=summary['agg_score'], latency=summary['agg_latency'],
                jitter=summary['jitter_ms'], n_healthy=summary['n_healthy'],
                verdict_valid=summary['valid'], per_detector=per_detector,
            )
        except Exception as exc:
            print(f"[CSVLogger] write failed: {exc}", flush=True)

    @staticmethod
    def _summarise(results, jitter_ms):
        """
        The 4-of-7 vote and the aggregate score, computed once per sample for
        every sample — not only for anomalies. Detectors that produced no
        reading are excluded, and the verdict is marked invalid when too few
        remain for it to mean anything.
        """
        healthy = [r for r in results if r['ok']]
        votes   = sum(1 for r in healthy if r['pred'] == -1)

        if len(healthy) < MIN_HEALTHY_FOR_VOTE:
            return {
                'n_healthy':     len(healthy),
                'anomaly_votes': votes,
                'is_anomaly':    None,
                'agg_score':     None,
                'agg_latency':   None,
                'valid':         False,
                'jitter_ms':     jitter_ms,
            }

        required = max(MIN_HEALTHY_FOR_VOTE, round(len(healthy) * 0.6))
        return {
            'n_healthy':     len(healthy),
            'anomaly_votes': votes,
            'is_anomaly':    votes >= required,
            'agg_score':     float(np.mean([r['score'] for r in healthy])),
            'agg_latency':   float(np.sum([r['latency'] for r in healthy])),
            'valid':         True,
            'jitter_ms':     jitter_ms,
        }

    @staticmethod
    def _failed(name, st, disabled):
        """Result for a detector that produced no reading. Values are None,
        never 0.0 — a zero score is a legitimate reading."""
        return {
            'name':    name,
            'pred':    None,
            'score':   None,
            'latency': None,
            'ok':      False,
            'disabled': disabled,
            'error':   st.last_error,
            'ph_stat': None,
        }

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
        self.detector_state = {name: DetectorState(name)
                               for name, _, _ in detectors}
        # One logger for the whole session. Created here, never inside the
        # collector: CSVLogger opens with mode 'w', so building a second one
        # on resume would truncate the log.
        self.logger = CSVLogger(LOG_PATH,
                                detector_names=[n for n, _, _ in detectors])
        # Scores differ in scale and meaning across detectors, so each gets its
        # own tracker rather than sharing one over a mixed stream.
        self.drift = {name: DriftDetector() for name, _, _ in detectors}
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
        self.lbl_detector_health = QLabel("Detectors: —")
        self.lbl_detector_health.setStyleSheet("color: #6b7280; font: 9pt 'Segoe UI';")

        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_clear)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_status)
        bar.addSpacing(16)
        bar.addWidget(self.lbl_anomaly_count)
        bar.addWidget(self.lbl_detector_health)
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
        Anomaly Log tab: updated to show full model names.
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

        fixed_cols  = ["Time", "CPU %", "CPU °C", "GPU %", "GPU °C",
                        "Agg Score", "Top Cause"]
        # Use full names for model columns
        model_cols  = model_names
        all_cols    = fixed_cols + model_cols

        self.anomaly_table = QTableWidget()
        self.anomaly_table.setColumnCount(len(all_cols))
        self.anomaly_table.setHorizontalHeaderLabels(all_cols)
        self.anomaly_table.setRowCount(0)

        # Use ResizeToContents to ensure full visibility of names
        self.anomaly_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Set the "Top Cause" (index 6) to Stretch for better layout
        self.anomaly_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)

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
    # Master refresh 
    # ─────────────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self.buffer:
            return
        data_list  = list(self.buffer)
        times      = [d[0] for d in data_list]
        t0         = times[0]
        rel        = [t - t0 for t in times]
        active_tab = self.tabs.currentIndex()

        self._update_system(data_list, rel)

        if active_tab == 1:
            self._update_cores(data_list, rel)
        elif active_tab == 2:
            self._update_models(data_list, rel)
        elif active_tab == 3:
            self._update_latency(data_list)
        elif active_tab == 4:
            self._update_anomaly_table()

        total = len(self.anomaly_log)
        self.lbl_anomaly_count.setText(f"Anomalies: {total}")
        self._refresh_detector_health()

    # ─────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ─────────────────────────────────────────────────────────────────────

    def _on_new_data(self, data):
        ts, metrics, results, summary = data
        self.buffer.append(data)

        for r in results:
            if r['ok']:
                self._latency_hist[r['name']].append(r['latency'])

        # The vote is computed once in the collector; do not recompute it here.
        if summary['is_anomaly']:
            agg_score = summary['agg_score']
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
                'results':     results,
            }
            self.anomaly_log.insert(0, record)
            if len(self.anomaly_log) > MAX_ANOMALY_ROWS:
                self.anomaly_log.pop()

    def _refresh_detector_health(self):
        """Name every detector that is failing or out of service."""
        disabled = [n for n, st in self.detector_state.items() if st.disabled]
        failing  = [n for n, st in self.detector_state.items()
                    if not st.disabled and st.consecutive_failures > 0]
        total    = len(self.detector_state)
        healthy  = total - len(disabled) - len(failing)

        parts = [f"Detectors: {healthy}/{total} OK"]
        if failing:
            parts.append("failing: " + ", ".join(sorted(failing)))
        if disabled:
            parts.append("DISABLED: " + ", ".join(sorted(disabled)))
        self.lbl_detector_health.setText("   |   ".join(parts))

        if disabled:
            colour = "#dc2626"
        elif failing:
            colour = "#d97706"
        else:
            colour = "#6b7280"
        self.lbl_detector_health.setStyleSheet(
            f"color: {colour}; font: 9pt 'Segoe UI'; font-weight: bold;"
        )

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
                if r['name'] == name and r['ok']
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
        lat_dict    = {r['name']: r['latency'] for r in latest_results}
        failed_dict = {r['name']: (not r['ok']) for r in latest_results}
        names       = [d[0] for d in self.detectors]

        max_lat = 0.001
        for i, name in enumerate(names):
            if failed_dict.get(name):
                # No reading this sample: no bar, greyed out.
                self._lat_bars[i].set_width(0)
                self._lat_bars[i].set_color('#9ca3af')
                continue
            lat = lat_dict.get(name) or 0
            self._lat_bars[i].set_width(lat)
            max_lat = max(max_lat, lat)
            if lat > LAT_CRIT:
                self._lat_bars[i].set_color('#dc2626')
            elif lat > LAT_WARN:
                self._lat_bars[i].set_color('#f59e0b')
            else:
                self._lat_bars[i].set_color('#16a34a')

        self.ax_lat_bar.set_xlim(0, max_lat * 1.2)
        self.canvas_lat_bar.draw_idle()

        for name, line in self._lat_history_lines.items():
            hist = list(self._latency_hist[name])
            if hist:
                line.set_data(range(len(hist)), hist)
        all_vals = [v for h in self._latency_hist.values()
                    for v in h if v is not None and v > 0]
        if all_vals:
            self.ax_lat_line.set_xlim(0, BUFFER_SIZE)
            self.ax_lat_line.set_ylim(0, max(all_vals) * 1.15)
        self.canvas_lat_line.draw_idle()

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
        """Rebuild anomaly table rows with full model visibility."""
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

            self.anomaly_table.setItem(row, 0, _c(rec['time'], Qt.AlignCenter, bold=True))
            self.anomaly_table.setItem(row, 1, _c(f"{rec['cpu_percent']:.1f}%"))
            self.anomaly_table.setItem(row, 2, _c(f"{rec['cpu_temp']:.1f}°C"))
            self.anomaly_table.setItem(row, 3, _c(f"{rec['gpu_percent']:.1f}%"))
            self.anomaly_table.setItem(row, 4, _c(f"{rec['gpu_temp']:.1f}°C"))

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

            causes = []
            if rec['cpu_percent'] > 70:
                causes.append(f"cpu:{rec['cpu_percent']:.0f}%")
            if rec.get('cpu_freq', 9999) < 1500:
                causes.append(f"freq:{rec['cpu_freq']:.0f}MHz")
            if rec.get('cpu_memory', 0) > 75:
                causes.append(f"ram:{rec['cpu_memory']:.0f}%")
            if rec['cpu_temp'] > 65:
                causes.append(f"cpu_temp:{rec['cpu_temp']:.0f}°C")
            if rec['gpu_percent'] > 60:
                causes.append(f"gpu:{rec['gpu_percent']:.0f}%")

            if not causes:
                # Use full names for flagging models instead of .split()[0]
                flagging = [
                    r['name']
                    for r in rec['results']
                    if r['pred'] == -1
                ]
                if flagging:
                    causes.append(f"models: {', '.join(flagging)}")

            cause_txt = "  |  ".join(causes) if causes else "—"
            self.anomaly_table.setItem(row, 6, _c(cause_txt, Qt.AlignLeft))

            for mi, mname in enumerate(model_names):
                r = results_map.get(mname)
                col = 7 + mi
                if r:
                    cell = _c(f"{r['score']:.5f}")
                    if r['pred'] == -1:
                        cell.setBackground(QBrush(QColor("#fee2e2")))
                        cell.setForeground(QBrush(QColor("#991b1b")))
                    else:
                        cell.setBackground(QBrush(QColor("#f0fdf4")))
                        cell.setForeground(QBrush(QColor("#166534")))
                    self.anomaly_table.setItem(row, col, cell)

        self.anomaly_table.setSortingEnabled(True)

        total = len(self.anomaly_log)
        total_samples = len(self.buffer)
        rate = (total / total_samples * 100) if total_samples > 0 else 0.0
        last_time = self.anomaly_log[0]['time'] if self.anomaly_log else "—"

        self.lbl_total_anomalies.setText(f"Total anomalies: {total}")
        self.lbl_last_anomaly.setText(f"Last anomaly: {last_time}")
        self.lbl_anomaly_rate.setText(f"Rate: {rate:.1f}%")

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

    def _start_collector(self):
        self.collector = DataCollector(self.detectors, self.detector_state,
                                       logger=self.logger, drift=self.drift)
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
            self.lbl_status.setStyleSheet("color:#d97706; font:9pt 'Segoe UI'; font-weight:bold;")
            self.collecting = False
        else:
            self._start_collector()
            self.btn_pause.setText("⏸  Pause")
            self.lbl_status.setText("● Collecting")
            self.lbl_status.setStyleSheet("color:#16a34a; font:9pt 'Segoe UI'; font-weight:bold;")
            self.collecting = True

    def clear_anomaly_log(self):
        self.anomaly_log.clear()
        self.anomaly_table.setRowCount(0)
        self.lbl_total_anomalies.setText("Total anomalies: 0")
        self.lbl_last_anomaly.setText("Last anomaly: —")
        self.lbl_anomaly_rate.setText("Rate: 0.0%")
        self.lbl_anomaly_count.setText("Anomalies: 0")

    def closeEvent(self, event):
        self.timer.stop()
        self.collector.stop()
        # Join before closing: stop() only sets a flag, so the worker may still
        # be mid-sample and would otherwise write to a closed file.
        thread = getattr(self, '_coll_thread', None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=SAMPLE_INTERVAL + 2.0)
        self.logger.close()
        event.accept()

SELFTEST_SECONDS = 30


def build_detectors():
    """Construct and train the detector registry."""
    normal_df        = pd.read_csv(NORMAL_DATA_PATH)
    labeled_train_df = pd.read_csv(LABELED_TRAIN_PATH)
    labeled_train_df['label'] = labeled_train_df['label'].apply(lambda x: 1 if x == 0 else -1)
    features = ['cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
                'gpu_percent', 'gpu_memory', 'gpu_temp']

    detectors = [
        ('Isolation Forest',     IsolationForestDetector(),    False),
        ('One-Class SVM',        OneClassSVMDetector(),        False),
        ('Local Outlier Factor', LocalOutlierFactorDetector(), False),
        ('PCA Reconstruction',   PCADetector(),                False),
        ('Random Forest',        RandomForestDetector(),       True),
        ('XGBoost',              XGBoostDetector(),            True),
        ('RL Agent',             RLAgentDetector(),            True),
    ]
    for name, det, needs_labels in detectors:
        if needs_labels:
            det.train(labeled_train_df[features], labeled_train_df['label'])
        else:
            det.train(normal_df[features])
    return detectors


def run_selftest(seconds: int = SELFTEST_SECONDS) -> int:
    """
    Collect for *seconds* with no GUI, then check the log is real data.

    Returns a process exit code: 0 if every check passes, 1 otherwise.
    """
    import csv as _csv

    set_global_seeds()
    detectors = build_detectors()
    names     = [n for n, _, _ in detectors]

    state  = {n: DetectorState(n) for n in names}
    drift  = {n: DriftDetector() for n in names}
    logger = CSVLogger(LOG_PATH, detector_names=names)

    collector = DataCollector(detectors, state, logger=logger, drift=drift)
    thread    = threading.Thread(target=collector.run, daemon=True)

    print(f"[selftest] collecting for {seconds}s ...", flush=True)
    thread.start()
    time.sleep(seconds)
    collector.stop()
    thread.join(timeout=SAMPLE_INTERVAL + 2.0)
    logger.close()

    with open(LOG_PATH, newline='', encoding='utf-8') as fh:
        rows = list(_csv.DictReader(fh))

    failures = []

    def check(label, ok, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              f"{'  — ' + detail if detail else ''}", flush=True)
        if not ok:
            failures.append(label)

    check('log has rows', len(rows) > 0, f'{len(rows)} rows')
    if not rows:
        print('[selftest] FAILED: no rows to inspect', flush=True)
        return 1

    def col(name):
        return [float(r[name]) for r in rows if r.get(name) not in (None, '')]

    lat = col('inference_latency_ms')
    check('latencies are not all identical', len(set(lat)) > 1,
          f'{len(set(lat))} distinct of {len(lat)}')

    sc = col('score')
    check('aggregate score is not constant', len(set(sc)) > 1,
          f'{len(set(sc))} distinct of {len(sc)}')

    for n in names:
        slug = _slug_name(n)
        vals = col(f'{slug}_score')
        if not vals:
            check(f'{n}: reported at least one score', False, 'no readings')
            continue
        variance = max(vals) - min(vals)
        check(f'{n}: score variance > 0', variance > 0,
              f'range {variance:.6g} over {len(vals)} samples')

    print('')
    verdict = 'PASSED' if not failures else 'FAILED'
    print(f"[selftest] {verdict} - {len(failures)} check(s) failed", flush=True)
    return 1 if failures else 0


def _slug_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


if __name__ == "__main__":
    if '--selftest' in sys.argv:
        sys.exit(run_selftest())

    # Detectors are trained at startup below, so seed first.
    set_global_seeds()

    normal_df        = pd.read_csv(NORMAL_DATA_PATH)
    labeled_train_df = pd.read_csv(LABELED_TRAIN_PATH)
    labeled_train_df['label'] = labeled_train_df['label'].apply(lambda x: 1 if x == 0 else -1)
    features = ['cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp', 'gpu_percent', 'gpu_memory', 'gpu_temp']

    DETECTORS = [
        ('Isolation Forest',     IsolationForestDetector(),    False),
        ('One-Class SVM',        OneClassSVMDetector(),        False),
        ('Local Outlier Factor', LocalOutlierFactorDetector(), False),
        ('PCA Reconstruction',   PCADetector(),                False),
        ('Random Forest',        RandomForestDetector(),       True),
        ('XGBoost',              XGBoostDetector(),            True),
        ('RL Agent',             RLAgentDetector(),            True),
    ]

    for name, det, needs_labels in DETECTORS:
        if needs_labels:
            det.train(labeled_train_df[features], labeled_train_df['label'])
        else:
            det.train(normal_df[features])

    app    = QApplication(sys.argv)
    window = MainWindow(DETECTORS)
    window.show()
    sys.exit(app.exec_())