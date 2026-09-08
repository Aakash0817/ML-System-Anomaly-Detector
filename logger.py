"""
logger.py
=========
Thread-safe CSV logger with:
  • Context-manager support  (with CSVLogger(...) as log: ...)
  • atexit registration so the file is always flushed/closed
  • Full 7-feature row (all FEATURE_ORDER columns)
  • Rolling stats written to a separate summary file on close
"""

import csv
import atexit
import threading
import time
from pathlib import Path
import numpy as np

# Single source of truth: a second hand-maintained copy of this list could
# drift out of order and silently mislabel every column in the log.
from detectors.base import FEATURE_ORDER

AGGREGATE_COLUMNS = [
    'anomaly', 'score', 'inference_latency_ms', 'jitter_ms',
    'n_healthy', 'verdict_valid',
]


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def _fmt(value, spec: str = '') -> str:
    """Format a value, or emit an empty field for None.

    A missing reading is written as null, never as 0 — zero is a legitimate
    score and a legitimate latency.
    """
    if value is None:
        return ''
    if isinstance(value, bool):
        return '1' if value else '0'
    return format(value, spec) if spec else str(value)


class CSVLogger:
    """
    Write one row per sample to a CSV file.

    Parameters
    ----------
    filename    : path to the output CSV
    flush_every : flush to disk every N rows (default 10)
    """

    def __init__(self, filename: str = 'performance_log.csv', flush_every: int = 10,
                 detector_names=None):
        self.detector_names = list(detector_names or [])
        self._path = Path(filename)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._lock = threading.Lock()
        self._row_count = 0
        self._start_time = time.time()

        # Stats accumulators
        self._latencies: list = []
        self._scores: list = []
        self._anomaly_count = 0

        self._file = open(self._path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header())
        self._file.flush()

        # Ensure cleanup even on crash
        atexit.register(self.close)

    def header(self) -> list:
        cols = ['timestamp', 'elapsed_s'] + FEATURE_ORDER + AGGREGATE_COLUMNS
        for name in self.detector_names:
            slug = _slug(name)
            cols += [f'{slug}_score', f'{slug}_latency_ms',
                     f'{slug}_ok', f'{slug}_ph_stat']
        return cols

    # ------------------------------------------------------------------ #
    def log(
        self,
        timestamp: float,
        metrics: dict,
        anomaly=None,
        score=None,
        latency=None,
        jitter=None,
        n_healthy=None,
        verdict_valid=None,
        per_detector: dict = None,
    ) -> None:
        """Append one row. Thread-safe. None is written as an empty field."""
        elapsed = timestamp - self._start_time
        per_detector = per_detector or {}

        row = (
            [f"{timestamp:.3f}", f"{elapsed:.1f}"]
            + [_fmt(metrics.get(f)) for f in FEATURE_ORDER]
            + [_fmt(anomaly), _fmt(score, '.6f'), _fmt(latency, '.3f'),
               _fmt(jitter, '.3f'), _fmt(n_healthy), _fmt(verdict_valid)]
        )
        for name in self.detector_names:
            d = per_detector.get(name, {})
            row += [
                _fmt(d.get('score'), '.6f'),
                _fmt(d.get('latency'), '.3f'),
                _fmt(d.get('ok')),
                _fmt(d.get('ph_stat'), '.6f'),
            ]

        with self._lock:
            self._writer.writerow(row)
            self._row_count += 1
            if latency is not None:
                self._latencies.append(latency)
            if score is not None:
                self._scores.append(score)
            if anomaly == -1:
                self._anomaly_count += 1
            if self._row_count % self._flush_every == 0:
                self._file.flush()

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Flush, write summary, and close the file."""
        try:
            with self._lock:
                if self._file.closed:
                    return
                self._file.flush()
                self._file.close()
            self._write_summary()
        except Exception as exc:
            print(f"[CSVLogger] close() error: {exc}")

    def _write_summary(self) -> None:
        if self._row_count == 0:
            return
        summary_path = self._path.with_suffix('.summary.txt')
        try:
            lines = [
                f"Log file      : {self._path}",
                f"Total rows    : {self._row_count}",
                f"Anomalies     : {self._anomaly_count} ({self._anomaly_count / self._row_count:.1%})",
            ]
            if self._latencies:
                lines += [
                    f"Avg latency   : {np.mean(self._latencies):.2f} ms",
                    f"P99 latency   : {np.percentile(self._latencies, 99):.2f} ms",
                    f"Max latency   : {np.max(self._latencies):.2f} ms",
                ]
            if self._scores:
                lines += [
                    f"Mean score    : {np.mean(self._scores):.4f}",
                    f"Score std     : {np.std(self._scores):.4f}",
                ]
            summary_path.write_text("\n".join(lines) + "\n")
        except Exception as exc:
            print(f"[CSVLogger] Could not write summary: {exc}")

    # ------------------------------------------------------------------ #
    #  Context-manager support
    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return (
            f"<CSVLogger path={self._path} "
            f"rows={self._row_count} anomalies={self._anomaly_count}>"
        )
