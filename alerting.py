"""
alerting.py
===========
Non-blocking alerting system for anomaly events.

Channels
--------
• Desktop notification  (via plyer — cross-platform)
• Terminal bell / beep  (fallback if plyer not installed)
• Alert history log     (in-memory ring buffer, also written to alerts.jsonl)

Usage
-----
    alerter = Alerter(cooldown_s=10)

    # Call from your producer loop:
    alerter.on_sample(pred, score, explanation, drift=False, metrics=metrics_dict)
"""

import json
import time
import threading
import os
from collections import deque
from pathlib import Path
from datetime import datetime


# ── Optional plyer ────────────────────────────────────────────────────────────
try:
    from plyer import notification as _plyer_notification
    _PLYER = True
except ImportError:
    _PLYER = False


class Alerter:
    """
    Parameters
    ----------
    cooldown_s  : minimum seconds between alerts of the same type
    history_len : number of recent alerts kept in memory
    log_path    : path to the JSONL alert log (None = no file logging)
    """

    def __init__(
        self,
        cooldown_s: float = 10.0,
        history_len: int = 200,
        log_path: str = 'alerts.jsonl',
    ):
        self.cooldown_s = cooldown_s
        self.history: deque = deque(maxlen=history_len)
        self.log_path = Path(log_path) if log_path else None
        self._last_anomaly_alert = 0.0
        self._last_drift_alert   = 0.0
        self._lock = threading.Lock()

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def on_sample(
        self,
        pred: int,
        score: float,
        explanation: list,
        drift: bool = False,
        metrics: dict = None,
    ) -> None:
        """
        Called once per monitoring sample.
        Fires alerts when warranted, respecting cooldowns.
        """
        now = time.time()

        if pred == -1 and (now - self._last_anomaly_alert) >= self.cooldown_s:
            self._fire('anomaly', score, explanation, metrics, now)
            with self._lock:
                self._last_anomaly_alert = now

        if drift and (now - self._last_drift_alert) >= self.cooldown_s * 6:
            self._fire('drift', score, explanation, metrics, now)
            with self._lock:
                self._last_drift_alert = now

    # ------------------------------------------------------------------ #
    def _fire(self, kind: str, score: float, explanation, metrics, ts: float):
        title, msg = self._format(kind, score, explanation)
        record = {
            'time': datetime.fromtimestamp(ts).isoformat(),
            'kind': kind,
            'score': round(score, 4),
            'explanation': explanation,
            'metrics': {k: round(v, 2) for k, v in (metrics or {}).items()
                        if isinstance(v, (int, float))},
        }

        with self._lock:
            self.history.append(record)

        # File log
        if self.log_path:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            except Exception as exc:
                print(f"[Alerter] Log write error: {exc}")

        # Desktop notification (non-blocking)
        threading.Thread(
            target=self._notify, args=(title, msg), daemon=True
        ).start()

        # Terminal output (always)
        icon = '🔴' if kind == 'anomaly' else '⚠️'
        print(f"\n{icon} ALERT [{kind.upper()}] {title}")
        if explanation:
            for line in (explanation or []):
                print(f"   {line}")
        print()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _format(kind: str, score: float, explanation) -> tuple:
        if kind == 'anomaly':
            title = f"Anomaly detected (score={score:.3f})"
            top = (explanation or ['unknown cause'])[0]
            msg = f"Top cause: {top}"
        else:
            title = "Score drift detected"
            msg = "Model may need retraining."
        return title, msg

    @staticmethod
    def _notify(title: str, msg: str) -> None:
        if _PLYER:
            try:
                _plyer_notification.notify(
                    title=title,
                    message=msg,
                    app_name='System Monitor',
                    timeout=5,
                )
                return
            except Exception:
                pass
        # Fallback: terminal bell
        try:
            print('\a', end='', flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def recent(self, n: int = 10) -> list:
        """Return the *n* most recent alert records."""
        with self._lock:
            return list(self.history)[-n:]

    def summary(self) -> str:
        with self._lock:
            hist = list(self.history)
        n_anom  = sum(1 for r in hist if r['kind'] == 'anomaly')
        n_drift = sum(1 for r in hist if r['kind'] == 'drift')
        return (
            f"Alerts in memory: {len(hist)} total | "
            f"{n_anom} anomaly | {n_drift} drift"
        )
