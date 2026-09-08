import psutil
import time
import random
import threading

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


# ── Background temperature reader ─────────────────────────────────────────────
# Runs WMI in its own dedicated thread with its own CoInitialize.
# The producer loop just reads _last_cpu_temp without waiting.

_last_cpu_temp = None    # None until a read actually succeeds
_last_cpu_temp_time = 0.0
_temp_lock     = threading.Lock()
_temp_interval = 2.0     # read temperature every 2 seconds
TEMP_MAX_AGE_S = 10.0    # a reading older than this is reported as unavailable

# Each distinct failure reason is reported once, not once per sample.
_logged_reasons: set = set()
_log_lock = threading.Lock()


def _log_once(key: str, message: str) -> None:
    with _log_lock:
        if key in _logged_reasons:
            return
        _logged_reasons.add(key)
    print(message, flush=True)


def _temperature_worker():
    """
    Dedicated thread for WMI temperature reads.
    Has its own CoInitialize so it never blocks the producer.
    """
    global _last_cpu_temp, _last_cpu_temp_time

    try:
        import pythoncom
    except ImportError:
        _log_once('temp_pythoncom',
                  "[TempWorker] pywin32 not installed - CPU temperature unavailable.")
        return
    pythoncom.CoInitialize()

    wmi_conn = None
    try:
        import wmi
        wmi_conn = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        print("[TempWorker] WMI connected.", flush=True)
    except Exception as e:
        _log_once('temp_wmi_connect', f"[TempWorker] WMI connect failed: {e}")

    while True:
        temp = None

        # ── WMI read ──────────────────────────────────────────────────────
        if wmi_conn:
            try:
                cpu_temps = [
                    float(s.Value)
                    for s in wmi_conn.Sensor()
                    if s.SensorType == 'Temperature'
                    and any(k in s.Name.lower()
                            for k in ['cpu', 'core', 'package'])
                    and s.Value is not None
                    and float(s.Value) > 0
                ]
                if cpu_temps:
                    temp = max(cpu_temps)
            except Exception as e:
                _log_once('temp_wmi_read', f"[TempWorker] WMI read error: {e}")
                # Try to reconnect next cycle
                try:
                    import wmi
                    wmi_conn = wmi.WMI(namespace="root\\OpenHardwareMonitor")
                except Exception:
                    wmi_conn = None

        # ── psutil fallback ───────────────────────────────────────────────
        if temp is None:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    all_vals = [
                        e.current
                        for entries in temps.values()
                        for e in entries
                        if e.current and e.current > 0
                    ]
                    if all_vals:
                        temp = max(all_vals)
            except Exception:
                pass

        # ── Update shared value ───────────────────────────────────────────
        if temp is not None:
            with _temp_lock:
                _last_cpu_temp = temp
                _last_cpu_temp_time = time.time()
        else:
            _log_once('temp_no_source',
                      "[TempWorker] No CPU temperature source available.")

        time.sleep(_temp_interval)


def get_cpu_temperature():
    """
    Non-blocking. Returns the last good reading from the background thread,
    or None when no read has ever succeeded or the most recent one is older
    than TEMP_MAX_AGE_S. Never substitutes a placeholder value.
    """
    with _temp_lock:
        value, taken_at = _last_cpu_temp, _last_cpu_temp_time

    if value is None:
        _log_once('temp_never',
                  "[TempWorker] No CPU temperature reading yet - reporting None.")
        return None

    if time.time() - taken_at > TEMP_MAX_AGE_S:
        _log_once('temp_stale',
                  f"[TempWorker] CPU temperature older than {TEMP_MAX_AGE_S:.0f}s "
                  "- reporting None.")
        return None

    return value


# Start the background temperature thread immediately on import
_temp_thread = threading.Thread(target=_temperature_worker, daemon=True)
_temp_thread.start()


# ── CPU metrics ───────────────────────────────────────────────────────────────
def get_cpu_metrics():
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'cpu_freq':    psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        'cpu_memory':  psutil.virtual_memory().percent,
        'cpu_temp':    get_cpu_temperature(),   # instant, non-blocking
    }


# ── GPU metrics ───────────────────────────────────────────────────────────────
GPU_FIELDS = ('gpu_percent', 'gpu_memory', 'gpu_temp')


def _gpu_unavailable() -> dict:
    """No reading is available; report that rather than inventing zeros."""
    d = {f: None for f in GPU_FIELDS}
    d['gpu_available'] = False
    return d


def get_gpu_metrics():
    """
    Returns the three GPU fields plus gpu_available. On any failure the
    fields are None and gpu_available is False — never 0, which is a
    legitimate reading for an idle GPU.
    """
    if not GPUTIL_AVAILABLE:
        _log_once('gpu_import',
                  "[GPU] GPUtil not installed - GPU metrics unavailable.")
        return _gpu_unavailable()

    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        _log_once('gpu_query',
                  f"[GPU] GPUtil query failed ({e}) - GPU metrics unavailable.")
        return _gpu_unavailable()

    if not gpus:
        _log_once('gpu_none', "[GPU] No GPU detected - GPU metrics unavailable.")
        return _gpu_unavailable()

    try:
        gpu = gpus[0]
        return {
            'gpu_percent': gpu.load * 100,
            'gpu_memory':  gpu.memoryUtil * 100,
            'gpu_temp':    gpu.temperature,
            'gpu_available': True,
        }
    except Exception as e:
        _log_once('gpu_read',
                  f"[GPU] GPU attribute read failed ({e}) - GPU metrics unavailable.")
        return _gpu_unavailable()


# ── Core type mapping ─────────────────────────────────────────────────────────
def get_core_types():
    logical = psutil.cpu_count(logical=True)
    core_types = {}
    if logical == 16:
        for i in range(8):
            core_types[i] = 'P'
        for i in range(8, 16):
            core_types[i] = 'E'
    else:
        for i in range(logical):
            core_types[i] = '?'
    return core_types


def get_per_cpu_percent_with_types():
    per_cpu    = psutil.cpu_percent(percpu=True, interval=0.1)
    core_types = get_core_types()
    return [
        {'logical_id': i, 'usage': usage, 'type': core_types.get(i, '?')}
        for i, usage in enumerate(per_cpu)
    ]


# ── Main collection function ──────────────────────────────────────────────────
def collect_all_metrics():
    cpu          = get_cpu_metrics()
    gpu          = get_gpu_metrics()
    per_cpu_info = get_per_cpu_percent_with_types()

    p_usages = [c['usage'] for c in per_cpu_info if c['type'] == 'P']
    e_usages = [c['usage'] for c in per_cpu_info if c['type'] == 'E']

    return {
        **cpu,
        **gpu,
        'per_cpu':    per_cpu_info,
        'avg_p_core': sum(p_usages) / len(p_usages) if p_usages else 0,
        'avg_e_core': sum(e_usages) / len(e_usages) if e_usages else 0,
        'timestamp':  time.time(),
    }