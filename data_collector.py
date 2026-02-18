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

_last_cpu_temp = 45.0    # sensible startup default
_temp_lock     = threading.Lock()
_temp_interval = 2.0     # read temperature every 2 seconds


def _temperature_worker():
    """
    Dedicated thread for WMI temperature reads.
    Has its own CoInitialize so it never blocks the producer.
    """
    global _last_cpu_temp

    import pythoncom
    pythoncom.CoInitialize()

    wmi_conn = None
    try:
        import wmi
        wmi_conn = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        print("[TempWorker] WMI connected.", flush=True)
    except Exception as e:
        print(f"[TempWorker] WMI connect failed: {e}", flush=True)

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
                print(f"[TempWorker] Read error: {e}", flush=True)
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

        time.sleep(_temp_interval)


def get_cpu_temperature():
    """Non-blocking — returns the last reading from the background thread."""
    with _temp_lock:
        return _last_cpu_temp


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
def get_gpu_metrics():
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'gpu_percent': gpu.load * 100,
                    'gpu_memory':  gpu.memoryUtil * 100,
                    'gpu_temp':    gpu.temperature,
                }
        except Exception:
            pass
    return {'gpu_percent': 0, 'gpu_memory': 0, 'gpu_temp': 0}


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