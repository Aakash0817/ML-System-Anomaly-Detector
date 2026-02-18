import psutil
import time
from collections import defaultdict

class ProcessTracker:
    def __init__(self, top_n=5, update_interval=5):
        self.top_n = top_n
        self.update_interval = update_interval  # seconds between full scans
        self.last_scan_time = 0
        self.cached_top = []
        self.cached_suspicious = []
        self.process_history = defaultdict(list)

    def _scan_processes(self):
        """Fast scan of all processes – non‑blocking cpu_percent."""
        top_candidates = []
        suspicious = []
        high_cpu_threshold = 50
        high_mem_threshold = 500  # MB

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent',
                                         'memory_percent', 'memory_info']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] is None or pinfo['memory_percent'] is None:
                    continue

                cpu = pinfo['cpu_percent'] / psutil.cpu_count()  # normalize
                mem_mb = pinfo['memory_info'].rss / (1024 * 1024) if pinfo['memory_info'] else 0

                # Collect top candidates (all, will sort later)
                top_candidates.append({
                    'pid': pinfo['pid'],
                    'name': pinfo['name'],
                    'cpu_percent': cpu,
                    'memory_percent': pinfo['memory_percent'],
                    'memory_mb': mem_mb
                })

                # Check for suspicious processes
                if cpu > high_cpu_threshold or mem_mb > high_mem_threshold:
                    suspicious.append({
                        'pid': pinfo['pid'],
                        'name': pinfo['name'],
                        'cpu': cpu,
                        'memory_mb': mem_mb,
                        'reason': 'High CPU' if cpu > high_cpu_threshold else 'High Memory'
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Sort by CPU and keep top N
        top_candidates.sort(key=lambda x: x['cpu_percent'], reverse=True)
        top = top_candidates[:self.top_n]

        # Update history (optional)
        timestamp = time.time()
        for proc in top:
            self.process_history[proc['pid']].append({
                'timestamp': timestamp,
                'cpu': proc['cpu_percent'],
                'memory': proc['memory_mb']
            })
            if len(self.process_history[proc['pid']]) > 60:
                self.process_history[proc['pid']].pop(0)

        return top, suspicious

    def get_top_processes(self):
        """Return cached top processes, refresh if needed."""
        now = time.time()
        if now - self.last_scan_time > self.update_interval:
            self.cached_top, self.cached_suspicious = self._scan_processes()
            self.last_scan_time = now
        return self.cached_top

    def get_suspicious_processes(self):
        """Return cached suspicious processes, refresh if needed."""
        now = time.time()
        if now - self.last_scan_time > self.update_interval:
            self.cached_top, self.cached_suspicious = self._scan_processes()
            self.last_scan_time = now
        return self.cached_suspicious