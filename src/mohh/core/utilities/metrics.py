import json
import os
import time

import numpy as np


def _json_safe(obj):
    """
    Recursively convert a python object into something JSON serializable.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


class RunMetrics:
    """
    Lightweight, process-level metrics collector used to log the WHOLE run:
      * per-scope computational timings (categories), excluding I/O
      * named event sections (HV per individual, ranks, fronts, recalc, ...)
      * counters (number of MOEA solves, offspring attempts/rejections, ...)
      * metadata + per-step JSON snapshots + an end-of-run summary

    The instrumentation never changes the behaviour of the experiment:
    if LOG_METRICS is disabled (or the collector has not been initialised)
    every method becomes a no-op.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RunMetrics, cls).__new__(cls)
            cls._instance._configure()
        return cls._instance

    def _configure(self):
        self.enabled = False
        self.initialized = False
        self.root_dir = None
        self.phase = None
        self._timers = {}
        self._active = {}
        self._sections = {}
        self._counters = {}
        self._meta = {}
        self._prev_timers = {}
        self._wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self, root_dir: str, phase: str):
        """Prepare the collector for a new phase (generation / evaluation)."""
        self.root_dir = root_dir
        self.phase = phase
        os.makedirs(root_dir, exist_ok=True)
        self.enabled = True
        self.initialized = True
        self._timers.clear()
        self._active.clear()
        self._sections.clear()
        self._counters.clear()
        self._prev_timers.clear()
        self._wall_start = time.perf_counter()
        return self

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def set_info(self, key, value):
        """Record persistent metadata (kept for every dump + the summary)."""
        self._meta[key] = value

    @classmethod
    def reset_instance(cls):
        if cls._instance is not None:
            cls._instance._configure()
            cls._instance = None

    # ------------------------------------------------------------------
    # Scoped computation timers  (category labels allow I/O exclusion)
    # ------------------------------------------------------------------
    def start(self, scope: str, category: str = None):
        if not (self.enabled and self.initialized):
            return
        if scope not in self._timers:
            self._timers[scope] = {"total": 0.0, "count": 0, "last": 0.0,
                                   "start": None, "category": category}
        self._timers[scope]["start"] = time.perf_counter()

    def stop(self, scope: str):
        if not (self.enabled and self.initialized):
            return
        t = self._timers.get(scope)
        if t is None or t["start"] is None:
            return
        delta = time.perf_counter() - t["start"]
        t["total"] += delta
        t["count"] += 1
        t["last"] = delta
        t["start"] = None

    def timing(self, scope: str, seconds: float, category: str = None):
        """Record an already-measured interval."""
        if not (self.enabled and self.initialized):
            return
        if scope not in self._timers:
            self._timers[scope] = {"total": 0.0, "count": 0, "last": 0.0,
                                   "start": None, "category": category}
        self._timers[scope]["total"] += seconds
        self._timers[scope]["count"] += 1
        self._timers[scope]["last"] = seconds

    # ------------------------------------------------------------------
    # Events / counters
    # ------------------------------------------------------------------
    def record(self, section: str, entry: dict):
        if not (self.enabled and self.initialized):
            return
        self._sections.setdefault(section, []).append(entry)

    def increment(self, section: str, key: str, amount: int = 1):
        if not (self.enabled and self.initialized):
            return
        bucket = self._counters.setdefault(section, {})
        bucket[key] = bucket.get(key, 0) + amount

    def save_json(self, rel_path: str, obj):
        """
        Write a JSON file under run_metrics/ (timed as I/O, i.e. excluded
        from the computational-cost categories on purpose).
        """
        if not (self.enabled and self.initialized):
            return
        t0 = time.perf_counter()
        full = os.path.join(self.root_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as file:
            json.dump(obj, file, indent=2, default=_json_safe)
        self.timing("io_metric_files", time.perf_counter() - t0, category="io")

    # ------------------------------------------------------------------
    # Snapshots / summary
    # ------------------------------------------------------------------
    def _timers_snapshot(self):
        snapshot = {}
        for scope, t in self._timers.items():
            prev = self._prev_timers.get(scope, {"total": 0.0})["total"]
            snapshot[scope] = {
                "total_s": round(t["total"], 9),
                "since_last_s": round(max(t["total"] - prev, 0.0), 9),
                "count": t["count"],
                "last_s": round(t["last"], 9),
                "category": t["category"],
            }
        return snapshot

    def _category_totals(self, since_last: bool = True):
        totals = {}
        for scope, t in self._timers.items():
            if t["category"] is None:
                # Container/informational scopes are excluded from the sums.
                continue
            prev = self._prev_timers.get(scope, {"total": 0.0})["total"]
            value = (max(t["total"] - prev, 0.0) if since_last else t["total"])
            totals[t["category"]] = totals.get(t["category"], 0.0) + value
        return {key: round(value, 9) for key, value in sorted(totals.items())}

    def dump(self, name: str, header: dict = None):
        """
        Write run_metrics/<name>.json with the timers delta+cumulative,
        category totals (pure computation vs io), event sections and counters.
        The events/counters accumulated since the previous dump are cleared.
        """
        if not (self.enabled and self.initialized):
            return

        payload = {
            "phase": self.phase,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": round(time.perf_counter() - self._wall_start, 9),
        }
        if header:
            payload.update(header)

        payload["timers"] = self._timers_snapshot()
        payload["category_totals"] = self._category_totals(since_last=True)
        payload["sections"] = self._sections
        payload["counters"] = self._counters

        with open(os.path.join(self.root_dir, name + ".json"), "w") as file:
            json.dump(payload, file, indent=2, default=_json_safe)

        self._prev_timers = {scope: {"total": t["total"]} for scope, t in self._timers.items()}
        self._sections.clear()
        self._counters.clear()

    def write_summary(self):
        """
        Write run_metrics/summary.json with the run-wide totals:
        wall time, per-category computational cost (I/O separated) and
        every scope's cumulative time together with the persistent metadata.
        """
        if not (self.enabled and self.initialized):
            return

        category_totals = self._category_totals(since_last=False)
        compute_total = sum(v for k, v in category_totals.items() if k != "io")

        timers = {
            scope: {"total_s": round(t["total"], 9),
                    "count": t["count"],
                    "last_s": round(t["last"], 9),
                    "category": t["category"]}
            for scope, t in self._timers.items()
        }

        payload = {
            "phase": self.phase,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": round(time.perf_counter() - self._wall_start, 9),
            "compute_total_s": round(compute_total, 9),
            "io_total_s": round(category_totals.get("io", 0.0), 9),
            "category_totals_s": category_totals,
            "timers": timers,
            "meta": _json_safe(self._meta),
        }

        with open(os.path.join(self.root_dir, "summary.json"), "w") as file:
            json.dump(payload, file, indent=2, default=_json_safe)

    def close(self):
        self.write_summary()
        self.initialized = False