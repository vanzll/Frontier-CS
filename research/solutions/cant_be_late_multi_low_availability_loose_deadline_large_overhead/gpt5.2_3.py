import json
import math
import os
from argparse import Namespace
from array import array
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

_TRACE_CACHE: Dict[str, bytearray] = {}


def _flatten_iter(x: Any) -> Iterable[Any]:
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten_iter(y)
    else:
        yield x


def _coerce_bool(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if float(v) > 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return 0
        if s in ("1", "true", "t", "yes", "y", "on", "available", "avail", "up"):
            return 1
        if s in ("0", "false", "f", "no", "n", "off", "unavailable", "down"):
            return 0
        try:
            return 1 if float(s) > 0.5 else 0
        except Exception:
            return 0
    if isinstance(v, dict):
        for k in ("available", "avail", "spot", "has_spot", "is_available", "up"):
            if k in v:
                return _coerce_bool(v.get(k))
        for k in ("unavailable", "interrupted", "down"):
            if k in v:
                return 0 if _coerce_bool(v.get(k)) else 1
        return 0
    return 0


def _extract_json_series(obj: Any) -> Optional[List[Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("availability", "availabilities", "has_spot", "spot", "data", "trace", "series", "values"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        for v in obj.values():
            if isinstance(v, list):
                return v
    return None


def _parse_trace_file(path: str) -> bytearray:
    cached = _TRACE_CACHE.get(path)
    if cached is not None:
        return cached

    ext = os.path.splitext(path)[1].lower()

    if ext in (".npy", ".npz"):
        try:
            import numpy as np  # type: ignore

            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):
                keys = list(arr.keys())
                if not keys:
                    data = bytearray()
                else:
                    a = arr[keys[0]]
                    flat = a.reshape(-1)
                    data = bytearray(int(x > 0.5) for x in flat)
            else:
                flat = arr.reshape(-1)
                data = bytearray(int(x > 0.5) for x in flat)
            _TRACE_CACHE[path] = data
            return data
        except Exception:
            pass

    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        head_str = head.decode("utf-8", errors="ignore").lstrip()
    except Exception:
        data = bytearray()
        _TRACE_CACHE[path] = data
        return data

    if head_str.startswith("{") or head_str.startswith("["):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            series = _extract_json_series(obj)
            if series is None:
                data = bytearray()
            else:
                data = bytearray(_coerce_bool(v) for v in _flatten_iter(series))
            _TRACE_CACHE[path] = data
            return data
        except Exception:
            pass

    data = bytearray()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.replace("\t", " ").replace(",", " ").split()
                if not parts:
                    continue
                data.append(_coerce_bool(parts[-1]))
    except Exception:
        data = bytearray()

    _TRACE_CACHE[path] = data
    return data


def _first_scalar(x: Any) -> float:
    if isinstance(x, (list, tuple)):
        if not x:
            return 0.0
        return float(x[0])
    return float(x)


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_ms_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        self._trace_files = list(config.get("trace_files", []))
        self._raw_traces: List[bytearray] = []
        for p in self._trace_files:
            try:
                self._raw_traces.append(_parse_trace_file(p))
            except Exception:
                self._raw_traces.append(bytearray())

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._runtime_inited = False
        self._committed = False

        self._done_cached = 0.0
        self._done_len = 0

        self._avail: List[bytearray] = []
        self._next_true: List[array] = []
        self._run_len: List[array] = []

        self._gap = 0.0
        self._deadline = 0.0
        self._task_duration = 0.0
        self._restart_overhead = 0.0

        self._buffer_seconds = 0.0
        self._switch_lookahead_steps = 0
        self._min_switch_interval_sec = 0.0
        self._last_switch_elapsed = -1e30

        return self

    def _init_runtime(self) -> None:
        if self._runtime_inited:
            return

        self._gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)

        self._deadline = _first_scalar(getattr(self, "deadline", 0.0))
        self._task_duration = _first_scalar(getattr(self, "task_duration", 0.0))
        self._restart_overhead = _first_scalar(getattr(self, "restart_overhead", 0.0))

        if self._deadline <= 0:
            self._deadline = 1.0
        if self._task_duration < 0:
            self._task_duration = 0.0
        if self._restart_overhead < 0:
            self._restart_overhead = 0.0

        num_regions = int(self.env.get_num_regions())
        steps_needed = int(math.ceil(self._deadline / self._gap)) + 3
        if steps_needed < 4:
            steps_needed = 4

        self._buffer_seconds = max(2.0 * self._gap, 4.0 * self._restart_overhead, 300.0)

        lookahead_sec = min(3.0 * 3600.0, max(1200.0, 2.0 * self._restart_overhead + self._gap))
        self._switch_lookahead_steps = int(math.ceil(lookahead_sec / self._gap))
        if self._switch_lookahead_steps < 1:
            self._switch_lookahead_steps = 1

        self._min_switch_interval_sec = max(600.0, 2.0 * self._restart_overhead, 2.0 * self._gap)

        self._avail = []
        self._next_true = []
        self._run_len = []

        for r in range(num_regions):
            raw = self._raw_traces[r] if r < len(self._raw_traces) else bytearray()
            if len(raw) < steps_needed:
                a = bytearray(raw)
                a.extend(b"\x00" * (steps_needed - len(a)))
            else:
                a = bytearray(raw[:steps_needed])
            self._avail.append(a)

            T = steps_needed
            inf = T + 10
            nt_arr = array("I", [inf]) * (T + 1)
            rl_arr = array("I", [0]) * (T + 1)
            nt = inf
            rl = 0
            for t in range(T - 1, -1, -1):
                if a[t]:
                    nt = t
                    rl += 1
                else:
                    rl = 0
                nt_arr[t] = nt
                rl_arr[t] = rl
            nt_arr[T] = inf
            rl_arr[T] = 0
            self._next_true.append(nt_arr)
            self._run_len.append(rl_arr)

        self._runtime_inited = True

    def _update_done_cache(self) -> None:
        td = self.task_done_time
        if not isinstance(td, list):
            return
        n = len(td)
        if n <= self._done_len:
            return
        self._done_cached += float(sum(td[self._done_len : n]))
        self._done_len = n

    def _select_best_region(self, search_idx: int) -> Optional[int]:
        num_regions = int(self.env.get_num_regions())
        if num_regions <= 1 or not self._runtime_inited:
            return None

        cur_r = int(self.env.get_current_region())
        if search_idx < 0:
            search_idx = 0

        T = len(self._avail[0]) if self._avail else 0
        if T <= 0:
            return None
        if search_idx >= T:
            search_idx = T - 1

        inf = T + 10

        best_r = cur_r
        best_nt = self._next_true[cur_r][search_idx]
        if best_nt >= inf:
            best_rl = 0
        else:
            best_rl = self._run_len[cur_r][best_nt]

        for r in range(num_regions):
            nt = self._next_true[r][search_idx]
            if nt < best_nt:
                best_r = r
                best_nt = nt
                best_rl = self._run_len[r][nt] if nt < inf else 0
            elif nt == best_nt and nt < inf:
                rl = self._run_len[r][nt]
                if rl > best_rl:
                    best_r = r
                    best_rl = rl

        if best_r == cur_r:
            return None

        cur_nt = self._next_true[cur_r][search_idx]
        if best_nt >= inf:
            return None
        if cur_nt >= inf:
            return best_r

        if best_nt + 1 < cur_nt:
            return best_r
        if best_nt == cur_nt:
            cur_rl = self._run_len[cur_r][cur_nt] if cur_nt < inf else 0
            if best_rl > cur_rl + 2:
                return best_r

        return None

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_runtime()
        self._update_done_cache()

        remaining_work = self._task_duration - self._done_cached
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_left = self._deadline - elapsed
        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        if self._committed:
            return ClusterType.ON_DEMAND

        rr = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)

        if last_cluster_type == ClusterType.ON_DEMAND:
            extra_overhead = rr
        else:
            extra_overhead = self._restart_overhead

        required_od_time = remaining_work + extra_overhead

        if time_left <= required_od_time + self._buffer_seconds:
            self._committed = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        if time_left - self._gap <= required_od_time + self._buffer_seconds:
            self._committed = True
            return ClusterType.ON_DEMAND

        if (elapsed - self._last_switch_elapsed) >= self._min_switch_interval_sec:
            t_idx = int(elapsed // self._gap) if self._gap > 0 else 0
            search_idx = t_idx + 1
            best = self._select_best_region(search_idx)
            if best is not None:
                try:
                    self.env.switch_region(int(best))
                    self._last_switch_elapsed = elapsed
                except Exception:
                    pass

        return ClusterType.NONE