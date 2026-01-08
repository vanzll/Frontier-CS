import json
import math
import os
import gzip
from argparse import Namespace
from typing import Any, List, Optional, Sequence, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _cluster_none() -> ClusterType:
    return getattr(ClusterType, "NONE", getattr(ClusterType, "None", None))


_CT_NONE = _cluster_none()


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def _as_bool_token(tok: str) -> Optional[bool]:
    t = tok.strip().strip('"').strip("'")
    if not t:
        return None
    tl = t.lower()
    if tl in ("true", "t", "yes", "y", "on"):
        return True
    if tl in ("false", "f", "no", "n", "off"):
        return False
    try:
        v = float(t)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v > 0.5


def _extract_values_from_json(obj: Any) -> List[bool]:
    if obj is None:
        return []
    if isinstance(obj, list):
        out: List[bool] = []
        for x in obj:
            if isinstance(x, bool):
                out.append(bool(x))
            elif isinstance(x, (int, float)):
                out.append(float(x) > 0.5)
            elif isinstance(x, str):
                b = _as_bool_token(x)
                if b is not None:
                    out.append(b)
            elif isinstance(x, dict):
                for k in ("available", "availability", "has_spot", "spot", "value", "avail"):
                    if k in x:
                        xv = x[k]
                        if isinstance(xv, bool):
                            out.append(bool(xv))
                        elif isinstance(xv, (int, float)):
                            out.append(float(xv) > 0.5)
                        elif isinstance(xv, str):
                            b = _as_bool_token(xv)
                            if b is not None:
                                out.append(b)
                        break
        return out
    if isinstance(obj, dict):
        for k in ("availability", "available", "has_spot", "spot", "values", "trace", "data"):
            if k in obj:
                return _extract_values_from_json(obj[k])
        for _, v in obj.items():
            arr = _extract_values_from_json(v)
            if arr:
                return arr
    return []


def _load_trace_file(path: str) -> List[bool]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json" or path.lower().endswith(".json.gz"):
        with _open_maybe_gzip(path) as f:
            try:
                obj = json.load(f)
            except Exception:
                obj = None
        vals = _extract_values_from_json(obj)
        if vals:
            return vals

    vals: List[bool] = []
    with _open_maybe_gzip(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s[0] in ("#", "%"):
                continue
            if any(c.isalpha() for c in s) and not any(ch in s for ch in ("true", "false", "True", "False", "0", "1")):
                continue
            parts = [p for p in s.replace(",", " ").split() if p]
            if not parts:
                continue
            picked = parts[-1]
            b = _as_bool_token(picked)
            if b is None and len(parts) >= 2:
                b = _as_bool_token(parts[-2])
            if b is None:
                continue
            vals.append(b)
    return vals


def _resample_to_length(seq: Sequence[bool], target_len: int) -> List[bool]:
    n = len(seq)
    if target_len <= 0:
        return []
    if n == 0:
        return [False] * target_len
    if n == target_len:
        return list(seq)

    ratio = target_len / n
    if ratio > 1.1:
        k = int(round(ratio))
        if k >= 2 and abs(ratio - k) / ratio < 0.02:
            out = [False] * (n * k)
            idx = 0
            for v in seq:
                for _ in range(k):
                    out[idx] = bool(v)
                    idx += 1
            if len(out) >= target_len:
                return out[:target_len]
            out.extend([False] * (target_len - len(out)))
            return out
    elif ratio < 0.9:
        inv = 1.0 / ratio
        k = int(round(inv))
        if k >= 2 and abs(inv - k) / inv < 0.02:
            out = []
            for i in range(0, n, k):
                out.append(bool(seq[i]))
                if len(out) >= target_len:
                    break
            if len(out) < target_len:
                out.extend([False] * (target_len - len(out)))
            return out[:target_len]

    out = list(seq[:target_len])
    if len(out) < target_len:
        out.extend([False] * (target_len - len(out)))
    return out


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._od_price_per_hour = 3.06
        self._spot_price_per_hour = 0.9701

        self._work_done = 0.0
        self._last_task_done_len = 0
        self._committed_od = False

        self._gap = float(getattr(self.env, "gap_seconds", 1.0))
        self._nsteps = int(math.ceil(self.deadline / self._gap)) + 2

        trace_files = config.get("trace_files") or []
        self._num_regions = int(getattr(self.env, "get_num_regions", lambda: len(trace_files))())
        if trace_files and len(trace_files) < self._num_regions:
            self._num_regions = len(trace_files)

        self._avail: List[List[bool]] = []
        for i in range(self._num_regions):
            path = trace_files[i]
            vals = _load_trace_file(path)
            vals = _resample_to_length(vals, self._nsteps)
            self._avail.append(vals)

        if not self._avail:
            self._avail = [[False] * self._nsteps]

        self._run_len: List[List[int]] = []
        self._next_spot: List[List[int]] = []
        for r in range(len(self._avail)):
            a = self._avail[r]
            rl = [0] * (self._nsteps + 1)
            ns = [self._nsteps + 1] * (self._nsteps + 1)
            next_idx = self._nsteps + 1
            for i in range(self._nsteps - 1, -1, -1):
                if a[i]:
                    rl[i] = rl[i + 1] + 1
                    next_idx = i
                else:
                    rl[i] = 0
                ns[i] = next_idx
            self._run_len.append(rl)
            self._next_spot.append(ns)

        denom = (self._od_price_per_hour - self._spot_price_per_hour)
        if denom <= 1e-9:
            self._switch_threshold_seconds = float("inf")
        else:
            self._switch_threshold_seconds = (self._od_price_per_hour / denom) * float(self.restart_overhead)

        self._wait_slack_threshold = max(3600.0, 8.0 * float(self.restart_overhead) + 5.0 * self._gap)
        self._hard_commit_threshold = max(600.0, 3.0 * float(self.restart_overhead) + 2.0 * self._gap)

        return self

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n <= self._last_task_done_len:
            return
        s = 0.0
        for i in range(self._last_task_done_len, n):
            s += float(td[i])
        self._work_done += s
        self._last_task_done_len = n

    def _best_spot_region_now(self, idx: int, num_regions: int) -> Tuple[int, int]:
        best_r = -1
        best_run = 0
        for r in range(num_regions):
            rl = self._run_len[r][idx] if idx < self._nsteps else 0
            if rl > best_run:
                best_run = rl
                best_r = r
        if best_run <= 0:
            return -1, 0
        return best_r, best_run

    def _earliest_next_spot_region(self, idx: int, num_regions: int) -> Optional[int]:
        best_r = None
        best_t = self._nsteps + 1
        best_run = -1
        for r in range(num_regions):
            t = self._next_spot[r][idx] if idx < self._nsteps else (self._nsteps + 1)
            if t > self._nsteps:
                continue
            run = self._run_len[r][t] if t < self._nsteps else 0
            if t < best_t or (t == best_t and run > best_run):
                best_t = t
                best_run = run
                best_r = r
        return best_r

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_work_done()

        remaining_work = float(self.task_duration) - float(self._work_done)
        if remaining_work <= 1e-9:
            self._committed_od = True
            return _CT_NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        time_left = float(self.deadline) - elapsed
        if time_left <= 0.0:
            self._committed_od = True
            return ClusterType.ON_DEMAND

        pending_ovh = float(getattr(self, "remaining_restart_overhead", 0.0) or 0.0)
        slack = time_left - remaining_work - pending_ovh

        if slack <= self._hard_commit_threshold:
            self._committed_od = True

        if self._committed_od:
            return ClusterType.ON_DEMAND

        idx = int(elapsed // self._gap) if self._gap > 0 else 0
        if idx < 0:
            idx = 0
        if idx >= self._nsteps:
            idx = self._nsteps - 1

        num_regions = self._num_regions
        if hasattr(self.env, "get_num_regions"):
            try:
                nr = int(self.env.get_num_regions())
                if nr > 0:
                    num_regions = min(num_regions, nr) if num_regions > 0 else nr
            except Exception:
                pass
        if num_regions <= 0:
            num_regions = 1

        try:
            cur_region = int(self.env.get_current_region())
        except Exception:
            cur_region = 0
        if cur_region < 0 or cur_region >= num_regions:
            cur_region = 0

        best_r, best_run = self._best_spot_region_now(idx, num_regions)
        any_spot_now = best_r != -1

        if any_spot_now:
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT

            best_run_sec = float(best_run) * self._gap
            restart_required = (last_cluster_type != ClusterType.SPOT) or (best_r != cur_region)

            if restart_required and best_run_sec < self._switch_threshold_seconds:
                return ClusterType.ON_DEMAND

            if not has_spot:
                if slack > self._wait_slack_threshold:
                    if best_r != cur_region:
                        try:
                            self.env.switch_region(best_r)
                        except Exception:
                            pass
                    return _CT_NONE
                return ClusterType.ON_DEMAND

            if best_r != cur_region:
                try:
                    self.env.switch_region(best_r)
                except Exception:
                    pass
            return ClusterType.SPOT

        if slack > self._wait_slack_threshold:
            r_next = self._earliest_next_spot_region(idx, num_regions)
            if r_next is not None and r_next != cur_region:
                try:
                    self.env.switch_region(r_next)
                except Exception:
                    pass
            return _CT_NONE

        return ClusterType.ON_DEMAND