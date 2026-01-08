import json
from argparse import Namespace
from array import array
from typing import List, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _parse_bool_token(tok: str) -> Optional[bool]:
    t = tok.strip().lower()
    if not t:
        return None
    if t in ("1", "true", "t", "yes", "y", "spot", "available", "avail"):
        return True
    if t in ("0", "false", "f", "no", "n", "none", "unavailable", "unavail"):
        return False
    try:
        v = float(t)
        return v >= 0.5
    except Exception:
        return None


def _load_trace_file(path: str) -> bytearray:
    # Try JSON first (fast path for list-like files)
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.read(4096)
            if first.lstrip().startswith(("[", "{")):
                f.seek(0)
                obj = json.load(f)
                if isinstance(obj, dict):
                    for k in ("has_spot", "spot", "availability", "avail", "trace", "data"):
                        if k in obj and isinstance(obj[k], list):
                            obj = obj[k]
                            break
                if isinstance(obj, list):
                    out = bytearray()
                    append = out.append
                    for x in obj:
                        if isinstance(x, bool):
                            append(1 if x else 0)
                        elif isinstance(x, (int, float)):
                            append(1 if float(x) >= 0.5 else 0)
                        elif isinstance(x, str):
                            b = _parse_bool_token(x)
                            if b is None:
                                continue
                            append(1 if b else 0)
                    if out:
                        return out
    except Exception:
        pass

    out = bytearray()
    append = out.append
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Try splitting by common separators, take the last token that parses
            tok = None
            if "," in s:
                parts = s.split(",")
                for p in reversed(parts):
                    b = _parse_bool_token(p)
                    if b is not None:
                        tok = b
                        break
            if tok is None:
                parts = s.split()
                for p in reversed(parts):
                    b = _parse_bool_token(p)
                    if b is not None:
                        tok = b
                        break
            if tok is None:
                continue
            append(1 if tok else 0)
    return out


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v1"

    _TRACE_CACHE = {}

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._t = 0
        self._offset = 0
        self._work_done = 0.0
        self._task_done_len = 0
        self._force_on_demand = False
        self._buffer_seconds = None

        self._spot: List[bytearray] = []
        self._runlen: List[array] = []
        self._any_spot: Optional[bytearray] = None
        self._no_spot_run: Optional[array] = None
        self._trace_len = 0

        trace_files = config.get("trace_files") or []
        for p in trace_files:
            if p in Solution._TRACE_CACHE:
                tr = Solution._TRACE_CACHE[p]
            else:
                tr = _load_trace_file(p)
                Solution._TRACE_CACHE[p] = tr
            self._spot.append(tr)

        if self._spot:
            n = min(len(tr) for tr in self._spot if tr is not None)
            if n <= 0:
                self._trace_len = 0
                return self
            self._trace_len = n
            for i in range(len(self._spot)):
                if len(self._spot[i]) != n:
                    self._spot[i] = self._spot[i][:n]

            rcount = len(self._spot)
            any_spot = bytearray(n)
            for t in range(n):
                v = 0
                for r in range(rcount):
                    if self._spot[r][t]:
                        v = 1
                        break
                any_spot[t] = v
            self._any_spot = any_spot

            runlen = []
            for r in range(rcount):
                rl = array("I", [0]) * (n + 1)
                tr = self._spot[r]
                for t in range(n - 1, -1, -1):
                    rl[t] = (rl[t + 1] + 1) if tr[t] else 0
                runlen.append(rl)
            self._runlen = runlen

            no_run = array("I", [0]) * (n + 1)
            for t in range(n - 1, -1, -1):
                no_run[t] = (no_run[t + 1] + 1) if (any_spot[t] == 0) else 0
            self._no_spot_run = no_run

        return self

    def _get_task_duration_seconds(self) -> float:
        td = getattr(self, "task_duration", None)
        if td is None:
            tds = getattr(self, "task_durations", None)
            if isinstance(tds, (list, tuple)) and tds:
                return float(tds[0])
            return 0.0
        try:
            return float(td)
        except Exception:
            if isinstance(td, (list, tuple)) and td:
                return float(td[0])
            return 0.0

    def _get_deadline_seconds(self) -> float:
        dl = getattr(self, "deadline", None)
        try:
            return float(dl)
        except Exception:
            return 0.0

    def _get_restart_overhead_seconds(self) -> float:
        ro = getattr(self, "restart_overhead", None)
        try:
            return float(ro)
        except Exception:
            return 0.0

    def _update_work_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n > self._task_done_len:
            self._work_done += float(sum(td[self._task_done_len : n]))
            self._task_done_len = n

    def _align_offset(self, region: int, has_spot: bool, idx: int) -> int:
        if self._trace_len <= 0 or not self._spot:
            return idx
        if idx < 0:
            idx = 0
        if idx >= self._trace_len:
            idx = self._trace_len - 1
        pred = bool(self._spot[region][idx])
        if pred == bool(has_spot):
            return idx
        best_delta = None
        best_abs = 1 << 30
        max_shift = 12
        for d in range(-max_shift, max_shift + 1):
            j = idx + d
            if j < 0 or j >= self._trace_len:
                continue
            if bool(self._spot[region][j]) == bool(has_spot):
                ad = -d if d < 0 else d
                if ad < best_abs:
                    best_abs = ad
                    best_delta = d
                    if ad == 0:
                        break
        if best_delta is not None and best_delta != 0:
            self._offset += best_delta
            idx += best_delta
            if idx < 0:
                idx = 0
            elif idx >= self._trace_len:
                idx = self._trace_len - 1
        return idx

    def _best_spot_region(self, idx: int) -> Optional[int]:
        if self._trace_len <= 0 or not self._runlen:
            return None
        if idx < 0:
            idx = 0
        elif idx >= self._trace_len:
            idx = self._trace_len - 1
        best_r = None
        best_len = 0
        for r, rl in enumerate(self._runlen):
            l = rl[idx]
            if l > best_len:
                best_len = l
                best_r = r
        return best_r if best_len > 0 else None

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_work_done()

        task_duration = self._get_task_duration_seconds()
        deadline = self._get_deadline_seconds()
        restart_overhead = self._get_restart_overhead_seconds()

        if self._work_done >= task_duration:
            self._t += 1
            return ClusterType.NONE

        time_left = float(deadline) - float(self.env.elapsed_seconds)
        remaining_work = float(task_duration) - float(self._work_done)

        if time_left <= 0.0:
            self._t += 1
            return ClusterType.NONE

        gap = float(self.env.gap_seconds)

        if self._buffer_seconds is None:
            self._buffer_seconds = max(600.0, 10.0 * gap, 4.0 * restart_overhead)

        slack = time_left - remaining_work - float(getattr(self, "remaining_restart_overhead", 0.0))

        if (not self._force_on_demand) and (slack <= self._buffer_seconds):
            self._force_on_demand = True

        if self._force_on_demand:
            self._t += 1
            return ClusterType.ON_DEMAND

        if float(getattr(self, "remaining_restart_overhead", 0.0)) > 0.0:
            if last_cluster_type == ClusterType.SPOT and has_spot:
                self._t += 1
                return ClusterType.SPOT
            if last_cluster_type == ClusterType.ON_DEMAND:
                self._t += 1
                return ClusterType.ON_DEMAND

        current_region = int(self.env.get_current_region())
        idx = self._t + self._offset

        if self._trace_len > 0 and self._spot and 0 <= current_region < len(self._spot):
            if idx < 0:
                idx = 0
            elif idx >= self._trace_len:
                idx = self._trace_len - 1
            idx = self._align_offset(current_region, has_spot, idx)

        any_spot_now = False
        if self._any_spot is not None and self._trace_len > 0:
            if 0 <= idx < self._trace_len:
                any_spot_now = bool(self._any_spot[idx])
            else:
                any_spot_now = bool(has_spot)
        else:
            any_spot_now = bool(has_spot)

        if any_spot_now:
            if has_spot:
                self._t += 1
                return ClusterType.SPOT
            best_r = self._best_spot_region(idx)
            if best_r is None:
                self._t += 1
                return ClusterType.ON_DEMAND
            if best_r != current_region:
                self.env.switch_region(best_r)
            self._t += 1
            return ClusterType.SPOT

        outage_steps = 1
        if self._no_spot_run is not None and self._trace_len > 0 and 0 <= idx < self._trace_len:
            outage_steps = int(self._no_spot_run[idx]) or 1

        if slack > float(outage_steps) * gap + self._buffer_seconds:
            self._t += 1
            return ClusterType.NONE

        self._t += 1
        return ClusterType.ON_DEMAND