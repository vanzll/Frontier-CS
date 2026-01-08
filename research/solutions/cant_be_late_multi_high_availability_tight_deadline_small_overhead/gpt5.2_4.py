import json
import math
import os
import re
from argparse import Namespace
from array import array
from typing import Any, List, Optional, Sequence

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _to_bool(x: Any) -> int:
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, float)):
        return 1 if x > 0 else 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return 1
        if s in ("0", "false", "f", "no", "n", "off", "", "none", "null"):
            return 0
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return 1 if float(m.group(0)) > 0 else 0
            except Exception:
                return 0
    return 0


def _load_trace_file(path: str) -> List[int]:
    # Returns list of 0/1 availability per step.
    p = path
    if not os.path.exists(p):
        return []
    ext = os.path.splitext(p)[1].lower()

    if ext == ".npy":
        try:
            import numpy as np  # type: ignore
            arr = np.load(p, allow_pickle=False)
            arr = arr.reshape(-1)
            return [1 if v else 0 for v in arr.tolist()]
        except Exception:
            pass

    try:
        with open(p, "rb") as f:
            data = f.read()
    except Exception:
        return []

    if not data:
        return []

    # Try JSON.
    first_non_ws = None
    for b in data:
        if b not in b" \t\r\n":
            first_non_ws = b
            break
    if first_non_ws in (ord("["), ord("{")):
        try:
            obj = json.loads(data.decode("utf-8"))
            if isinstance(obj, dict):
                for k in ("availability", "trace", "spot", "data", "values"):
                    if k in obj and isinstance(obj[k], list):
                        obj = obj[k]
                        break
            if isinstance(obj, list):
                out = []
                for v in obj:
                    if isinstance(v, (list, tuple)) and v:
                        out.append(_to_bool(v[-1]))
                    elif isinstance(v, dict):
                        if "available" in v:
                            out.append(_to_bool(v["available"]))
                        elif "spot" in v:
                            out.append(_to_bool(v["spot"]))
                        elif "value" in v:
                            out.append(_to_bool(v["value"]))
                        else:
                            out.append(0)
                    else:
                        out.append(_to_bool(v))
                return out
        except Exception:
            pass

    # Try text lines.
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return []

    lines = text.splitlines()
    out: List[int] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Common formats:
        # "0" / "1"
        # "timestamp,0/1"
        # "..., availability=1"
        parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
        if not parts:
            continue
        # Prefer last token; if not parseable, search for a 0/1 in the line.
        v = parts[-1]
        if v in ("0", "1", "true", "false", "True", "False"):
            out.append(_to_bool(v))
            continue
        m01 = re.search(r"\b([01])\b", s[::-1])
        if m01:
            # m01 is on reversed string; find last 0/1 in original.
            m = re.search(r"([01])\b(?!.*\b[01]\b)", s)
            if m:
                out.append(1 if m.group(1) == "1" else 0)
                continue
        out.append(_to_bool(v))
    return out


class Solution(MultiRegionStrategy):
    NAME = "trace_aware_wait_v1"

    # Prices (given in problem statement). Used only for heuristics.
    _PRICE_ON_DEMAND = 3.06
    _PRICE_SPOT = 0.9701

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

        trace_files = config.get("trace_files", []) or []
        base = os.path.dirname(os.path.abspath(spec_path))
        self._trace_paths = [p if os.path.isabs(p) else os.path.join(base, p) for p in trace_files]
        self._raw_traces: List[List[int]] = [_load_trace_file(p) for p in self._trace_paths]

        # Lazy-prepared once env.gap_seconds is known.
        self._prepared = False
        self._use_traces = bool(self._raw_traces)

        self._gap = None  # type: Optional[float]
        self._n_steps = 0
        self._avail: List[bytearray] = []
        self._streak: List[array] = []
        self._next_true: List[array] = []

        self._overhead_steps = 1
        self._switch_threshold_steps = 1
        self._min_spot_switch_steps = 1

        # Efficient tracking of done work.
        self._td_len = 0
        self._done = 0.0

        # Trace calibration (to avoid errors if trace resolution mismatches).
        self._calib_obs = 0
        self._calib_mismatch = 0

        return self

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._prepared = True

        try:
            gap = float(getattr(self.env, "gap_seconds"))
            if not math.isfinite(gap) or gap <= 0:
                gap = 1.0
        except Exception:
            gap = 1.0
        self._gap = gap

        # Number of steps we care about: cover deadline with some padding.
        n_needed = int(math.ceil(float(self.deadline) / gap)) + 5
        if n_needed < 16:
            n_needed = 16
        self._n_steps = n_needed

        self._overhead_steps = max(1, int(math.ceil(float(self.restart_overhead) / gap)))
        # Avoid switching unless it meaningfully increases continuous spot run.
        self._switch_threshold_steps = max(1, 3 * self._overhead_steps)

        # Switching from ON_DEMAND to SPOT only if expected spot run is sufficiently long.
        min_secs = 600.0  # 10 minutes
        self._min_spot_switch_steps = max(5 * self._overhead_steps, int(math.ceil(min_secs / gap)))

        if not self._use_traces:
            self._avail = []
            self._streak = []
            self._next_true = []
            return

        num_regions = len(self._raw_traces)
        if num_regions <= 0:
            self._use_traces = False
            return

        self._avail = []
        self._streak = []
        self._next_true = []

        for r in range(num_regions):
            raw = self._raw_traces[r] if r < len(self._raw_traces) else []
            if raw is None:
                raw = []
            if len(raw) >= n_needed:
                a = raw[:n_needed]
            else:
                a = raw + [0] * (n_needed - len(raw))
            av = bytearray((1 if v else 0) for v in a)
            self._avail.append(av)

            st = array("I", [0]) * (n_needed + 1)
            nt = array("I", [0]) * (n_needed + 1)
            inf = n_needed + 10

            nxt = inf
            run = 0
            for t in range(n_needed - 1, -1, -1):
                if av[t]:
                    run += 1
                    nxt = t
                else:
                    run = 0
                st[t] = run
                nt[t] = nxt
            st[n_needed] = 0
            nt[n_needed] = inf

            self._streak.append(st)
            self._next_true.append(nt)

    def _update_done(self) -> None:
        td = self.task_done_time
        n = len(td)
        if n > self._td_len:
            s = 0.0
            for i in range(self._td_len, n):
                s += float(td[i])
            self._done += s
            self._td_len = n

    def _time_index(self) -> int:
        if self._gap is None or self._gap <= 0:
            return 0
        t = float(getattr(self.env, "elapsed_seconds", 0.0))
        idx = int(t // self._gap)
        if idx < 0:
            return 0
        if idx >= self._n_steps:
            return self._n_steps - 1
        return idx

    def _best_spot_region(self, idx: int, num_regions: int) -> Optional[int]:
        best_r = None
        best_st = 0
        for r in range(num_regions):
            if idx < len(self._avail[r]) and self._avail[r][idx]:
                st = int(self._streak[r][idx])
                if st > best_st:
                    best_st = st
                    best_r = r
        return best_r

    def _next_spot_wait(self, idx: int, num_regions: int) -> int:
        # Returns min wait steps until any region has spot. Large value means none.
        inf = self._n_steps + 10
        best = inf
        for r in range(num_regions):
            nt = int(self._next_true[r][idx]) if idx < len(self._next_true[r]) else inf
            if nt < best:
                best = nt
        if best >= inf:
            return inf
        w = best - idx
        return w if w >= 0 else 0

    def _calibrate(self, idx: int, has_spot: bool) -> None:
        if not self._use_traces:
            return
        try:
            cur = int(self.env.get_current_region())
        except Exception:
            return
        if cur < 0 or cur >= len(self._avail):
            return
        if idx < 0 or idx >= len(self._avail[cur]):
            return
        pred = bool(self._avail[cur][idx])
        self._calib_obs += 1
        if pred != bool(has_spot):
            self._calib_mismatch += 1
        if self._calib_obs >= 50:
            if self._calib_mismatch / max(1, self._calib_obs) > 0.25:
                self._use_traces = False
                self._avail = []
                self._streak = []
                self._next_true = []

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._prepared:
            self._prepare()

        self._update_done()
        remaining_work = float(self.task_duration) - float(self._done)
        if remaining_work <= 0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        remaining_time = float(self.deadline) - elapsed
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        idx = self._time_index()
        self._calibrate(idx, has_spot)

        gap = float(self._gap or 1.0)
        # Conservative safety buffer.
        safety = 2.0 * float(self.restart_overhead) + 2.0 * gap

        # If we're very close to the deadline, run on-demand and avoid pauses.
        slack = remaining_time - remaining_work
        if slack <= safety:
            return ClusterType.ON_DEMAND

        # Avoid resetting overhead: if overhead is pending and current setup is feasible, keep it.
        try:
            pending_oh = float(getattr(self, "remaining_restart_overhead", 0.0))
        except Exception:
            pending_oh = 0.0

        if pending_oh > 0:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            # If spot isn't available but we were on spot, we must switch.
            return ClusterType.ON_DEMAND

        # If traces are not usable, fall back to single-region logic.
        if not self._use_traces:
            if has_spot:
                return ClusterType.SPOT
            # If slack is comfortably large, wait instead of paying on-demand.
            if slack > safety + gap:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        num_regions = int(self.env.get_num_regions())
        if num_regions <= 0:
            num_regions = len(self._avail)
        num_regions = min(num_regions, len(self._avail))

        cur_region = int(self.env.get_current_region())
        if cur_region < 0 or cur_region >= num_regions:
            cur_region = 0

        # Determine if any region has spot now.
        best_r = self._best_spot_region(idx, num_regions)

        if best_r is None:
            # No spot anywhere: decide to wait or use on-demand.
            wait_steps = self._next_spot_wait(idx, num_regions)
            if wait_steps >= self._n_steps:
                # No spot expected before deadline window.
                return ClusterType.ON_DEMAND

            # If we can afford waiting through the gap to next spot while still being safe, wait.
            wait_seconds = float(wait_steps) * gap
            # Reserve some buffer for the overhead when we resume.
            if slack > wait_seconds + safety + float(self.restart_overhead):
                # If currently on-demand, waiting might reintroduce overhead later; still can be worth it when slack is large.
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        # Spot is available in at least one region.
        # If we are on on-demand, only switch to spot if it will last long enough.
        if last_cluster_type == ClusterType.ON_DEMAND:
            best_st = int(self._streak[best_r][idx])
            if best_st < self._min_spot_switch_steps:
                return ClusterType.ON_DEMAND

        # Decide whether to switch regions for a longer spot streak.
        chosen_r = best_r
        if cur_region != best_r:
            cur_has = bool(self._avail[cur_region][idx]) if idx < len(self._avail[cur_region]) else False
            if cur_has:
                cur_st = int(self._streak[cur_region][idx])
                best_st = int(self._streak[best_r][idx])
                # Stay if current region is nearly as good to avoid overhead.
                if best_st - cur_st <= self._switch_threshold_steps:
                    chosen_r = cur_region

        if chosen_r != cur_region:
            try:
                self.env.switch_region(int(chosen_r))
            except Exception:
                # If switching fails, just stay.
                chosen_r = cur_region

        # After possibly switching, decide on spot vs on-demand for this step.
        # Trust the environment for current-region spot availability when staying put.
        if chosen_r == cur_region:
            if has_spot:
                return ClusterType.SPOT
            # Our trace may say spot but env says no; avoid error.
            if slack > safety + gap:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        # If we switched regions, rely on trace prediction.
        if idx < len(self._avail[chosen_r]) and self._avail[chosen_r][idx]:
            return ClusterType.SPOT

        # Trace predicted no spot; avoid spot to prevent error.
        if slack > safety + gap:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND