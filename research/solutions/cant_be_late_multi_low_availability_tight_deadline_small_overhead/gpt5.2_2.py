import json
import math
import os
from argparse import Namespace
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _as_scalar(x: Any) -> float:
    if isinstance(x, (list, tuple)) and x:
        return float(x[0])
    return float(x)


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multiregion"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        self._config = config
        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._trace_files: List[str] = list(config.get("trace_files", []) or [])
        self._raw_traces: Optional[List[bytearray]] = None
        self._precomp_ready: bool = False
        self._steps: int = 0
        self._any_spot: Optional[bytearray] = None
        self._best_spot_region: Optional[array] = None  # array('B')
        self._best_future_region: Optional[array] = None  # array('B')
        self._next_any_spot: Optional[array] = None  # array('I')
        self._run_len: Optional[List[array]] = None  # per region array('I')
        self._next_spot: Optional[List[array]] = None  # per region array('I')
        self._region_mean: Optional[List[float]] = None

        self._done_len: int = 0
        self._work_done: float = 0.0

        self._commit_ondemand: bool = False
        self._no_spot_streak: int = 0
        self._last_switch_step: int = -10**18

        self._online_scores: Optional[List[float]] = None
        self._online_counts: Optional[List[int]] = None

        self._task_duration_s: float = _as_scalar(getattr(self, "task_duration", 0.0))
        self._deadline_s: float = _as_scalar(getattr(self, "deadline", 0.0))
        self._restart_overhead_s: float = _as_scalar(getattr(self, "restart_overhead", 0.0))

        try:
            self._raw_traces = self._try_load_traces(self._trace_files)
        except Exception:
            self._raw_traces = None

        return self

    def _try_load_traces(self, paths: List[str]) -> Optional[List[bytearray]]:
        if not paths:
            return None
        traces: List[bytearray] = []
        for p in paths:
            if not p or not isinstance(p, str):
                return None
            if not os.path.exists(p):
                return None
            tr = self._load_trace_file(p)
            if tr is None:
                return None
            traces.append(tr)
        return traces

    def _load_trace_file(self, path: str) -> Optional[bytearray]:
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except Exception:
            return None

        try:
            text = raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None

        if not text:
            return None

        values: List[int] = []

        def push_val(v: Any) -> None:
            try:
                if isinstance(v, bool):
                    values.append(1 if v else 0)
                elif isinstance(v, (int, float)):
                    values.append(1 if float(v) > 0.0 else 0)
                elif isinstance(v, str):
                    s = v.strip().lower()
                    if not s:
                        return
                    if s in ("true", "t", "yes", "y", "on"):
                        values.append(1)
                    elif s in ("false", "f", "no", "n", "off"):
                        values.append(0)
                    else:
                        try:
                            values.append(1 if float(s) > 0.0 else 0)
                        except Exception:
                            return
            except Exception:
                return

        if text[0] in "[{":
            try:
                obj = json.loads(text)
                seq = None
                if isinstance(obj, list):
                    seq = obj
                elif isinstance(obj, dict):
                    for k in ("availability", "avail", "spot", "trace", "data", "values", "series"):
                        if k in obj and isinstance(obj[k], list):
                            seq = obj[k]
                            break
                    if seq is None:
                        return None
                else:
                    return None

                if isinstance(seq, list):
                    for item in seq:
                        if isinstance(item, dict):
                            v = None
                            for k in ("availability", "avail", "spot", "value", "v", "is_spot", "available"):
                                if k in item:
                                    v = item[k]
                                    break
                            if v is None:
                                continue
                            push_val(v)
                        else:
                            push_val(item)
            except Exception:
                values = []

        if not values:
            # Line-based fallback parse (no regex)
            try:
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    for tok in line.replace(",", " ").split():
                        push_val(tok)
            except Exception:
                return None

        if not values:
            return None
        return bytearray(values)

    def _maybe_build_precompute(self) -> None:
        if self._precomp_ready:
            return
        if self._raw_traces is None:
            self._precomp_ready = False
            return
        if not hasattr(self, "env") or self.env is None:
            return

        try:
            gap = float(self.env.gap_seconds)
            if gap <= 0:
                return
        except Exception:
            return

        deadline_s = float(self._deadline_s)
        steps_needed = int(math.ceil(deadline_s / gap)) + 3
        if steps_needed <= 0:
            return

        traces = self._raw_traces
        num_regions = len(traces)
        if num_regions <= 0:
            return

        padded: List[bytearray] = []
        max_len = 0
        for tr in traces:
            if tr is None:
                return
            max_len = max(max_len, len(tr))
        steps = max(steps_needed, max_len)
        self._steps = steps

        # Pad / truncate traces to steps
        for tr in traces:
            if len(tr) >= steps:
                padded.append(tr[:steps])
            else:
                ext = bytearray(steps - len(tr))
                padded.append(tr + ext)

        # Precompute per region mean
        region_mean: List[float] = []
        for tr in padded:
            if len(tr) == 0:
                region_mean.append(0.0)
            else:
                region_mean.append(float(sum(tr)) / float(len(tr)))
        self._region_mean = region_mean

        # run_len and next_spot per region
        run_len: List[array] = []
        next_spot: List[array] = []
        inf = steps + 10

        for r in range(num_regions):
            tr = padded[r]
            rl = array("I", [0]) * steps
            ns = array("I", [inf]) * steps
            nxt = inf
            cur_run = 0
            for t in range(steps - 1, -1, -1):
                if tr[t]:
                    cur_run += 1
                    rl[t] = cur_run
                    nxt = t
                    ns[t] = t
                else:
                    cur_run = 0
                    rl[t] = 0
                    ns[t] = nxt
            run_len.append(rl)
            next_spot.append(ns)

        any_spot = bytearray(steps)
        best_spot_region = array("B", [0]) * steps
        best_future_region = array("B", [0]) * steps
        next_any = array("I", [inf]) * steps

        # any_spot and best region decisions
        for t in range(steps):
            best_r = 0
            best_rl = 0
            any_here = 0
            for r in range(num_regions):
                rl = run_len[r][t]
                if rl:
                    any_here = 1
                    if rl > best_rl:
                        best_rl = rl
                        best_r = r
            any_spot[t] = any_here
            best_spot_region[t] = best_r

            if any_here:
                best_future_region[t] = best_r
            else:
                # choose earliest next spot; tie-break by region mean
                best_r2 = 0
                best_nxt = inf
                best_m = -1.0
                for r in range(num_regions):
                    nxt = next_spot[r][t]
                    m = region_mean[r]
                    if nxt < best_nxt or (nxt == best_nxt and m > best_m):
                        best_nxt = nxt
                        best_r2 = r
                        best_m = m
                best_future_region[t] = best_r2

        # next_any
        nxt_any = inf
        for t in range(steps - 1, -1, -1):
            if any_spot[t]:
                nxt_any = t
            next_any[t] = nxt_any

        self._any_spot = any_spot
        self._best_spot_region = best_spot_region
        self._best_future_region = best_future_region
        self._next_any_spot = next_any
        self._run_len = run_len
        self._next_spot = next_spot
        self._precomp_ready = True

    def _update_progress(self) -> None:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return
        n = len(tdt)
        if n <= self._done_len:
            return
        inc = 0.0
        # Usually 1 element per step; sum slice to be safe but cheap
        for v in tdt[self._done_len : n]:
            try:
                inc += float(v)
            except Exception:
                pass
        self._done_len = n
        self._work_done += inc

    def _remaining_work(self) -> float:
        return max(0.0, float(self._task_duration_s) - float(self._work_done))

    def _time_left(self) -> float:
        try:
            return float(self._deadline_s) - float(self.env.elapsed_seconds)
        except Exception:
            return float(self._deadline_s)

    def _cur_step_index(self, gap: float) -> int:
        try:
            t = int(float(self.env.elapsed_seconds) // gap)
        except Exception:
            t = 0
        if self._precomp_ready and self._steps > 0:
            if t < 0:
                return 0
            if t >= self._steps:
                return self._steps - 1
        return max(0, t)

    def _should_switch_for_spot(self, cur_r: int, cand_r: int, t: int, gap: float, last_cluster_type: ClusterType) -> bool:
        if cur_r == cand_r:
            return False
        if not self._precomp_ready or self._run_len is None:
            return False
        if t < 0 or t >= self._steps:
            return False

        # Avoid switching too frequently
        if t - self._last_switch_step <= 0:
            return False

        # If we are not currently on spot (or not running), switching region is effectively "free"
        # since we'd restart anyway to launch spot.
        if last_cluster_type != ClusterType.SPOT:
            return True

        # If currently on spot, a region switch causes overhead; require significant streak gain.
        rl_cur = int(self._run_len[cur_r][t])
        rl_cand = int(self._run_len[cand_r][t])
        if rl_cand <= rl_cur:
            return False

        overhead_steps = int(math.ceil(float(self._restart_overhead_s) / gap)) if gap > 0 else 1
        # Need to gain enough consecutive steps to justify restart
        return (rl_cand - rl_cur) > (overhead_steps + 1)

    def _online_init(self) -> None:
        if self._online_scores is not None:
            return
        try:
            n = int(self.env.get_num_regions())
            if n <= 0:
                n = 1
        except Exception:
            n = 1
        self._online_scores = [0.5] * n
        self._online_counts = [0] * n

    def _online_update(self, region: int, has_spot: bool) -> None:
        self._online_init()
        if self._online_scores is None or self._online_counts is None:
            return
        if region < 0 or region >= len(self._online_scores):
            return
        a = 0.05
        x = 1.0 if has_spot else 0.0
        self._online_scores[region] = (1.0 - a) * self._online_scores[region] + a * x
        self._online_counts[region] += 1

    def _online_best_region(self) -> int:
        self._online_init()
        assert self._online_scores is not None
        best_r = 0
        best_s = self._online_scores[0]
        for r, s in enumerate(self._online_scores):
            if s > best_s:
                best_s = s
                best_r = r
        return best_r

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._maybe_build_precompute()
        self._update_progress()

        try:
            gap = float(self.env.gap_seconds)
            if gap <= 0:
                gap = 1.0
        except Exception:
            gap = 1.0

        remaining = self._remaining_work()
        if remaining <= 0.0:
            return ClusterType.NONE

        time_left = self._time_left()
        if time_left <= 0.0:
            return ClusterType.NONE

        try:
            cur_region = int(self.env.get_current_region())
        except Exception:
            cur_region = 0

        self._online_update(cur_region, has_spot)

        t = self._cur_step_index(gap)
        slack = time_left - remaining

        critical_slack = max(2.0 * float(self._restart_overhead_s) + 2.0 * gap, 0.5 * gap)
        switch_back_slack = max(4.0 * float(self._restart_overhead_s) + 2.0 * gap, 2.0 * gap)

        # If already committed, stick with on-demand to avoid penalties.
        if (not self._commit_ondemand) and slack <= critical_slack:
            self._commit_ondemand = True

        if self._commit_ondemand:
            # When switching into on-demand, optionally move to a region that's good for future spot.
            if self._precomp_ready and self._best_future_region is not None:
                if last_cluster_type != ClusterType.ON_DEMAND and float(getattr(self, "remaining_restart_overhead", 0.0)) <= 0.0:
                    trg = int(self._best_future_region[t])
                    if trg != cur_region:
                        try:
                            self.env.switch_region(trg)
                            self._last_switch_step = t
                        except Exception:
                            pass
            return ClusterType.ON_DEMAND

        # Avoid thrashing during overhead: keep running if possible.
        try:
            rem_ov = float(getattr(self, "remaining_restart_overhead", 0.0))
        except Exception:
            rem_ov = 0.0

        if rem_ov > 0.0 and last_cluster_type != ClusterType.NONE:
            if last_cluster_type == ClusterType.SPOT:
                if has_spot:
                    return ClusterType.SPOT
                # Forced off spot if unavailable; fall through to pick on-demand/none.
            elif last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND

        # Decision logic
        if has_spot:
            self._no_spot_streak = 0

            # If currently on-demand and slack isn't ample, avoid restarting to spot.
            if last_cluster_type == ClusterType.ON_DEMAND and slack <= switch_back_slack:
                return ClusterType.ON_DEMAND

            # Choose region for spot if we can.
            if self._precomp_ready and self._any_spot is not None and self._best_spot_region is not None:
                # If traces say no spot anywhere but has_spot True, don't switch.
                if t < self._steps and self._any_spot[t]:
                    cand = int(self._best_spot_region[t])
                    if cand != cur_region and rem_ov <= 0.0:
                        if self._should_switch_for_spot(cur_region, cand, t, gap, last_cluster_type):
                            try:
                                self.env.switch_region(cand)
                                self._last_switch_step = t
                                cur_region = cand
                            except Exception:
                                pass
            return ClusterType.SPOT

        # No spot (per signal) => cannot return SPOT.
        self._no_spot_streak += 1

        # If plenty of slack, wait for spot instead of paying on-demand.
        if self._precomp_ready and self._next_any_spot is not None:
            nxt = int(self._next_any_spot[t]) if t < self._steps else self._steps + 10
            if nxt < self._steps:
                wait_time = float(nxt - t) * gap
                # If we can afford to wait until next spot window plus one restart, pause.
                if slack > (wait_time + float(self._restart_overhead_s) + gap):
                    # Reposition while idle to the region that will get spot soonest.
                    if self._best_future_region is not None and last_cluster_type == ClusterType.NONE and rem_ov <= 0.0:
                        trg = int(self._best_future_region[t])
                        if trg != cur_region:
                            try:
                                self.env.switch_region(trg)
                                self._last_switch_step = t
                            except Exception:
                                pass
                    return ClusterType.NONE

        # Online heuristic without traces: if slack is large, pause and occasionally rotate.
        if (not self._precomp_ready) and slack > (6.0 * gap):
            if rem_ov <= 0.0 and self._no_spot_streak >= 3 and last_cluster_type == ClusterType.NONE:
                try:
                    nreg = int(self.env.get_num_regions())
                except Exception:
                    nreg = 1
                if nreg > 1:
                    best = self._online_best_region()
                    trg = best if best != cur_region else ((cur_region + 1) % nreg)
                    try:
                        self.env.switch_region(trg)
                        self._last_switch_step = t
                    except Exception:
                        pass
                self._no_spot_streak = 0
            return ClusterType.NONE

        # Otherwise, run on-demand. If we're restarting anyway, move to region with earliest next spot.
        if self._precomp_ready and self._best_future_region is not None:
            if last_cluster_type != ClusterType.ON_DEMAND and rem_ov <= 0.0:
                trg = int(self._best_future_region[t])
                if trg != cur_region:
                    try:
                        self.env.switch_region(trg)
                        self._last_switch_step = t
                    except Exception:
                        pass
        return ClusterType.ON_DEMAND