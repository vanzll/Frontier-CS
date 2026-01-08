import json
import math
from argparse import Namespace
from array import array
from typing import List, Optional, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


def _coerce_bool_token(tok: str) -> Optional[bool]:
    t = tok.strip().lower()
    if t in ("1", "true", "t", "yes", "y"):
        return True
    if t in ("0", "false", "f", "no", "n"):
        return False
    try:
        v = float(t)
        return v > 0.5
    except Exception:
        return None


def _load_trace_to_bytearray(path: str) -> bytearray:
    # Expected formats:
    # - JSON list of 0/1 or booleans
    # - JSON dict containing a list under a key (spot/availability/trace/data)
    # - Text file with one value per line or CSV-like; last token parseable as bool/number
    try:
        if path.lower().endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for k in ("spot", "availability", "trace", "data", "values"):
                    if k in data and isinstance(data[k], list):
                        data = data[k]
                        break
                if isinstance(data, dict):
                    vals = []
                    for v in data.values():
                        if isinstance(v, list):
                            vals = v
                            break
                    data = vals

            if not isinstance(data, list):
                data = []

            out = bytearray(len(data))
            for i, v in enumerate(data):
                if isinstance(v, bool):
                    out[i] = 1 if v else 0
                elif isinstance(v, (int, float)):
                    out[i] = 1 if float(v) > 0.5 else 0
                elif isinstance(v, str):
                    b = _coerce_bool_token(v)
                    out[i] = 1 if b else 0
                else:
                    out[i] = 0
            return out
    except Exception:
        pass

    vals: List[int] = []
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").split()
                bval = None
                for tok in reversed(parts):
                    bval = _coerce_bool_token(tok)
                    if bval is not None:
                        break
                if bval is None:
                    continue
                vals.append(1 if bval else 0)
    except Exception:
        vals = []

    return bytearray(vals)


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        self._trace_files = list(config.get("trace_files", []))

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self._gap = float(getattr(self.env, "gap_seconds", 1.0))
        self._deadline = float(getattr(self, "deadline", float(config["deadline"]) * 3600.0))
        self._restart = float(getattr(self, "restart_overhead", float(config["overhead"]) * 3600.0))

        td = getattr(self, "task_duration", float(config["duration"]) * 3600.0)
        if isinstance(td, list):
            td = float(td[0]) if td else 0.0
        self._task_duration = float(td)

        self._num_regions = int(self.env.get_num_regions())
        self._work_done_sum = 0.0
        self._work_done_len = 0

        self._commit_od = False
        self._trace_enabled = False
        self._trace_mismatch_count = 0

        self._horizon_steps = int(math.ceil(self._deadline / self._gap)) + 5

        self._avail: List[bytearray] = []
        self._run_len: List[array] = []
        self._next_on: List[array] = []
        self._trace_len = 0

        self._init_parameters()

        self._try_load_and_precompute_traces()

        return self

    def _init_parameters(self) -> None:
        gap = self._gap
        restart = self._restart
        self._overhead_steps = int(math.ceil(restart / gap - 1e-12)) if gap > 0 else 0

        self._min_finish_buffer = max(2.0 * restart, 10.0 * gap, 1800.0)
        self._od_commit_slack = max(1.0 * restart, 20.0 * gap, 900.0)

        self._max_wait_sec = min(8.0 * 3600.0, max(0.0, (self._deadline - self._task_duration) * 0.9))
        self._max_wait_sec = max(self._max_wait_sec, 2.0 * gap)

        self._od_pause_wait_threshold = max(1800.0, 10.0 * gap)

        self._min_spot_run_to_switch_sec = max(3.0 * restart, 1800.0)
        self._region_switch_min_gain_sec = max(2.0 * restart, 30.0 * gap)
        self._region_switch_min_slack_sec = max(3.0 * restart, 60.0 * gap, 3600.0)

    def _try_load_and_precompute_traces(self) -> None:
        files = self._trace_files[: self._num_regions]
        if not files or len(files) < self._num_regions:
            self._trace_enabled = False
            return

        avail_all: List[bytearray] = []
        min_len = None
        for p in files:
            a = _load_trace_to_bytearray(p)
            if len(a) == 0:
                self._trace_enabled = False
                return
            avail_all.append(a)
            min_len = len(a) if min_len is None else min(min_len, len(a))

        if min_len is None or min_len == 0:
            self._trace_enabled = False
            return

        target_len = max(min_len, self._horizon_steps + 1)

        self._avail = []
        for a in avail_all:
            if len(a) < target_len:
                a = a + bytearray([0]) * (target_len - len(a))
            else:
                a = a[:target_len]
            self._avail.append(a)

        self._trace_len = target_len

        self._run_len = []
        self._next_on = []

        N = self._trace_len
        for r in range(self._num_regions):
            a = self._avail[r]

            rl = array("I", [0]) * N
            nxt = array("I", [0]) * N

            next_idx = N
            run = 0
            for i in range(N - 1, -1, -1):
                if a[i]:
                    run += 1
                    next_idx = i
                else:
                    run = 0
                rl[i] = run
                nxt[i] = next_idx

            self._run_len.append(rl)
            self._next_on.append(nxt)

        self._trace_enabled = True
        self._trace_mismatch_count = 0

    def _update_work_done(self) -> None:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return
        n = len(tdt)
        if n <= self._work_done_len:
            return
        if n == self._work_done_len + 1:
            self._work_done_sum += float(tdt[-1])
        else:
            s = 0.0
            for i in range(self._work_done_len, n):
                s += float(tdt[i])
            self._work_done_sum += s
        self._work_done_len = n

    def _time_index(self) -> int:
        gap = self._gap
        if gap <= 0:
            return 0
        return int(self.env.elapsed_seconds / gap + 1e-9)

    def _best_region_available_now(self, t: int) -> Optional[int]:
        if not self._trace_enabled or t < 0 or t >= self._trace_len:
            return None
        best_r = None
        best_rl = 0
        for r in range(self._num_regions):
            if self._avail[r][t]:
                rl = int(self._run_len[r][t])
                if best_r is None or rl > best_rl:
                    best_r = r
                    best_rl = rl
        return best_r

    def _best_future_spot(self, t: int) -> Tuple[Optional[int], int, float]:
        if not self._trace_enabled:
            return None, self._trace_len, 0.0
        if t < 0:
            t = 0
        if t >= self._trace_len:
            return None, self._trace_len, 0.0

        best_r = None
        best_next = self._trace_len
        best_run = 0

        for r in range(self._num_regions):
            nt = int(self._next_on[r][t])
            if nt >= self._trace_len:
                continue
            run = int(self._run_len[r][nt])
            if nt < best_next or (nt == best_next and run > best_run):
                best_next = nt
                best_run = run
                best_r = r

        return best_r, best_next, float(best_run) * self._gap

    def _should_commit_on_demand(self, last_cluster_type: ClusterType, time_left: float, remaining_work: float) -> bool:
        if remaining_work <= 0:
            return False
        if time_left <= 0:
            return True

        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_to_pay = float(getattr(self, "remaining_restart_overhead", 0.0))
        else:
            overhead_to_pay = self._restart

        min_time_if_od = remaining_work + max(0.0, overhead_to_pay)
        return (time_left - min_time_if_od) <= self._od_commit_slack

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_work_done()

        remaining_work = self._task_duration - self._work_done_sum
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        time_left = self._deadline - float(self.env.elapsed_seconds)
        if time_left <= 0:
            return ClusterType.NONE

        t = self._time_index()
        cur_region = int(self.env.get_current_region())
        rem_overhead = float(getattr(self, "remaining_restart_overhead", 0.0))

        if self._trace_enabled and 0 <= cur_region < self._num_regions and 0 <= t < self._trace_len:
            if bool(self._avail[cur_region][t]) != bool(has_spot):
                self._trace_mismatch_count += 1
                if self._trace_mismatch_count >= 2:
                    self._trace_enabled = False

        if not self._commit_od and self._should_commit_on_demand(last_cluster_type, time_left, remaining_work):
            self._commit_od = True

        if self._commit_od:
            return ClusterType.ON_DEMAND

        slack = time_left - remaining_work

        if rem_overhead > 1e-12:
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND

        if has_spot:
            if self._trace_enabled and slack >= self._region_switch_min_slack_sec and rem_overhead <= 1e-12 and 0 <= t < self._trace_len:
                best_r = self._best_region_available_now(t)
                if best_r is not None and best_r != cur_region:
                    cur_run = int(self._run_len[cur_region][t]) if (0 <= cur_region < self._num_regions) else 0
                    best_run = int(self._run_len[best_r][t])
                    gain_sec = float(max(0, best_run - cur_run)) * self._gap
                    if gain_sec >= self._region_switch_min_gain_sec:
                        self.env.switch_region(best_r)
                        cur_region = best_r

            if last_cluster_type == ClusterType.ON_DEMAND:
                if rem_overhead > 1e-12:
                    return ClusterType.ON_DEMAND
                if slack < (self._min_finish_buffer + self._restart):
                    return ClusterType.ON_DEMAND
                if self._trace_enabled and 0 <= cur_region < self._num_regions and 0 <= t < self._trace_len:
                    run_sec = float(int(self._run_len[cur_region][t])) * self._gap
                    if run_sec < self._min_spot_run_to_switch_sec:
                        return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        if slack <= self._min_finish_buffer:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND:
            if rem_overhead > 1e-12:
                return ClusterType.ON_DEMAND

            if self._trace_enabled:
                best_r, next_t, run_sec = self._best_future_spot(t)
                if best_r is not None and next_t < self._trace_len:
                    wait_sec = float(max(0, next_t - t)) * self._gap
                    if (
                        wait_sec <= self._od_pause_wait_threshold
                        and wait_sec <= self._max_wait_sec
                        and (slack - wait_sec) >= (self._min_finish_buffer + self._restart)
                        and run_sec >= self._min_spot_run_to_switch_sec
                    ):
                        if best_r != cur_region:
                            self.env.switch_region(best_r)
                        return ClusterType.NONE

            return ClusterType.ON_DEMAND

        if self._trace_enabled:
            best_r, next_t, _run_sec = self._best_future_spot(t)
            if best_r is not None and next_t < self._trace_len:
                wait_sec = float(max(0, next_t - t)) * self._gap
                if wait_sec <= self._max_wait_sec and wait_sec <= (slack - self._min_finish_buffer):
                    if best_r != cur_region:
                        self.env.switch_region(best_r)
                    return ClusterType.NONE

        if slack >= self._gap:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND