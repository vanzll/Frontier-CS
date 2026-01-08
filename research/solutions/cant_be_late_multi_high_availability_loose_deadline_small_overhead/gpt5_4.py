import json
import os
import re
from argparse import Namespace
from typing import List, Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_heuristic_v5"

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

        self._committed_to_od: bool = False
        self._safety_buffer_seconds: float = 0.0
        self._traces: List[List[int]] = []
        self._trace_len: int = 0
        self._runlen_circ: List[List[int]] = []
        self._regions: int = 0

        self._regions = 0
        try:
            trace_files = config.get("trace_files", None)
            if isinstance(trace_files, list) and len(trace_files) > 0:
                self._traces = [self._parse_trace_file(p) for p in trace_files]
                # Ensure all traces have same length; clip to min length if necessary
                lengths = [len(t) for t in self._traces if isinstance(t, list)]
                if lengths:
                    min_len = min(lengths)
                    # Avoid empty traces
                    if min_len <= 0:
                        self._traces = []
                    else:
                        self._traces = [t[:min_len] for t in self._traces]
                        self._trace_len = min_len
                        self._regions = len(self._traces)
                        self._runlen_circ = [self._compute_circular_runlen(t) for t in self._traces]
        except Exception:
            # Any failure to parse traces: fall back to reactive-only strategy
            self._traces = []
            self._runlen_circ = []
            self._trace_len = 0
            self._regions = 0

        # Safety buffer: commit to OD when time_left <= remaining_work + restart_overhead + buffer
        # Choose buffer conservatively: two steps plus two overheads
        gap = getattr(self.env, "gap_seconds", 3600.0)
        self._safety_buffer_seconds = max(2.0 * gap + 2.0 * self.restart_overhead, 1.5 * gap)

        # Maintain minimal switching hysteresis: not strictly necessary, but avoids thrashing
        self._last_switch_step: Optional[int] = None
        self._min_steps_between_switch = 1  # allow switch again after >=1 step

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already finished, do nothing
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = max(0.0, self.task_duration - work_done)
        if remaining_work <= 0:
            return ClusterType.NONE

        # Always continue on On-Demand once committed
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        gap = self.env.gap_seconds

        # Compute time needed if we start OD now (worst-case overhead on switch)
        od_switch_overhead = self.restart_overhead if last_cluster_type != ClusterType.ON_DEMAND else self.remaining_restart_overhead
        time_needed_if_od = od_switch_overhead + remaining_work

        # If we must start OD to guarantee finishing, commit and run OD
        if time_left <= time_needed_if_od + self._safety_buffer_seconds:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Prefer using SPOT when available in current region
        if has_spot and last_cluster_type != ClusterType.ON_DEMAND:
            return ClusterType.SPOT

        # If current region has no spot, consider switching to a region with spot now
        # Only switch if we have preloaded traces to check availability elsewhere
        if self._traces and self._runlen_circ and self._trace_len > 0 and last_cluster_type != ClusterType.ON_DEMAND:
            step_idx = int(elapsed // gap)
            idx = step_idx % self._trace_len

            # Hysteresis: avoid switching too frequently
            can_switch = True
            if self._last_switch_step is not None:
                if (step_idx - self._last_switch_step) < self._min_steps_between_switch:
                    can_switch = False

            # Find region(s) with availability now and choose the one with largest consecutive run length
            if can_switch:
                best_region = None
                best_run = -1
                for r in range(min(self._regions, self.env.get_num_regions())):
                    # If environment has fewer regions than traces, bounding above will handle mismatch
                    avail_now = self._traces[r][idx] == 1
                    if avail_now:
                        run_len = self._runlen_circ[r][idx]
                        # Prefer staying in current region if tie
                        if run_len > best_run or (run_len == best_run and r == self.env.get_current_region()):
                            best_run = run_len
                            best_region = r

                if best_region is not None:
                    # If we are not already in best_region, switch
                    if best_region != self.env.get_current_region():
                        self.env.switch_region(best_region)
                        self._last_switch_step = step_idx
                    # Run on spot in the chosen region
                    return ClusterType.SPOT

        # If we reached here: either no other region has spot now or we chose not to switch
        # With ample slack, it's cheaper to wait than to use On-Demand
        # But we re-check if OD is needed imminently in the next step
        # If the next step we will be forced almost surely, we could still wait one step; but commit logic above will handle at next step.
        return ClusterType.NONE

    # ------------------------ Helpers ------------------------

    def _parse_trace_file(self, path: str) -> List[int]:
        arr: Optional[List[int]] = None
        try:
            with open(path, "r") as f:
                content = f.read().strip()
        except Exception:
            return []

        # Try JSON
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                arr = obj
            elif isinstance(obj, dict):
                # Heuristics for keys
                for k in ("trace", "values", "availability", "avail", "spot", "on_off", "availability_trace", "data"):
                    if k in obj and isinstance(obj[k], list):
                        arr = obj[k]
                        break
                if arr is None:
                    # Fallback: values of dict
                    vals = list(obj.values())
                    if vals and all(isinstance(v, (int, float, bool, str)) for v in vals):
                        arr = vals
        except Exception:
            arr = None

        if arr is None:
            # Tokenize simple text: accept 0/1/true/false tokens
            tokens = re.findall(r'(?:True|False|TRUE|FALSE|true|false|0|1)', content)
            arr = [self._token_to_bit(tok) for tok in tokens]
        else:
            arr = [self._obj_to_bit(x) for x in arr]

        # Ensure non-empty
        if not arr:
            return []
        return arr

    @staticmethod
    def _token_to_bit(tok: str) -> int:
        t = tok.strip().lower()
        if t in ("1", "true", "t", "spot", "up", "avail"):
            return 1
        return 0

    @staticmethod
    def _obj_to_bit(x) -> int:
        if isinstance(x, bool):
            return 1 if x else 0
        if isinstance(x, (int, float)):
            return 1 if x != 0 else 0
        if isinstance(x, str):
            return 1 if x.strip().lower() in ("1", "true", "t", "spot", "up", "avail") else 0
        return 0

    @staticmethod
    def _compute_circular_runlen(arr: List[int]) -> List[int]:
        # Compute run length of consecutive ones starting at each index assuming array is circular
        n = len(arr)
        if n == 0:
            return []
        arr2 = arr + arr
        L = len(arr2)
        run = [0] * L
        # Backward pass
        for i in range(L - 1, -1, -1):
            if arr2[i] == 1:
                if i + 1 < L:
                    run[i] = run[i + 1] + 1
                else:
                    run[i] = 1
            else:
                run[i] = 0
        # For indices 0..n-1, cap at n
        res = [min(run[i], n) for i in range(n)]
        return res