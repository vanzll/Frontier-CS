import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that uses a lookahead heuristic based on
    full spot availability traces.
    """

    NAME = "lookahead_heuristic"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config and load spot traces.
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self.spot_traces = self._load_traces(config["trace_files"])
        self.num_regions = len(self.spot_traces)
        if self.num_regions > 0:
            self.trace_len = len(self.spot_traces[0])
        else:
            self.trace_len = 0
            
        self._gap_seconds = None
        self._memo_spot_run = {}

        return self

    def _load_traces(self, trace_files: list[str]) -> list[list[bool]]:
        """Loads spot availability traces from JSON files."""
        traces = []
        for file_path in trace_files:
            with open(file_path) as f:
                trace_data = json.load(f)
                traces.append([bool(x) for x in trace_data])
        return traces

    def _get_spot_run(self, region_idx: int, start_step: int) -> int:
        """Calculates the length of a consecutive spot availability run with memoization."""
        if (region_idx, start_step) in self._memo_spot_run:
            return self._memo_spot_run[(region_idx, start_step)]

        run = 0
        if start_step >= self.trace_len:
            return 0
        
        # Check if the previous step's run can be reused
        if start_step > 0 and (region_idx, start_step - 1) in self._memo_spot_run:
             prev_run = self._memo_spot_run[(region_idx, start_step - 1)]
             if prev_run > 1:
                 self._memo_spot_run[(region_idx, start_step)] = prev_run - 1
                 return prev_run - 1

        for i in range(start_step, self.trace_len):
            if self.spot_traces[region_idx][i]:
                run += 1
            else:
                break
        
        self._memo_spot_run[(region_idx, start_step)] = run
        return run

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the next action based on a lookahead heuristic.
        """
        if self._gap_seconds is None:
            self._gap_seconds = self.env.gap_seconds
        gap = self._gap_seconds

        work_done = sum(self.task_done_time)
        if work_done >= self.task_duration:
            return ClusterType.NONE

        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - self.env.elapsed_seconds

        # --- Panic Mode Check ---
        work_per_step_dirty_od = gap - self.restart_overhead
        
        time_needed_panic = float('inf')
        if work_per_step_dirty_od > 1e-9:
            steps_needed = math.ceil(remaining_work / work_per_step_dirty_od)
            time_needed_panic = steps_needed * gap
        
        if remaining_time < time_needed_panic:
            return ClusterType.ON_DEMAND

        # --- Normal Mode ---
        current_step = int(self.env.elapsed_seconds / gap)
        
        if current_step >= self.trace_len:
            return ClusterType.ON_DEMAND

        # 1. If current region has spot, use it.
        if has_spot:
            return ClusterType.SPOT

        # 2. Current region has no spot. Look for the best alternative.
        best_r, best_run = -1, -1
        for r in range(self.num_regions):
            if self.spot_traces[r][current_step]:
                run = self._get_spot_run(r, current_step)
                if run > best_run:
                    best_run = run
                    best_r = r
        
        if best_r != -1:
            self.env.switch_region(best_r)
            return ClusterType.SPOT

        # 3. No spot available anywhere right now. Decide between ON_DEMAND and NONE.
        slack = remaining_time - time_needed_panic
        
        if slack > gap:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND