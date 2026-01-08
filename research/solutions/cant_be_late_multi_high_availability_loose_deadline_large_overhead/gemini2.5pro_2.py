import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "heuristic_streak_watcher"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
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

        self.spot_traces = []
        for trace_file in config["trace_files"]:
            try:
                with open(trace_file) as f:
                    trace = [bool(int(line.strip())) for line in f]
                self.spot_traces.append(trace)
            except (IOError, ValueError):
                self.spot_traces.append([])

        self.num_regions = len(self.spot_traces)
        if self.num_regions > 0 and self.spot_traces[0]:
            self.num_steps = len(self.spot_traces[0])
        else:
            self.num_steps = 0

        self.spot_streaks = []
        for r in range(self.num_regions):
            if not self.spot_traces[r]:
                self.spot_streaks.append([])
                continue
            
            trace = self.spot_traces[r]
            num_trace_steps = len(trace)
            streaks = [0] * num_trace_steps
            if num_trace_steps > 0:
                streaks[-1] = 1 if trace[-1] else 0
                for i in range(num_trace_steps - 2, -1, -1):
                    if trace[i]:
                        streaks[i] = streaks[i + 1] + 1
                    else:
                        streaks[i] = 0
            self.spot_streaks.append(streaks)

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_seconds = self.env.elapsed_seconds
        time_remaining = self.deadline - elapsed_seconds
        
        if time_remaining <= 0:
            return ClusterType.ON_DEMAND

        current_step = int(elapsed_seconds / self.env.gap_seconds)
        
        if current_step >= self.num_steps:
            return ClusterType.ON_DEMAND

        panic_threshold = work_remaining + self.restart_overhead + self.env.gap_seconds * 3
        if time_remaining <= panic_threshold:
            return ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        
        current_streak = 0
        if has_spot and current_step < len(self.spot_streaks[current_region]):
             current_streak = self.spot_streaks[current_region][current_step]

        best_alt_region = -1
        max_alt_streak = 0
        for r in range(self.num_regions):
            if r == current_region:
                continue
            if current_step < len(self.spot_streaks[r]):
                streak = self.spot_streaks[r][current_step]
                if streak > max_alt_streak:
                    max_alt_streak = streak
                    best_alt_region = r
        
        overhead_steps = math.ceil(self.restart_overhead / self.env.gap_seconds)
        switch_profit_margin = overhead_steps + 2

        if max_alt_streak > current_streak + switch_profit_margin:
            self.env.switch_region(best_alt_region)
            return ClusterType.SPOT
        else:
            if current_streak > 0:
                return ClusterType.SPOT
            else:
                slack = time_remaining - work_remaining
                wait_threshold = 3 * self.restart_overhead
                if slack > wait_threshold:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND