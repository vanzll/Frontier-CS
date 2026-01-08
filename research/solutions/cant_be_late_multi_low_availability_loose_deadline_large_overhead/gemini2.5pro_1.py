import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

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
        trace_files = config.get('trace_files', [])
        for trace_file in trace_files:
            with open(trace_file) as f:
                self.spot_traces.append([bool(x) for x in json.load(f)])
        
        if not self.spot_traces:
            self.num_regions = 0
            self.total_steps = 0
            return self

        self.num_regions = len(self.spot_traces)
        self.total_steps = len(self.spot_traces[0])

        self.next_spot_step = [[-1] * self.total_steps for _ in range(self.num_regions)]
        for r in range(self.num_regions):
            next_s = -1
            for t in range(self.total_steps - 1, -1, -1):
                if self.spot_traces[r][t]:
                    next_s = t
                self.next_spot_step[r][t] = next_s

        self.spot_suffix_sum = [[0] * (self.total_steps + 1) for _ in range(self.num_regions)]
        for r in range(self.num_regions):
            for t in range(self.total_steps - 1, -1, -1):
                self.spot_suffix_sum[r][t] = int(self.spot_traces[r][t]) + self.spot_suffix_sum[r][t + 1]

        restart_overhead_steps = math.ceil(self.restart_overhead / self.env.gap_seconds)
        self.critical_slack_steps = 2.0 * (1.0 + restart_overhead_steps) + 1.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_now = self.env.elapsed_seconds
        
        if self.num_regions == 0:
            return ClusterType.ON_DEMAND
        
        if time_now >= self.total_steps * self.env.gap_seconds:
            return ClusterType.ON_DEMAND
        
        current_step = int(time_now / self.env.gap_seconds)
        
        time_needed_for_od = work_remaining + self.remaining_restart_overhead
        time_to_deadline = self.deadline - time_now
        slack = time_to_deadline - time_needed_for_od
        
        if slack / self.env.gap_seconds <= self.critical_slack_steps:
            return ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        
        if self.spot_traces[current_region][current_step]:
            return ClusterType.SPOT
        
        switch_candidates = []
        for r in range(self.num_regions):
            if self.spot_traces[r][current_step]:
                switch_candidates.append(r)

        if switch_candidates:
            best_region = -1
            max_future_spots = -1
            for r in switch_candidates:
                future_spots = self.spot_suffix_sum[r][current_step]
                if future_spots > max_future_spots:
                    max_future_spots = future_spots
                    best_region = r
            
            self.env.switch_region(best_region)
            return ClusterType.SPOT

        best_wait_region = -1
        min_wait_steps = float('inf')
        for r in range(self.num_regions):
            next_spot_t = self.next_spot_step[r][current_step]
            if next_spot_t == -1:
                continue
            
            wait_steps = next_spot_t - current_step
            if wait_steps < min_wait_steps:
                min_wait_steps = wait_steps
                best_wait_region = r
            elif wait_steps == min_wait_steps:
                if (best_wait_region != -1 and 
                    self.spot_suffix_sum[r][current_step] > self.spot_suffix_sum[best_wait_region][current_step]):
                    best_wait_region = r
        
        if best_wait_region == -1:
            return ClusterType.ON_DEMAND
        
        if best_wait_region != current_region:
            self.env.switch_region(best_wait_region)
        
        return ClusterType.NONE