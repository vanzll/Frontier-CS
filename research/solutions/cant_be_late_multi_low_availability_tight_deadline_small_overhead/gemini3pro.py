import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "robust_greedy_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        
        Strategy:
        1. Calculate remaining work and remaining time.
        2. "Panic Mode": If time is running out (slack is low), force On-Demand to guarantee completion.
        3. "Economy Mode": If we have time, prefer Spot.
           - If Spot is available in current region, use it.
           - If Spot is NOT available, switch to the next region and wait (return NONE).
             We cannot return SPOT if has_spot is False. Searching allows us to find a region
             with availability without paying OD costs, consuming only time (slack).
        """
        # Calculate work remaining
        # self.task_done_time is a list of work segments completed in seconds
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done
        
        # If task is completed, return NONE (environment should handle termination)
        if work_left <= 0:
            return ClusterType.NONE

        # Time calculations
        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time
        
        # Calculate Panic Threshold
        # We must switch to OD if time_left is getting close to the minimum time required.
        # Minimum time = Work Remaining + Restart Overhead (to start OD).
        # We add a safety buffer to account for the discrete time steps (gap_seconds).
        # 2.0 * gap_seconds ensures we make the decision with at least one or two steps of margin.
        
        safety_buffer = 2.0 * self.env.gap_seconds
        min_time_needed = work_left + self.restart_overhead
        panic_threshold = min_time_needed + safety_buffer
        
        # 1. Panic Mode: Ensure deadline is met
        if time_left < panic_threshold:
            return ClusterType.ON_DEMAND
            
        # 2. Economy Mode: Prefer Spot
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Search Mode: Spot unavailable in current region, but we have slack.
        # Switch to next region (round-robin) and pause for one step.
        # In the next step, we will check has_spot for the new region.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        
        # Return NONE to incur no monetary cost while switching/checking.
        # This consumes 'gap_seconds' of time, reducing slack.
        return ClusterType.NONE