import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cant_be_late_strategy"

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
        1. Calculate remaining slack time.
        2. If slack is critically low (approaching deadline), force On-Demand to guarantee completion.
        3. If slack is sufficient:
           - Use Spot if available in the current region.
           - If Spot is unavailable, switch to the next region and wait (NONE) for the next step.
        """
        # Calculate remaining work
        done_time = sum(self.task_done_time)
        remaining_work = self.task_duration - done_time

        # If work is done, pause (though simulation should handle termination)
        if remaining_work <= 0:
            return ClusterType.NONE

        # Current simulation state
        current_time = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        # Calculate time remaining
        time_left = deadline - current_time

        # Calculate safety threshold for forced On-Demand
        # We need time for remaining work + potential restart overhead.
        # We add a buffer of 3 time steps (gaps) to be safe against discretization and delays.
        safety_buffer = 3.0 * gap
        min_time_required = remaining_work + overhead
        
        # CRITICAL CONDITION: If time is running out, use On-Demand immediately.
        if time_left < min_time_required + safety_buffer:
            return ClusterType.ON_DEMAND

        # OPPORTUNISTIC CONDITION: Use Spot if available.
        if has_spot:
            return ClusterType.SPOT
        
        # Spot Unavailable Handling:
        # If Spot is not available in the current region, we switch to a different region.
        # We cycle through regions round-robin.
        current_region_idx = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region_idx = (current_region_idx + 1) % num_regions
        
        self.env.switch_region(next_region_idx)
        
        # Since has_spot was False for the current step/region, we cannot run Spot now.
        # We return NONE to wait for the next step, where we will check availability in the new region.
        return ClusterType.NONE