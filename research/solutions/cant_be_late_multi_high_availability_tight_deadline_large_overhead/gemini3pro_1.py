import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"

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

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        # If task is effectively done, return NONE
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # Retrieve environment parameters
        current_time = self.env.elapsed_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Calculate time needed to finish using On-Demand (safe path)
        # We need to cover remaining work plus any potential restart overhead to switch to OD
        time_needed_od = remaining_work + overhead
        
        # Safety buffer to account for:
        # 1. Discrete time steps (decision granularity)
        # 2. Potential overheads and switching delays
        # 2.0 * gap is a conservative buffer to ensure we catch the deadline window
        safety_buffer = 2.0 * gap
        
        time_left = deadline - current_time
        
        # Panic Mode: If time remaining is tight, force On-Demand usage.
        # This prevents missing the deadline which has a huge penalty (-100000).
        # We ensure we have enough time for the work + overhead + a safety margin.
        if time_left < (time_needed_od + safety_buffer):
            return ClusterType.ON_DEMAND

        # Economy Mode: Try to use Spot instances to save cost (ratio ~1:3 vs OD)
        if has_spot:
            # Current region has Spot availability
            return ClusterType.SPOT
        else:
            # Current region does not have Spot availability.
            # Strategy: Switch to the next region and wait 1 step to probe its availability.
            # We cycle through regions round-robin until we find one with Spot or hit Panic Mode.
            
            num_regions = self.env.get_num_regions()
            current_region_idx = self.env.get_current_region()
            next_region_idx = (current_region_idx + 1) % num_regions
            
            self.env.switch_region(next_region_idx)
            
            # Return NONE because we cannot assume Spot availability in the new region 
            # until the next time step (when `has_spot` will be updated for the new region).
            return ClusterType.NONE