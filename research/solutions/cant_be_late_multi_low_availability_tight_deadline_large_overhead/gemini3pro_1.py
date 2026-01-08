import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "hunt_and_panic"

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        Strategy:
        1. Panic Mode: If time remaining is close to the minimum needed to finish on On-Demand,
           force On-Demand to guarantee deadline.
        2. Hunt Mode: If time is sufficient, prefer Spot.
           - If current region has Spot: Use it.
           - If current region has NO Spot: Switch to next region and wait (return NONE) 
             to check its availability in the next step.
        """
        # Safety buffer to account for simulation noise and overhead risks (30 minutes)
        safety_buffer = 1800.0

        # Calculate progress
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds

        # Determine if we must switch to On-Demand to meet deadline.
        # We need enough time for: Remaining Work + Restart Overhead (to launch OD) + Buffer
        time_needed_for_od = work_remaining + self.restart_overhead + safety_buffer

        if time_left < time_needed_for_od:
            return ClusterType.ON_DEMAND

        # If we have slack, prioritize Spot (cheaper)
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot is unavailable in current region.
            # Switch to the next region (Round-Robin) to search for availability.
            # We must return NONE for this step as we cannot verify availability
            # of the new region until the next step.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            return ClusterType.NONE