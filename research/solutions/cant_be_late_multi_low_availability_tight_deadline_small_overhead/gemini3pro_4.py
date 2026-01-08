import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Adaptive Multi-Region Strategy that hunts for Spot availability."""

    NAME = "AdaptiveSpotHunter"

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
        Strategically switches regions if Spot is unavailable and slack permits.
        Forces On-Demand if deadline approaches.
        """
        # Calculate remaining work and time
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # If work is effectively done, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        
        # Safety parameters
        # Buffer to account for time step granularity and switching overheads.
        # 30 minutes (1800s) provides a safe margin to switch to On-Demand.
        safety_buffer = 1800.0
        
        # Calculate the absolute latest time we must start On-Demand to finish.
        # We assume worst-case restart overhead.
        required_time_for_od = remaining_work + self.restart_overhead + safety_buffer
        
        # Panic Mode: If slack is critical, force On-Demand to guarantee completion
        if time_left < required_time_for_od:
            return ClusterType.ON_DEMAND

        # Happy Path: Use Spot if available
        if has_spot:
            return ClusterType.SPOT
        
        # Search Mode: Spot is unavailable in current region, but we have slack.
        # Switch to the next region in a round-robin fashion.
        num_regions = self.env.get_num_regions()
        current_region = self.env.get_current_region()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        
        # Return NONE to pause for one step. This allows the environment to 
        # transition to the new region so we can check 'has_spot' in the next step.
        return ClusterType.NONE