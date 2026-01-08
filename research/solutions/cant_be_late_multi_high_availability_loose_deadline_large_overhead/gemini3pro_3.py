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
        1. Calculate remaining work and time left.
        2. If time is critical (approaching deadline + buffer), force ON_DEMAND to ensure completion.
        3. If time allows:
           - If Spot is available in current region, use SPOT (cheapest).
           - If Spot is unavailable, switch to the next region and return NONE.
             This allows us to probe the next region in the following timestep without 
             incurring the cost of an On-Demand instance.
        """
        # Calculate state
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        # If work is effectively complete, pause (though simulation should stop)
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        
        # Define safety buffer:
        # We reserve a buffer of 3 hours (10800 seconds) or 10% of remaining work, whichever is larger.
        # This protects against restart overheads and ensures we have time to finish on reliable hardware
        # if Spot availability becomes terrible.
        # 3 hours is generally small compared to the 24h slack, minimizing unnecessary OD usage.
        buffer = max(10800.0, remaining_work * 0.1)
        
        # Panic Check: If we are close to the point of no return, switch to On-Demand.
        if time_left < (remaining_work + buffer):
            return ClusterType.ON_DEMAND
        
        # Normal Operation (Slack available)
        if has_spot:
            # Spot is available and we have slack -> Use Spot to minimize cost
            return ClusterType.SPOT
        else:
            # Spot is unavailable in current region.
            # Instead of paying for On-Demand or waiting blindly, we move to the next region.
            # We return NONE to incur 0 monetary cost while switching/probing.
            # Next timestep will report availability for the new region.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            return ClusterType.NONE