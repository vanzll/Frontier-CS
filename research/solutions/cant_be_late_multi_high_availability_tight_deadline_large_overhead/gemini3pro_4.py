import json
from argparse import Namespace

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Current state
        elapsed = self.env.elapsed_seconds
        done_work = sum(self.task_done_time)
        
        # Constants (all in seconds)
        duration = self.task_duration
        deadline = self.deadline
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        work_remaining = duration - done_work
        time_remaining = deadline - elapsed
        
        # Safety Buffer Calculation:
        # We need enough time to finish work using On-Demand (reliable).
        # Worst case time on OD = work_remaining + overhead.
        # We add a buffer of 2 * gap to allow for:
        # 1. Wasted time if a Spot instance is preempted in the current step.
        # 2. Time spent switching regions (searching) which consumes a step.
        safety_threshold = work_remaining + overhead + (2.0 * gap)
        
        # If we are close to the deadline, force On-Demand to guarantee completion.
        if time_remaining < safety_threshold:
            return ClusterType.ON_DEMAND
            
        # If we have slack, prioritize Spot (cheaper).
        if has_spot:
            return ClusterType.SPOT
        
        # If no Spot in current region and we have slack, switch region to search.
        # Use round-robin to cycle through regions.
        num_regions = self.env.get_num_regions()
        current_region = self.env.get_current_region()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        
        # Return NONE for this step. We pay the time cost (gap) but no money.
        # In the next step, we will check has_spot for the new region.
        return ClusterType.NONE