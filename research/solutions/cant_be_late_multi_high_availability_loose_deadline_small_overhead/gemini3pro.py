import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cant_be_late_solution"

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
        # Retrieve current state variables
        elapsed = self.env.elapsed_seconds
        done_time = sum(self.task_done_time)
        remaining_work = self.task_duration - done_time
        pending_overhead = self.remaining_restart_overhead
        gap = self.env.gap_seconds
        
        # Calculate time metrics
        time_until_deadline = self.deadline - elapsed
        
        # Minimum time needed to finish if we start running immediately
        # We must account for any pending overhead plus the actual work remaining
        time_needed = remaining_work + pending_overhead
        
        # Slack is the time buffer we have available before we strictly must run to meet deadline
        slack = time_until_deadline - time_needed
        
        # Safety Threshold:
        # If slack is low, we cannot afford to search for Spot or risk preemption.
        # We use a buffer of 2.5 time steps (approx 2.5 hours if gap is 1h) to be safe.
        # If slack < buffer, force On-Demand usage to guarantee completion.
        safety_buffer = 2.5 * gap
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND

        # If we have sufficient slack, prioritize cost optimization.
        if has_spot:
            # If Spot is available in the current region, use it.
            return ClusterType.SPOT
        else:
            # Spot is not available in the current region.
            # Since we have slack, we can afford to switch regions to find Spot availability.
            # Strategy: Cycle through regions in a round-robin fashion.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            
            # Return NONE for this step.
            # This incurs no cost (unlike OD) but consumes time (gap).
            # This allows us to "poll" the next region in the next step.
            return ClusterType.NONE