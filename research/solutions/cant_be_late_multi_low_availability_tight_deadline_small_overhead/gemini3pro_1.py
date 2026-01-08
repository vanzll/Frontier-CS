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
        gap = self.env.gap_seconds
        elapsed = self.env.elapsed_seconds
        
        # Work progress
        total_work = self.task_duration
        done_work = sum(self.task_done_time)
        work_remaining = max(0.0, total_work - done_work)
        
        # Time constraints
        time_remaining = self.deadline - elapsed
        
        # Calculate minimum time required to finish using reliable resources (On-Demand)
        # We include the current pending overhead. If we are running, this decreases.
        # If we switch to OD, we might incur overhead if we weren't running, but OD is reliable.
        # We also add a full restart overhead as a conservative buffer in case we are in a transition.
        conservative_time_needed = work_remaining + self.remaining_restart_overhead + self.restart_overhead
        
        # Calculate slack: how much time we can afford to waste searching/waiting
        slack = time_remaining - conservative_time_needed
        
        # Safety threshold: If slack is low, force On-Demand to guarantee deadline.
        # 3 * gap allows for a few hours of buffer (assuming gap ~1h) to handle overheads and edge cases.
        SAFE_BUFFER_SECONDS = 3.0 * gap
        
        if slack < SAFE_BUFFER_SECONDS:
            return ClusterType.ON_DEMAND

        # Cost Optimization Strategy
        if has_spot:
            # If Spot is available and we have slack, use it (cheapest option).
            return ClusterType.SPOT
        else:
            # Spot unavailable in current region, but we have slack.
            # Switch to the next region and wait (NONE) for one step to avoid OD costs.
            # This effectively searches for a region with Spot availability.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            return ClusterType.NONE