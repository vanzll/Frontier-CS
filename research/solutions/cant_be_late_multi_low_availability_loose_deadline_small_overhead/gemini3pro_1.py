import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Adaptive multi-region scheduling strategy that balances cost and deadline risk."""

    NAME = "AdaptiveSlackStrategy"

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
        Uses a slack-based approach:
        - If slack is low (panic mode), force On-Demand to guarantee completion.
        - If slack is high and Spot is available, use Spot.
        - If slack is high but Spot is unavailable, switch region and wait (NONE).
        """
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If task is effectively done, stop
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate slack: Time available buffer beyond what is strictly needed.
        # We subtract overhead to account for potential restart costs.
        slack = time_left - (work_remaining + overhead)
        
        # Panic Threshold:
        # If slack falls below a safety margin (approx 2 time steps), we must ensure progress.
        # On-Demand is expensive but reliable.
        PANIC_THRESHOLD = 2.0 * gap
        
        if slack < PANIC_THRESHOLD:
            return ClusterType.ON_DEMAND
            
        # If we have sufficient slack, prioritize Spot instances (cheaper)
        if has_spot:
            return ClusterType.SPOT
            
        # If no Spot in current region but we have slack:
        # Switch to the next region to search for Spot capacity.
        # Return NONE for this step to avoid paying OD cost while switching/searching.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        return ClusterType.NONE