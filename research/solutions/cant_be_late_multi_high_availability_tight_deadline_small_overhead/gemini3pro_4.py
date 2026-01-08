import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cost_aware_scheduler"

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
        Prioritizes finishing before deadline, then minimizes cost by using Spot instances.
        """
        # 1. Calculate remaining work
        done_time = sum(self.task_done_time)
        work_remaining = self.task_duration - done_time
        
        # If task is done, pause (environment should handle termination)
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        # 2. Safety Check
        # Ensure we switch to On-Demand with enough time to complete the task
        time_elapsed = self.env.elapsed_seconds
        time_left = self.deadline - time_elapsed
        
        # Buffer logic:
        # We need time for:
        # - The remaining work
        # - The restart overhead (in case we are not currently running or need to switch types)
        # - A safety margin covering at least a few timesteps to account for granularity
        # Using 2.0 * gap_seconds ensures we catch the deadline safely.
        buffer = max(self.env.gap_seconds * 2.0, self.restart_overhead * 2.0)
        
        # If remaining time is tight, strictly use On-Demand to guarantee completion.
        if time_left < (work_remaining + self.restart_overhead + buffer):
            return ClusterType.ON_DEMAND

        # 3. Cost Optimization (Slack Mode)
        # We have enough time to try Spot instances.
        if has_spot:
            # Spot is available in the current region; use it to save cost.
            return ClusterType.SPOT
        else:
            # Spot is unavailable in the current region.
            # Since we have slack, we hunt for Spot in other regions instead of paying for On-Demand.
            
            # Switch to the next region (Round Robin search)
            num_regions = self.env.get_num_regions()
            current_region = self.env.get_current_region()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            
            # Return NONE for this step.
            # We cannot return SPOT because we don't know availability in the new region yet.
            # We don't want to return ON_DEMAND because we are trying to save money.
            # This incurs a time penalty (one gap) + restart overhead on the next effective run,
            # but preserves the budget.
            return ClusterType.NONE