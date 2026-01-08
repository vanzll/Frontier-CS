import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cost_optimized_strategy"

    def solve(self, spec_path: str) -> "Solution":
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
        # Calculate current state
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate safety thresholds
        # We need to guarantee completion even if we are forced to On-Demand.
        # We include a buffer of 2 time steps plus one restart overhead.
        safety_buffer = 2.0 * self.env.gap_seconds
        # self.restart_overhead is the penalty for switching/starting
        panic_threshold = work_remaining + self.restart_overhead + safety_buffer
        
        # 1. PANIC MODE: If deadline is approaching, prioritize completion over cost.
        # Force On-Demand usage to guarantee reliability.
        if time_remaining < panic_threshold:
            return ClusterType.ON_DEMAND

        # 2. SPOT AVAILABLE: If current region has Spot, use it.
        # This is the most cost-effective option.
        if has_spot:
            return ClusterType.SPOT
        
        # 3. SPOT UNAVAILABLE: Search other regions.
        # Cycle to the next region index.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        self.env.switch_region(next_region)
        
        # Decide action during transition:
        # Calculate slack: how much extra time we have beyond required work.
        slack = time_remaining - (work_remaining + self.restart_overhead)
        
        # High Slack Threshold: 5 time steps (e.g., 5 hours)
        high_slack_threshold = 5.0 * self.env.gap_seconds
        
        if slack > high_slack_threshold:
            # High slack: We can afford to wait. Return NONE to save money 
            # while "probing" the new region (we'll check availability next step).
            return ClusterType.NONE
        else:
            # Low/Moderate slack: We cannot afford to waste full steps.
            # Run On-Demand to make progress (paying higher cost) while moving to new region.
            return ClusterType.ON_DEMAND