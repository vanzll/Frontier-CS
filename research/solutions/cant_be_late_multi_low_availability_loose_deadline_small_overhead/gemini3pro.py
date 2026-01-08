import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cost_optimized_strategy"

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
        done_work = sum(self.task_done_time)
        remaining_work = self.task_duration - done_work

        # If work is effectively done, stop
        if remaining_work <= 1e-4:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        # Calculate Safety Threshold
        # We must switch to On-Demand if the remaining time is close to the minimum required time.
        # Required time = Remaining Work + Restart Overhead (in case we need to restart/switch).
        # We add a buffer of 2 time steps (gaps) to ensure we don't miss the deadline due to granularity.
        safety_buffer = (2.0 * gap) + self.restart_overhead
        min_required_time = remaining_work + safety_buffer

        if time_left < min_required_time:
            # Panic mode: Deadline is approaching. Use On-Demand to guarantee completion.
            return ClusterType.ON_DEMAND

        if has_spot:
            # Optimal case: Spot is available and we have time.
            return ClusterType.SPOT

        # Spot is unavailable in current region, but we have slack time.
        # Strategy: Switch to the next region and wait (NONE) for one step to probe availability.
        # Returning NONE avoids paying On-Demand costs while hunting for Spot.
        num_regions = self.env.get_num_regions()
        if num_regions > 1:
            current_region = self.env.get_current_region()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            return ClusterType.NONE

        # If only one region exists and Spot is down, wait (NONE) to save money 
        # (since we are above the safety threshold).
        return ClusterType.NONE