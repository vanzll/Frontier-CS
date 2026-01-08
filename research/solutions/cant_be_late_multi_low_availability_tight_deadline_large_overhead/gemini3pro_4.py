import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "SlackAwareStrategy"

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
        
        # If task is logically complete, stop (though env should handle this)
        if remaining_work <= 0:
            return ClusterType.NONE

        time_elapsed = self.env.elapsed_seconds
        time_left = self.deadline - time_elapsed
        
        # Calculate slack: the buffer time we have before we MUST run perfectly to finish.
        # We subtract restart_overhead to be conservative, effectively reserving time for one final setup.
        slack = time_left - remaining_work - self.restart_overhead
        
        # Define safety threshold.
        # If we use Spot and get preempted, we lose the progress of the current step (gap_seconds)
        # plus we incur restart overhead.
        # Similarly, if we switch regions and return NONE (wait), we consume gap_seconds of time.
        # If slack falls below this risk cost, we cannot afford to gamble on Spot or searching.
        risk_cost = self.env.gap_seconds + self.restart_overhead
        
        # Add a small safety margin (10%) to account for floating point inaccuracies or minor delays
        safety_threshold = risk_cost * 1.1

        # 1. Panic Mode: If slack is too low, force On-Demand to guarantee completion.
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND

        # 2. Spot Availability: If Spot is available and we have slack, use it.
        if has_spot:
            return ClusterType.SPOT
        
        # 3. Search Mode: No Spot in current region, but we have enough slack to look elsewhere.
        # Switch to the next region in a round-robin fashion.
        # We return NONE (pause) for this step because we cannot know if the new region 
        # has Spot availability immediately (blind switch). Returning SPOT blindly might error.
        current_region_idx = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region_idx = (current_region_idx + 1) % num_regions
        
        self.env.switch_region(next_region_idx)
        return ClusterType.NONE