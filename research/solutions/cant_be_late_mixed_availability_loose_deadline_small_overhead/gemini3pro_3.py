import argparse
from typing import List

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def __init__(self, args):
        super().__init__(args)

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide the next cluster type based on remaining time, work, and spot availability.
        """
        # 1. Retrieve environment state
        elapsed_seconds = self.env.elapsed_seconds
        gap_seconds = self.env.gap_seconds
        task_duration = self.task_duration
        task_done_time = self.task_done_time
        deadline = self.deadline
        restart_overhead = self.restart_overhead

        # 2. Calculate remaining work and time
        # task_done_time is a list of completed segment durations
        work_done = sum(task_done_time)
        remaining_work = max(0.0, task_duration - work_done)
        remaining_time = deadline - elapsed_seconds

        # If work is completed, stop the cluster
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # 3. Determine Safety Threshold
        # We must switch to On-Demand if the remaining time is dangerously close to 
        # the time required to complete the work plus overhead.
        #
        # Required Time = Remaining Work + Restart Overhead
        # Safety Buffer = 2 * Gap Seconds (to handle time step discretization)
        #
        # If we wait longer than this threshold, we risk missing the deadline 
        # due to the restart overhead or lack of available time.
        
        threshold = remaining_work + restart_overhead + (2.0 * gap_seconds)

        # 4. Decision Logic
        
        # Critical Zone: Deadline is approaching.
        # Force On-Demand to guarantee completion. OD is expensive but reliable.
        if remaining_time <= threshold:
            return ClusterType.ON_DEMAND

        # Safe Zone: We have plenty of slack.
        # Prioritize minimizing cost.
        if has_spot:
            # Use Spot instances (Cheapest)
            return ClusterType.SPOT
        else:
            # Spot unavailable, but we have slack.
            # Wait (NONE) rather than burning budget on On-Demand.
            # This allows us to wait out Spot unavailability windows.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)