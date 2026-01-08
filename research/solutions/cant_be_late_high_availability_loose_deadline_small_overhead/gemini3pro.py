import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "Solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide which cluster type to use for the next time step.
        """
        # Calculate remaining work
        # self.task_done_time is a list of completed work segments in seconds
        done_work = sum(self.task_done_time)
        remaining_work = self.task_duration - done_work
        
        # Check if task is effectively done
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        # Get environment variables
        current_time = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        restart_overhead = self.restart_overhead
        
        # Calculate slack: the amount of time we can afford to not be working
        # Slack = Time Remaining - Work Remaining
        # If slack is 0, we must run On-Demand continuously to finish exactly at deadline.
        time_left = deadline - current_time
        slack = time_left - remaining_work
        
        # Calculate safety buffer
        # We need to maintain enough slack to account for:
        # 1. The current step duration (gap), as we might wait (NONE) or lose progress (Spot preemption) this step.
        # 2. The restart overhead required to switch to On-Demand if we get too close to the critical path.
        #
        # We use multipliers for robustness:
        # - 2.0 * overhead covers the transition cost plus a margin for potential preemption handling.
        # - 1.5 * gap ensures we trigger the switch to On-Demand at least one step before strictly necessary.
        safety_buffer = (restart_overhead * 2.0) + (gap * 1.5)
        
        # Decision Logic
        
        # 1. Critical Phase: If slack falls below the safety buffer, we are running out of time.
        # We must use On-Demand to guarantee completion before the deadline.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Economic Phase: We have sufficient slack to take risks or wait.
        # If Spot instances are available, use them to minimize cost.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Waiting Phase: Spot is unavailable, but we still have plenty of slack.
        # Return NONE (pause) to avoid the high cost of On-Demand, waiting for Spot to return.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)