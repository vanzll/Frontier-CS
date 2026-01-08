from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "DeadlineAwareStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate work remaining
        # self.task_done_time is a list of durations of completed segments
        completed_work = sum(self.task_done_time)
        work_remaining = self.task_duration - completed_work
        
        # If task is complete, stop using resources
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time remaining until deadline
        time_elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - time_elapsed
        
        # Calculate slack: The amount of time we can afford to waste (not working)
        slack = time_remaining - work_remaining
        
        # Safety Threshold Calculation
        # We must ensure that if we are running on Spot and get preempted,
        # we still have enough time to pay the restart overhead and finish on On-Demand.
        # We add a buffer of 2 time steps to account for simulation granularity.
        safety_threshold = self.restart_overhead + (self.env.gap_seconds * 2.0)
        
        # CRITICAL SAFETY CHECK:
        # If slack drops below the threshold, we cannot risk any interruptions.
        # We must use On-Demand to guarantee completion before the deadline.
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND
            
        # COST OPTIMIZATION:
        # If we have sufficient slack, prefer Spot instances to minimize cost.
        if has_spot:
            return ClusterType.SPOT
            
        # FALLBACK:
        # Spot is unavailable, but we are not in the critical zone yet.
        # However, since the deadline is tight (4h slack on 48h task), we cannot afford
        # to idle (NONE). We use On-Demand to maintain progress and preserve slack.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)