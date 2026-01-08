from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate work remaining
        work_done = sum(self.task_done_time)
        work_remaining = max(0.0, self.task_duration - work_done)
        
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time remaining until deadline
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate time needed to finish on OD (reliable path)
        # We include restart_overhead to account for the worst-case scenario 
        # (switching/starting an instance)
        time_needed = work_remaining + self.restart_overhead
        
        # Calculate slack (spare time)
        slack = time_remaining - time_needed
        
        # Safety buffer in seconds (2 hours)
        # If slack drops below this, we switch to On-Demand to guarantee completion.
        # This provides a margin for preemption loops or trace granularity.
        SAFE_BUFFER = 7200.0
        
        # 1. Critical Phase: Risk of missing deadline
        if slack < SAFE_BUFFER:
            return ClusterType.ON_DEMAND
        
        # 2. Economical Phase: Use Spot if available
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Waiting Phase: Spot unavailable, but plenty of slack
        # Pause and wait for Spot to return to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)