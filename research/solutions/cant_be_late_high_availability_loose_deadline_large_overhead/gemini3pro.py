from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CostOptimizedDeadlineStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the cluster type for the current step based on deadline constraints
        and cost minimization.
        """
        # Calculate total work completed
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_remaining = self.task_duration - work_done
        
        # If the task is effectively finished, stop using resources
        if work_remaining <= 1e-4:
            return ClusterType.NONE
            
        # Calculate wall-clock time remaining until the hard deadline
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate the "Must-Run" threshold (Panic Threshold).
        # We must ensure we have enough time to finish using reliable On-Demand instances
        # if we switch or start now.
        #
        # Required Time = Remaining Work + Restart Overhead (incurred when switching to OD)
        #
        # We add a safety buffer to account for:
        # 1. Discrete simulation time steps (gap_seconds)
        # 2. A safety margin for the restart overhead transition
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Buffer logic: 
        # - 1.5x overhead ensures we can comfortably cover the transition cost
        # - 2.0x gap covers granularity issues where a step might push us past the limit
        safety_buffer = (overhead * 1.5) + (gap * 2.0)
        
        panic_threshold = work_remaining + overhead + safety_buffer
        
        # CRITICAL: If we are nearing the point of no return, force On-Demand.
        # This prevents missing the deadline (-100,000 penalty).
        if time_remaining < panic_threshold:
            return ClusterType.ON_DEMAND
            
        # COST OPTIMIZATION: If we have sufficient slack (time_remaining >= panic_threshold):
        # 1. Prefer Spot instances as they are significantly cheaper (~1/3 cost).
        if has_spot:
            return ClusterType.SPOT
            
        # 2. If Spot is unavailable, wait (NONE).
        #    Since we have slack, waiting is better than burning money on On-Demand.
        #    We consume slack time in hopes that Spot becomes available later.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)