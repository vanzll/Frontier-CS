from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CostAwareSafetyStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current state
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # Check if job is finished
        if work_remaining <= 1e-6:
            return ClusterType.NONE
            
        time_remaining = self.deadline - elapsed
        slack = time_remaining - work_remaining
        
        # Configuration and thresholds
        gap = self.env.gap_seconds if self.env.gap_seconds is not None else 60.0
        overhead = self.restart_overhead
        
        # Safety Threshold Calculation
        # We must switch to ON_DEMAND if slack becomes critically low.
        # If we are not currently running ON_DEMAND, we need to account for the restart overhead
        # required to start an ON_DEMAND instance.
        # We add a buffer (2.0 * gap) to handle discrete time steps and floating point jitter safely.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Already running OD: no overhead penalty to continue
            safety_threshold = 2.0 * gap
        else:
            # Need to start OD: overhead penalty applies
            safety_threshold = overhead + 2.0 * gap
            
        # Critical Safety Check
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Logic
        if has_spot:
            # If we are already on Spot, continue using it (most efficient state)
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            
            # If we are not on Spot, we must decide if switching is safe.
            # Switching incurs 'overhead' cost immediately.
            # We must ensure that after paying this overhead, we still have enough slack 
            # to fall back to ON_DEMAND (with its own overhead) if Spot becomes unavailable later.
            # Required slack: overhead (for current switch) + overhead (for future fallback) + buffer
            switch_safety_threshold = 2.0 * overhead + 2.0 * gap
            
            if slack > switch_safety_threshold:
                return ClusterType.SPOT
            else:
                # Spot is available but switching is too risky for the remaining slack.
                # If we are already ON_DEMAND, stay there to avoid incurring restart costs later.
                if last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                # If we are NONE, we wait.
                return ClusterType.NONE

        # No Spot available and slack is healthy.
        # Wait (NONE) to preserve budget, burning slack instead of money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)