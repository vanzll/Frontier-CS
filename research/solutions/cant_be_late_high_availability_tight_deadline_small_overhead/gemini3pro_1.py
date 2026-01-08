from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate total work done and work remaining
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_remaining = max(0.0, self.task_duration - work_done)
        
        # Calculate time remaining
        time_elapsed = self.env.elapsed_seconds
        time_remaining = max(0.0, self.deadline - time_elapsed)
        
        # Calculate Slack
        # Slack is the time buffer we have before we MUST run On-Demand to meet the deadline.
        # We assume the "safe path" is running On-Demand until completion.
        # If we are not currently on On-Demand, we must account for the overhead to switch to it.
        switch_liability = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            switch_liability = self.restart_overhead
            
        min_time_needed_od = work_remaining + switch_liability
        slack = time_remaining - min_time_needed_od
        
        # Define a safety buffer
        # We must decide to switch before slack drops below zero.
        # Since decisions happen in discrete steps (gap_seconds), we need a buffer larger than the step size.
        # We also add a small margin relative to restart overhead for robustness.
        buffer = self.env.gap_seconds * 2.0 + self.restart_overhead * 0.2

        # 1. Panic Condition: If slack is low, we must use On-Demand to guarantee completion.
        if slack < buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Opportunistic Condition: If we have slack, try to use Spot.
        if has_spot:
            # Hysteresis check: If we are currently on On-Demand, avoid flapping back to Spot
            # unless the slack is large enough to justify the restart cost.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Switching to Spot incurs a restart overhead (time lost).
                # We ensure that after paying this cost, we still have enough slack.
                if slack > buffer + self.restart_overhead:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            return ClusterType.SPOT
            
        # 3. Wait Condition: Spot is unavailable, but we have plenty of slack.
        # Pause execution (ClusterType.NONE) to save money rather than burning expensive On-Demand.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)