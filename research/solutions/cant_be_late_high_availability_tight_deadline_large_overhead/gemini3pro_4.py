from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate completed work and remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If work is effectively complete, return NONE
        if work_remaining <= 0:
            return ClusterType.NONE
            
        # Current environment state
        current_time = self.env.elapsed_seconds
        time_remaining = self.deadline - current_time
        gap = self.env.gap_seconds
        
        # Conservative estimate of time needed to finish using On-Demand.
        # We include restart_overhead to ensure we can always afford the switch/start cost.
        time_needed_od = work_remaining + self.restart_overhead
        
        # Slack is the time buffer we have before we MUST start On-Demand
        slack = time_remaining - time_needed_od
        
        # Define a safety buffer based on the simulation step size.
        # We need to ensure that after this step (gap seconds), we still have non-negative slack.
        # Using 2x gap provides a robust margin for granularity and float precision.
        buffer = 2.0 * gap if gap > 0 else 60.0
        
        # Critical Path: If slack is exhausted, we must use On-Demand to guarantee deadline.
        if slack < buffer:
            return ClusterType.ON_DEMAND
            
        # If we have buffer, try to minimize cost using Spot
        if has_spot:
            # Hysteresis Logic:
            # If we are currently using On-Demand, switching to Spot incurs a restart overhead penalty.
            # We should only switch if the slack is large enough to absorb this penalty and still maintain our buffer.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > (self.restart_overhead + buffer):
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            # If we were on Spot or None, continue/start Spot
            return ClusterType.SPOT
            
        # If Spot is unavailable but we have plenty of slack, pause to save money (wait for Spot)
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)