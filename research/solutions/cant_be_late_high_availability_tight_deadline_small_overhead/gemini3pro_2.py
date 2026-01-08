from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Gather state
        current_time = self.env.elapsed_seconds
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = self.task_duration - work_done
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        time_until_deadline = self.deadline - current_time
        
        # Calculate slack based on On-Demand fallback
        # If we are not currently On-Demand, switching or starting incurs restart overhead
        overhead_cost = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            overhead_cost = self.restart_overhead
            
        time_needed_od = remaining_work + overhead_cost
        slack = time_until_deadline - time_needed_od
        
        # Strategy Parameters (in seconds)
        # CRITICAL_BUFFER: 30 minutes. If slack falls below this, force On-Demand to ensure deadline.
        CRITICAL_BUFFER = 1800.0
        # WAIT_BUFFER: 60 minutes. If Spot is unavailable, we wait (NONE) only if we have > 1h slack.
        WAIT_BUFFER = 3600.0
        
        # 1. Safety Override
        if slack < CRITICAL_BUFFER:
            return ClusterType.ON_DEMAND
            
        # 2. Prefer Spot if available
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Handle Spot Unavailability
        # If plenty of slack, wait to save money. Otherwise, use On-Demand.
        if slack > WAIT_BUFFER:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)