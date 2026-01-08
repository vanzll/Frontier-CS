import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        time_left = self.deadline - self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE
        
        time_needed_on_demand = work_remaining
        time_needed_on_spot = work_remaining + self.restart_overhead
        
        safety_margin = max(self.restart_overhead * 2, 3600)
        
        if time_left <= time_needed_on_demand + safety_margin:
            return ClusterType.ON_DEMAND
        
        if has_spot and time_left >= time_needed_on_spot:
            return ClusterType.SPOT
        
        if not has_spot and time_left > time_needed_on_demand + safety_margin:
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)