import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        progress_per_step = self.env.gap_seconds
        remaining_steps = remaining_work / progress_per_step
        
        if time_left <= 0:
            return ClusterType.ON_DEMAND
            
        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            else:
                safety_margin = self.restart_overhead * 2
                if time_left - safety_margin >= remaining_work:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        else:
            if last_cluster_type == ClusterType.SPOT:
                if time_left - self.restart_overhead >= remaining_work:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND
            elif last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                if time_left >= remaining_work + self.restart_overhead:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)