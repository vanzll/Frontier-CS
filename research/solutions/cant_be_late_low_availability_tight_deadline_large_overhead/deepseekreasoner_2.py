import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.remaining_work = 0.0
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.safety_margin = 0.0
        self.last_decision = ClusterType.NONE
        self.work_rate = 0.0
        self.overhead_remaining = 0.0
        
    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if hasattr(self, 'first_step'):
            elapsed = self.env.elapsed_seconds
            gap = self.env.gap_seconds
            
            if self.first_step:
                self.first_step = False
                self.remaining_work = self.task_duration
                self.safety_margin = self.restart_overhead * 2.0
                
            self.overhead_remaining = max(0.0, self.overhead_remaining - gap)
            
            completed = sum(self.task_done_time)
            self.remaining_work = max(0.0, self.task_duration - completed)
            
            time_left = self.deadline - elapsed
            work_needed = self.remaining_work
            
            if self.overhead_remaining > 0:
                time_left -= self.overhead_remaining
                if time_left <= 0:
                    return ClusterType.ON_DEMAND
            
            if work_needed <= 0:
                return ClusterType.NONE
            
            required_rate = work_needed / max(0.001, time_left)
            
            spot_available = has_spot and self.overhead_remaining <= 0
            
            critical_time = work_needed + self.restart_overhead
            
            if time_left <= critical_time + self.safety_margin:
                if self.overhead_remaining > 0:
                    return ClusterType.NONE
                return ClusterType.ON_DEMAND
            
            if spot_available:
                if required_rate > 0.9:
                    if time_left - work_needed < self.restart_overhead * 3:
                        return ClusterType.ON_DEMAND
                
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                else:
                    if time_left - work_needed > self.restart_overhead * 4:
                        self.overhead_remaining = self.restart_overhead
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
            else:
                if self.overhead_remaining > 0:
                    return ClusterType.NONE
                
                if last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                
                if time_left - work_needed < self.restart_overhead * 2:
                    return ClusterType.ON_DEMAND
                else:
                    if last_cluster_type == ClusterType.SPOT:
                        self.overhead_remaining = self.restart_overhead
                    return ClusterType.NONE
        
        else:
            self.first_step = True
            self.remaining_work = self.task_duration
            self.overhead_remaining = 0.0
            return ClusterType.NONE if not has_spot else ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)