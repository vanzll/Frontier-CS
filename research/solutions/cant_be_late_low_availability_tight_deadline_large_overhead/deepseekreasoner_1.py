import math
from typing import List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.overhead_remaining = 0
        self.current_work = 0
        self.total_work_needed = 0
        self.spot_ratio = 0.4
        self.initialized = False
        self.conservative_threshold = 0.3
        self.aggressive_threshold = 0.7
        self.switch_count = 0
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized:
            self.total_work_needed = self.task_duration
            self.initialized = True
            
        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = max(0, self.total_work_needed - work_done)
        
        effective_time_needed = remaining_work
        if last_cluster_type == ClusterType.NONE and self.overhead_remaining <= 0:
            effective_time_needed += self.restart_overhead
            
        critical_ratio = effective_time_needed / max(time_left, 0.001)
        
        if self.overhead_remaining > 0:
            self.overhead_remaining -= self.env.gap_seconds
            return ClusterType.NONE
            
        if critical_ratio > self.aggressive_threshold:
            return ClusterType.ON_DEMAND
        elif critical_ratio > self.conservative_threshold:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)