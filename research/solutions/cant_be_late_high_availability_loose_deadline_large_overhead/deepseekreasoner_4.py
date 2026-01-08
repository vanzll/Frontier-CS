import sys
from enum import Enum
from typing import List, Optional
import numpy as np

class ClusterType(Enum):
    SPOT = 0
    ON_DEMAND = 1
    NONE = 2

class Strategy:
    def __init__(self, args):
        self.env = None
        self.task_duration = 0
        self.task_done_time = []
        self.deadline = 0
        self.restart_overhead = 0
        self.args = args
    
    def solve(self, spec_path: str) -> "Strategy":
        return self

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97
        self.od_price = 3.06
        self.initialized = False
        self.remaining_work = 0.0
        self.last_spot_available = True
        self.in_overhead = 0.0
        self.consecutive_spot_failures = 0
        self.spot_history = []
        self.current_progress = 0.0
        self.time_elapsed = 0.0
        self.spot_availability_rate = 0.6
    
    def solve(self, spec_path: str) -> "Solution":
        self.initialized = True
        return self
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
    
    def _calculate_required_od_time(self, remaining_time, remaining_work):
        if remaining_time <= 0:
            return float('inf')
        if remaining_work <= 0:
            return 0.0
        work_rate = 1.0
        overhead_penalty = 0.0
        
        if remaining_work / work_rate > remaining_time:
            return remaining_time
        
        min_od_time = max(0.0, remaining_work - (remaining_time * self.spot_availability_rate))
        return min_od_time / work_rate
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.time_elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 100:
            self.spot_history.pop(0)
        
        if len(self.spot_history) > 10:
            self.spot_availability_rate = sum(self.spot_history) / len(self.spot_history)
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
            if self.in_overhead <= 0:
                self.in_overhead = self.restart_overhead
        else:
            self.consecutive_spot_failures = 0
        
        if self.in_overhead > 0:
            self.in_overhead = max(0, self.in_overhead - gap)
        
        total_done = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - total_done)
        time_left = self.deadline - self.time_elapsed
        
        if time_left <= 0 or remaining_work <= 0:
            return ClusterType.NONE
        
        progress_rate_needed = remaining_work / time_left
        
        safe_margin = 2 * self.restart_overhead
        conservative_time_left = time_left - safe_margin
        
        if conservative_time_left <= 0:
            return ClusterType.ON_DEMAND
        
        min_required_od_time = self._calculate_required_od_time(conservative_time_left, remaining_work)
        time_critical = min_required_od_time > 0.1 * conservative_time_left
        
        if time_critical or progress_rate_needed > 0.9:
            if self.in_overhead > 0:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT
        
        risk_factor = min(1.0, max(0.0, (remaining_work / time_left) / 0.5))
        use_spot_prob = 1.0 - risk_factor
        
        expected_spot_available = self.spot_availability_rate
        
        spot_utility = expected_spot_available * use_spot_prob
        
        if spot_utility > 0.7 and has_spot and self.in_overhead <= 0:
            if self.consecutive_spot_failures < 3:
                return ClusterType.SPOT
        
        if time_left < 4 * self.restart_overhead:
            if has_spot and self.in_overhead <= 0:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        if has_spot and self.in_overhead <= 0 and spot_utility > 0.4:
            if self.consecutive_spot_failures < 5:
                return ClusterType.SPOT
        
        if self.in_overhead > 0:
            if time_left < remaining_work + self.in_overhead:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        if not has_spot:
            if time_left < remaining_work + 2 * self.restart_overhead:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND