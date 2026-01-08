import argparse
from enum import Enum
from typing import List
import numpy as np

class ClusterType(Enum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"
    NONE = "none"

class Strategy:
    pass

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        self.args = args
        self.spot_price = 0.97
        self.od_price = 3.06
        self.restart_overhead_seconds = 180
        self.safety_factor = 1.2
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        remaining_time = deadline - elapsed
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        work_left = self.task_duration - work_done
        
        if work_left <= 0:
            return ClusterType.NONE
            
        gap = self.env.gap_seconds
        current_type = self.env.cluster_type
        
        if last_cluster_type == ClusterType.NONE:
            last_was_running = False
        else:
            last_was_running = True
            
        if not has_spot:
            spot_possible = False
        else:
            spot_possible = True
            
        time_needed = work_left
        if current_type == ClusterType.NONE:
            time_needed += self.restart_overhead_seconds
            
        critical_time = time_needed * self.safety_factor
        
        if remaining_time <= critical_time:
            return ClusterType.ON_DEMAND
            
        if spot_possible:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
                
            expected_spot_uptime = 3600
            if remaining_time > expected_spot_uptime * 1.5:
                if np.random.random() < 0.8:
                    return ClusterType.SPOT
                else:
                    return ClusterType.NONE
            else:
                spot_ratio = self.spot_price / self.od_price
                if np.random.random() < spot_ratio:
                    return ClusterType.SPOT
                else:
                    return ClusterType.NONE
        else:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
                
            if remaining_time < time_needed * 2:
                return ClusterType.ON_DEMAND
                
            if np.random.random() < 0.3:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)