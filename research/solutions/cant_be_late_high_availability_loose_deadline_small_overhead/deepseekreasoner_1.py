import argparse
import math
from typing import List
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self.spot_price = 0.97 / 3600
        self.ondemand_price = 3.06 / 3600
        self.time_step = 1.0
        self.safety_margin = 0.0
        self.min_spot_usage = 0.0
        self.use_aggressive_spot = True
        self.max_wait_steps = 0

    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, 'r') as f:
                content = f.read().strip()
                if content:
                    parts = content.split(',')
                    if len(parts) >= 1:
                        self.safety_margin = float(parts[0])
                    if len(parts) >= 2:
                        self.min_spot_usage = float(parts[1])
                    if len(parts) >= 3:
                        self.use_aggressive_spot = bool(int(parts[2]))
                    if len(parts) >= 4:
                        self.max_wait_steps = int(parts[3])
        except:
            pass
        
        if self.task_duration is not None:
            self.time_step = self.env.gap_seconds
            total_steps = int(self.deadline / self.time_step)
            spot_steps_needed = int(self.task_duration / self.time_step)
            self.min_spot_usage = max(self.min_spot_usage, 
                                    spot_steps_needed / total_steps if total_steps > 0 else 0)
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        completed_work = sum(segment[1] - segment[0] for segment in self.task_done_time)
        remaining_work = self.task_duration - completed_work
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND
        
        work_rate = 1.0
        restart_overhead = self.restart_overhead
        
        min_time_needed = remaining_work / work_rate
        
        if last_cluster_type != ClusterType.NONE:
            effective_work_rate = work_rate
        else:
            effective_work_rate = 0
        
        can_use_spot = has_spot
        
        if remaining_time <= min_time_needed + restart_overhead + self.safety_margin:
            return ClusterType.ON_DEMAND
        
        if not self.use_aggressive_spot:
            if remaining_time > min_time_needed * 1.5 + restart_overhead * 2:
                if can_use_spot:
                    return ClusterType.SPOT
                elif self.max_wait_steps > 0:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND
            else:
                if can_use_spot and remaining_time > min_time_needed + restart_overhead * 2:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        
        spot_steps_used = sum(1 for segment in self.task_done_time 
                            if segment[2] == ClusterType.SPOT.value)
        total_steps_used = len(self.task_done_time)
        spot_ratio = spot_steps_used / total_steps_used if total_steps_used > 0 else 0
        
        if spot_ratio < self.min_spot_usage:
            if can_use_spot:
                return ClusterType.SPOT
            elif self.max_wait_steps > 0:
                return ClusterType.NONE
        
        time_buffer = remaining_time - min_time_needed
        safe_for_spot = time_buffer > restart_overhead * 3
        
        if can_use_spot and safe_for_spot:
            return ClusterType.SPOT
        elif not can_use_spot and self.max_wait_steps > 0 and time_buffer > restart_overhead * 2:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)