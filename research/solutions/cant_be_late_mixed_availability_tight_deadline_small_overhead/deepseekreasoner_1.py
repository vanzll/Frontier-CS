import argparse
from typing import Optional, List
import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "dynamic_threshold_solution"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.remaining_work = 0.0
        self.time_to_deadline = 0.0
        self.spot_availability_history = []
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.spot_reliability = 0.5
        self.restart_overhead = 0.05 * 3600  # Convert hours to seconds
        self.use_spot_threshold = 0.0
        self.min_work_to_finish_on_demand = 0.0
        self.critical_time_threshold = 0.0
        self.last_spot_available = False
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.spot_available_fraction = 0.5

    def solve(self, spec_path: str) -> "Solution":
        return self

    def update_state_variables(self, last_cluster_type: ClusterType, has_spot: bool):
        self.remaining_work = self.task_duration - sum(self.task_done_time)
        self.time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        self.spot_available_fraction = (
            sum(self.spot_availability_history) / len(self.spot_availability_history)
            if self.spot_availability_history else 0.5
        )
        
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_available = 0
            self.consecutive_spot_unavailable += 1
        
        self.last_spot_available = has_spot
        
        self.min_work_to_finish_on_demand = self.remaining_work
        self.critical_time_threshold = (
            self.min_work_to_finish_on_demand + 
            2 * self.restart_overhead * self.spot_available_fraction
        )

    def calculate_dynamic_threshold(self) -> float:
        base_threshold = 0.3
        
        time_ratio = self.time_to_deadline / (self.task_duration * 1.1)
        work_ratio = self.remaining_work / self.task_duration
        
        if time_ratio < 0.1:
            return 0.9
        elif time_ratio < 0.2:
            return 0.7
        elif time_ratio < 0.3:
            return 0.5
        
        if work_ratio > 0.8:
            return 0.2
        
        recent_spot_available = (
            self.spot_available_fraction > 0.6 and 
            self.consecutive_spot_available > 5
        )
        
        if recent_spot_available:
            return max(0.1, base_threshold - 0.15)
        
        if self.consecutive_spot_unavailable > 3:
            return min(0.8, base_threshold + 0.3)
        
        return base_threshold

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.update_state_variables(last_cluster_type, has_spot)
        
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        if self.time_to_deadline <= self.min_work_to_finish_on_demand:
            return ClusterType.ON_DEMAND
        
        dynamic_threshold = self.calculate_dynamic_threshold()
        
        time_pressure = self.min_work_to_finish_on_demand / self.time_to_deadline
        
        if time_pressure > dynamic_threshold:
            return ClusterType.ON_DEMAND
        
        if not has_spot:
            if time_pressure > dynamic_threshold * 0.7:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            if self.consecutive_spot_unavailable > 2 and time_pressure > 0.2:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        if self.consecutive_spot_available >= 3:
            return ClusterType.SPOT
        
        if self.time_to_deadline > self.critical_time_threshold:
            return ClusterType.SPOT
        
        if time_pressure < 0.15 and has_spot:
            return ClusterType.SPOT
        
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)