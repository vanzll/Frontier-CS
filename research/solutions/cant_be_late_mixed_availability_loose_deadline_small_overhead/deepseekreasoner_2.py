import argparse
from typing import List
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97 / 3600  # $/sec
        self.ondemand_price = 3.06 / 3600  # $/sec
        self.task_duration_seconds = None
        self.spot_availability_history = []
        self.last_decision = ClusterType.NONE
        self.consecutive_spot_failures = 0
        self.max_consecutive_failures = 3
        self.spot_confidence = 1.0
        self.safety_margin_factor = 1.2
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not hasattr(self, 'task_duration_seconds'):
            self.task_duration_seconds = self.task_duration
        
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        if has_spot:
            self.consecutive_spot_failures = max(0, self.consecutive_spot_failures - 1)
        else:
            self.consecutive_spot_failures += 1
        
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration_seconds - work_done
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        if remaining_time <= 0:
            return ClusterType.NONE
        
        current_gap = self.env.gap_seconds
        effective_gap = current_gap
        
        if last_cluster_type == ClusterType.NONE and self.last_decision != ClusterType.NONE:
            effective_gap -= self.restart_overhead
        
        time_needed_with_od = remaining_work
        if last_cluster_type != ClusterType.ON_DEMAND:
            time_needed_with_od += self.restart_overhead
        
        time_needed_with_spot = remaining_work
        if last_cluster_type != ClusterType.SPOT:
            time_needed_with_spot += self.restart_overhead
        
        spot_availability_rate = sum(self.spot_availability_history) / max(1, len(self.spot_availability_history))
        
        if remaining_time <= time_needed_with_od * 1.1:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        cost_od = time_needed_with_od * self.ondemand_price
        expected_spot_time = time_needed_with_spot / max(0.01, spot_availability_rate)
        cost_spot = expected_spot_time * self.spot_price
        
        risk_factor = remaining_work / max(1, remaining_time - self.restart_overhead * 2)
        
        if has_spot and self.consecutive_spot_failures < self.max_consecutive_failures:
            if risk_factor > 0.8:
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                else:
                    return ClusterType.SPOT
            elif cost_spot < cost_od * 0.7:
                if risk_factor < 0.6:
                    return ClusterType.SPOT
                elif last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
        
        if remaining_time - current_gap < time_needed_with_od * self.safety_margin_factor:
            return ClusterType.ON_DEMAND
        
        if has_spot and self.consecutive_spot_failures < 2:
            if remaining_work > current_gap * 5:
                return ClusterType.SPOT
        
        if remaining_time > time_needed_with_od * 1.5:
            return ClusterType.NONE
        
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND
        
        if has_spot and spot_availability_rate > 0.5:
            return ClusterType.SPOT
        
        self.last_decision = ClusterType.NONE
        return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)