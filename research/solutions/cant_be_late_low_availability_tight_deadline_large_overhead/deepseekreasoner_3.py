import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__()
        self.spot_price = 0.97 / 3600
        self.ondemand_price = 3.06 / 3600
        self.expected_spot_availability = 0.2
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        total_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - total_done
        time_remaining = self.deadline - elapsed
        
        if work_remaining <= 0:
            return ClusterType.NONE
            
        if time_remaining <= 0:
            return ClusterType.ON_DEMAND
            
        conservative_time_needed = work_remaining + self.restart_overhead
        if time_remaining < conservative_time_needed:
            return ClusterType.ON_DEMAND
            
        if not has_spot:
            if work_remaining <= gap:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
            
        time_slack = time_remaining - work_remaining
        
        risk_tolerance = min(1.0, time_slack / (2 * self.restart_overhead))
        if risk_tolerance < 0.2:
            return ClusterType.ON_DEMAND
            
        if risk_tolerance < 0.4 and last_cluster_type != ClusterType.SPOT:
            return ClusterType.ON_DEMAND
            
        if last_cluster_type == ClusterType.NONE and work_remaining > time_slack:
            return ClusterType.ON_DEMAND
            
        expected_spot_cost = self.spot_price * (work_remaining + self.restart_overhead / self.expected_spot_availability)
        ondemand_cost = self.ondemand_price * work_remaining
        
        if expected_spot_cost < ondemand_cost * 0.8 and risk_tolerance > 0.3:
            return ClusterType.SPOT
            
        if has_spot and risk_tolerance > 0.5:
            return ClusterType.SPOT
            
        return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)