import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97
        self.od_price = 3.06
        self.price_ratio = self.spot_price / self.od_price
        self.restart_hours = 0.20
        self.task_hours = 48.0
        self.deadline_hours = 70.0
        self.slack_hours = self.deadline_hours - self.task_hours
        self.max_restarts = int(self.slack_hours / self.restart_hours) - 1
        self.restart_count = 0
        self.consecutive_spot_failures = 0
        self.spot_availability_history = []
        self.in_overhead = False
        self.overhead_remaining = 0.0
        self.work_remaining = 0.0
        self.time_remaining = 0.0
        self.risk_tolerance = 0.7

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_state(self):
        self.work_remaining = self.task_duration - sum(self.task_done_time)
        self.time_remaining = self.deadline - self.env.elapsed_seconds
        
        if self.in_overhead:
            self.overhead_remaining -= self.env.gap_seconds
            if self.overhead_remaining <= 0:
                self.in_overhead = False
                self.overhead_remaining = 0.0

    def _calculate_safety_margin(self):
        critical_ratio = self.time_remaining / (self.work_remaining + 0.001)
        
        if critical_ratio < 1.2:
            return 0.0
        elif critical_ratio < 1.5:
            return self.restart_overhead * 0.5
        else:
            return self.restart_overhead * 2.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state()
        
        if self.work_remaining <= 0:
            return ClusterType.NONE
            
        if self.time_remaining <= 0:
            return ClusterType.NONE
        
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        spot_availability = np.mean(self.spot_availability_history) if self.spot_availability_history else 0.5
        
        if last_cluster_type == ClusterType.NONE:
            self.consecutive_spot_failures = 0
        elif last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = 0
        
        if self.in_overhead:
            return ClusterType.NONE
        
        time_needed = self.work_remaining / 3600.0
        time_available = self.time_remaining / 3600.0
        
        if time_needed > time_available:
            return ClusterType.ON_DEMAND
        
        safety_margin = self._calculate_safety_margin()
        safe_time_needed = time_needed + safety_margin / 3600.0
        
        if safe_time_needed > time_available:
            return ClusterType.ON_DEMAND
        
        if self.consecutive_spot_failures >= 3:
            return ClusterType.ON_DEMAND
        
        spot_success_prob = spot_availability ** max(1, int(time_needed / 0.5))
        
        expected_spot_cost = time_needed * self.spot_price
        expected_od_cost = time_needed * self.od_price
        expected_hybrid_cost = (time_needed * spot_success_prob * self.spot_price + 
                               time_needed * (1 - spot_success_prob) * self.od_price)
        
        cost_saving = (expected_od_cost - expected_hybrid_cost) / expected_od_cost
        
        if (has_spot and spot_success_prob > self.risk_tolerance and 
            cost_saving > 0.1 and self.restart_count < self.max_restarts):
            
            if last_cluster_type != ClusterType.SPOT and last_cluster_type != ClusterType.NONE:
                self.in_overhead = True
                self.overhead_remaining = self.restart_overhead
                self.restart_count += 1
                return ClusterType.NONE
            
            return ClusterType.SPOT
        
        if time_available - time_needed < self.restart_hours:
            return ClusterType.ON_DEMAND
        
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)