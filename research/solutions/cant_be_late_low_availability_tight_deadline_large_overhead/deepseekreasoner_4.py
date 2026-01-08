import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_threshold"

    def __init__(self, args):
        super().__init__(args)
        self.spot_available_history = []
        self.consecutive_spot_unavailable = 0
        self.time_since_last_spot = 0
        self.spot_start_time = None
        self.overhead_active = False
        self.overhead_remaining = 0.0
        self.state = "INITIAL"
        self.spot_usage_ratio = 0.0
        self.total_spot_time = 0.0
        self.total_on_demand_time = 0.0
        self.last_action = None
        self.in_overhead_period = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_overhead(self, last_cluster_type, next_type, gap_seconds):
        if last_cluster_type == ClusterType.SPOT and next_type != ClusterType.SPOT:
            self.overhead_active = True
            self.overhead_remaining = self.restart_overhead
            self.in_overhead_period = True
        elif self.overhead_active:
            if self.overhead_remaining > 0:
                self.overhead_remaining -= gap_seconds
                if self.overhead_remaining <= 0:
                    self.overhead_active = False
                    self.in_overhead_period = False

    def _should_switch_to_on_demand(self, remaining_time, remaining_work, gap_seconds):
        if remaining_work <= 0:
            return False
            
        safety_factor = 1.2
        estimated_time = remaining_work
        
        if self.overhead_active:
            estimated_time += self.overhead_remaining
            
        time_needed = estimated_time * safety_factor
        
        return remaining_time < time_needed

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        gap = self.env.gap_seconds
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        total_done = sum(self.task_done_time)
        remaining_work = self.task_duration - total_done
        
        self.spot_available_history.append(1 if has_spot else 0)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)
        
        spot_availability = sum(self.spot_available_history) / len(self.spot_available_history)
        
        self._update_overhead(last_cluster_type, last_cluster_type, gap)
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        if self._should_switch_to_on_demand(remaining_time, remaining_work, gap):
            return ClusterType.ON_DEMAND
        
        if has_spot and not self.overhead_active:
            if remaining_work > 0:
                threshold = 0.3
                
                if spot_availability < 0.2:
                    threshold = 0.7
                elif spot_availability < 0.3:
                    threshold = 0.5
                
                if remaining_time > remaining_work * 1.5:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            return ClusterType.NONE
        elif has_spot and self.overhead_active:
            return ClusterType.NONE
        else:
            if remaining_time < remaining_work * 1.1:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)