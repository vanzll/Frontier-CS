import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import math

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.safety_margin = None
        self.critical_threshold = None
        self.min_spot_confidence = 0.6
        self.spot_history = []
        self.history_size = 100
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.last_decision = None
        self.restart_protection_active = False
        self.restart_protection_end = 0
        self.conservative_mode = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_spot_history(self, has_spot: bool):
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > self.history_size:
            self.spot_history.pop(0)
        
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0

    def _calculate_metrics(self):
        remaining_work = self.task_duration - sum(self.task_done_time)
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        work_done_ratio = sum(self.task_done_time) / self.task_duration if self.task_duration > 0 else 0
        time_ratio = elapsed / self.deadline if self.deadline > 0 else 0
        
        return remaining_work, remaining_time, work_done_ratio, time_ratio

    def _should_use_spot(self, has_spot: bool) -> bool:
        if not has_spot:
            return False
            
        remaining_work, remaining_time, work_done_ratio, time_ratio = self._calculate_metrics()
        
        if remaining_time <= 0 or remaining_work <= 0:
            return False
            
        spot_availability = sum(self.spot_history) / len(self.spot_history) if self.spot_history else 0.5
        
        if self.conservative_mode:
            return False
            
        if self.restart_protection_active and self.env.elapsed_seconds < self.restart_protection_end:
            return False
            
        time_needed_od = remaining_work
        time_needed_spot = remaining_work + (self.restart_overhead * (1 - spot_availability) * 2)
        
        if time_needed_spot > remaining_time:
            return False
            
        progress_behind = work_done_ratio < time_ratio * 0.9
        
        if progress_behind:
            if remaining_time < remaining_work * 1.2:
                return False
        
        if self.consecutive_spot_available >= 10:
            return True
            
        if spot_availability > 0.7 and self.consecutive_spot_unavailable < 3:
            return True
            
        if remaining_time > remaining_work * 1.5 and spot_availability > 0.5:
            return True
            
        return False

    def _should_use_ondemand(self) -> bool:
        remaining_work, remaining_time, work_done_ratio, time_ratio = self._calculate_metrics()
        
        if remaining_work <= 0:
            return False
            
        time_needed = remaining_work
        spot_availability = sum(self.spot_history) / len(self.spot_history) if self.spot_history else 0.5
        
        if self.restart_protection_active and self.env.elapsed_seconds < self.restart_protection_end:
            return True
            
        if self.conservative_mode:
            return True
            
        critical_time = remaining_time < remaining_work * 1.3
        
        if critical_time:
            return True
            
        if self.consecutive_spot_unavailable >= 5:
            return True
            
        if spot_availability < 0.3 and remaining_time < remaining_work * 1.8:
            return True
            
        progress_behind = work_done_ratio < time_ratio * 0.8
        if progress_behind and remaining_time < remaining_work * 2.0:
            return True
            
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_spot_history(has_spot)
        
        remaining_work, remaining_time, work_done_ratio, time_ratio = self._calculate_metrics()
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        if self.env.elapsed_seconds < self.restart_protection_end:
            self.restart_protection_active = True
        else:
            self.restart_protection_active = False
            
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.restart_protection_end = self.env.elapsed_seconds + self.restart_overhead
            self.restart_protection_active = True
            self.conservative_mode = True
        elif has_spot and self.consecutive_spot_available > 20:
            self.conservative_mode = False
            
        if time_ratio > 0.8 and work_done_ratio < 0.7:
            self.conservative_mode = True
            
        if self._should_use_ondemand():
            self.last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
            
        if self._should_use_spot(has_spot):
            self.last_decision = ClusterType.SPOT
            return ClusterType.SPOT
            
        self.last_decision = ClusterType.NONE
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)