import os
import json
import math
from typing import Dict, List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cautious_spot_optimizer"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_availability_history = []
        self.spot_unavailable_streak = 0
        self.max_spot_unavailable = 0
        self.conservative_threshold = 0.15
        self.spot_confidence = 0.7
        self.last_decision = ClusterType.NONE
        
    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    config = json.load(f)
                    self.conservative_threshold = config.get('conservative_threshold', 0.15)
                    self.spot_confidence = config.get('spot_confidence', 0.7)
            except:
                pass
        return self
    
    def _calculate_time_metrics(self) -> Tuple[float, float, float]:
        total_work_done = sum(self.task_done_time)
        work_remaining = max(0, self.task_duration - total_work_done)
        
        time_remaining = max(0, self.deadline - self.env.elapsed_seconds)
        
        if work_remaining <= 0:
            progress_ratio = 1.0
        elif self.task_duration > 0:
            progress_ratio = total_work_done / self.task_duration
        else:
            progress_ratio = 0.0
            
        return work_remaining, time_remaining, progress_ratio
    
    def _should_use_spot(self, has_spot: bool, time_remaining: float, 
                        work_remaining: float, progress_ratio: float) -> bool:
        if not has_spot:
            return False
            
        if progress_ratio > 0.9:
            time_needed = work_remaining + self.restart_overhead
            buffer_needed = 2 * self.restart_overhead
            if time_remaining < time_needed + buffer_needed:
                return False
        
        if time_remaining <= work_remaining + 2 * self.restart_overhead:
            return False
            
        if self.env.elapsed_seconds < 3600:
            return True
            
        if len(self.spot_availability_history) > 10:
            recent_availability = sum(self.spot_availability_history[-10:]) / 10.0
            if recent_availability < 0.3:
                return False
                
        return True
    
    def _should_use_ondemand(self, work_remaining: float, time_remaining: float) -> bool:
        if time_remaining <= 0:
            return False
            
        if work_remaining <= 0:
            return False
            
        time_needed_with_overhead = work_remaining + self.restart_overhead
        
        if time_remaining < time_needed_with_overhead * 1.2:
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.spot_availability_history.append(1.0 if has_spot else 0.0)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        if has_spot:
            self.spot_unavailable_streak = 0
        else:
            self.spot_unavailable_streak += 1
            self.max_spot_unavailable = max(self.max_spot_unavailable, 
                                          self.spot_unavailable_streak)
        
        work_remaining, time_remaining, progress_ratio = self._calculate_time_metrics()
        
        if work_remaining <= 0:
            return ClusterType.NONE
            
        if time_remaining <= 0:
            return ClusterType.NONE
            
        if last_cluster_type == ClusterType.NONE:
            use_spot = self._should_use_spot(has_spot, time_remaining, 
                                           work_remaining, progress_ratio)
            if use_spot:
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            if self._should_use_ondemand(work_remaining, time_remaining):
                self.last_decision = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND
            else:
                self.last_decision = ClusterType.NONE
                return ClusterType.NONE
        
        if self._should_use_ondemand(work_remaining, time_remaining):
            self.last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        if has_spot:
            use_spot = self._should_use_spot(has_spot, time_remaining, 
                                           work_remaining, progress_ratio)
            if use_spot:
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
        
        if last_cluster_type == ClusterType.NONE:
            if has_spot and self._should_use_spot(has_spot, time_remaining, 
                                                work_remaining, progress_ratio):
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
        
        self.last_decision = ClusterType.NONE
        return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)