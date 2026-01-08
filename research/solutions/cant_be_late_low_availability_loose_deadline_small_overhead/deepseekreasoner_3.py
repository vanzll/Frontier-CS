import os
import json
import math
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np

# Import required for API compatibility
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class SolutionState(Enum):
    INIT = "init"
    SPOT_RUNNING = "spot_running"
    OD_RUNNING = "od_running"
    PAUSED = "paused"
    RESTART_OVERHEAD = "restart_overhead"

class Solution(Strategy):
    NAME = "adaptive_threshold"
    
    def __init__(self, args=None):
        super().__init__(args)
        self.state = SolutionState.INIT
        self.remaining_work = 0.0
        self.time_until_deadline = 0.0
        self.spot_availability_history = []
        self.spot_price = 0.97
        self.od_price = 3.06
        self.restart_overhead_seconds = 0.0
        self.spot_unavailable_streak = 0
        self.consecutive_spot_steps = 0
        self.safety_margin_factor = 1.2
        self.min_spot_ratio = 0.3
        self.aggressiveness = 0.7
        self.switch_to_od_threshold = 0.85
        self.switch_to_spot_threshold = 0.65
        
    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    spec = json.load(f)
                    if 'safety_margin' in spec:
                        self.safety_margin_factor = spec['safety_margin']
                    if 'aggressiveness' in spec:
                        self.aggressiveness = spec['aggressiveness']
                    if 'min_spot_ratio' in spec:
                        self.min_spot_ratio = spec['min_spot_ratio']
            except:
                pass
        return self
    
    def _update_state_variables(self, last_cluster_type: ClusterType, has_spot: bool) -> None:
        self.remaining_work = self.task_duration - sum(self.task_done_time)
        self.time_until_deadline = self.deadline - self.env.elapsed_seconds
        self.restart_overhead_seconds = self.restart_overhead
        
        self.spot_availability_history.append(1 if has_spot else 0)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
            
        if last_cluster_type == ClusterType.SPOT:
            self.consecutive_spot_steps += 1
            self.spot_unavailable_streak = 0
        elif last_cluster_type == ClusterType.NONE and not has_spot:
            self.spot_unavailable_streak += 1
        else:
            self.consecutive_spot_steps = 0
            if not has_spot:
                self.spot_unavailable_streak += 1
            else:
                self.spot_unavailable_streak = 0
    
    def _calculate_required_progress_rate(self) -> float:
        if self.time_until_deadline <= 0:
            return float('inf')
        return self.remaining_work / self.time_until_deadline
    
    def _calculate_spot_availability_probability(self) -> float:
        if not self.spot_availability_history:
            return 0.5
        return sum(self.spot_availability_history) / len(self.spot_availability_history)
    
    def _calculate_risk_score(self) -> float:
        progress_rate = self._calculate_required_progress_rate()
        spot_prob = self._calculate_spot_availability_probability()
        
        time_ratio = self.time_until_deadline / (self.deadline * 0.5)
        work_ratio = self.remaining_work / self.task_duration
        
        base_risk = progress_rate * (1.0 / max(spot_prob, 0.01))
        time_risk = max(0, 1.0 - time_ratio) * 2.0
        work_risk = work_ratio * 0.5
        
        return min(base_risk + time_risk + work_risk, 3.0)
    
    def _calculate_cost_benefit(self, has_spot: bool) -> Tuple[float, float]:
        spot_cost_per_second = self.spot_price / 3600.0
        od_cost_per_second = self.od_price / 3600.0
        
        spot_effective_rate = 1.0
        if has_spot:
            spot_prob = self._calculate_spot_availability_probability()
            expected_restarts = (1.0 - spot_prob) / max(spot_prob, 0.01)
            restart_penalty = expected_restarts * self.restart_overhead_seconds
            spot_effective_rate = 1.0 / (1.0 + restart_penalty / self.env.gap_seconds)
        
        spot_value = spot_effective_rate / spot_cost_per_second
        od_value = 1.0 / od_cost_per_second
        
        return spot_value, od_value
    
    def _should_switch_to_ondemand(self, has_spot: bool) -> bool:
        if not has_spot:
            return False
            
        risk_score = self._calculate_risk_score()
        progress_rate = self._calculate_required_progress_rate()
        
        if progress_rate > 0.9:
            return True
            
        if risk_score > self.switch_to_od_threshold:
            return True
            
        if self.spot_unavailable_streak > 10:
            return True
            
        remaining_time_per_work = self.time_until_deadline / max(self.remaining_work, 1.0)
        if remaining_time_per_work < 1.5:
            return True
            
        spot_prob = self._calculate_spot_availability_probability()
        expected_spot_progress = spot_prob * self.env.gap_seconds
        required_progress_per_step = self.remaining_work / (self.time_until_deadline / self.env.gap_seconds)
        
        if expected_spot_progress < required_progress_per_step * 0.8:
            return True
            
        return False
    
    def _should_use_spot(self, has_spot: bool) -> bool:
        if not has_spot:
            return False
            
        risk_score = self._calculate_risk_score()
        
        if risk_score < self.switch_to_spot_threshold:
            return True
            
        if self.consecutive_spot_steps > 20:
            return True
            
        spot_value, od_value = self._calculate_cost_benefit(has_spot)
        
        if spot_value > od_value * self.aggressiveness:
            return True
            
        min_spot_time = self.task_duration * self.min_spot_ratio
        actual_spot_time = sum(1 for t in self.task_done_time if t > 0)
        
        if actual_spot_time < min_spot_time:
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state_variables(last_cluster_type, has_spot)
        
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        if self.time_until_deadline <= 0:
            return ClusterType.NONE
        
        if self._should_switch_to_ondemand(has_spot):
            return ClusterType.ON_DEMAND
        
        if self._should_use_spot(has_spot):
            return ClusterType.SPOT
        
        if has_spot and self.consecutive_spot_steps < 5:
            return ClusterType.SPOT
        
        return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)