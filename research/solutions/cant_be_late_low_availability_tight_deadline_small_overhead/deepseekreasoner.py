import sys
import argparse
import math
from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass

class ClusterType(Enum):
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"
    NONE = "NONE"

@dataclass
class EnvState:
    elapsed_seconds: float
    gap_seconds: float
    cluster_type: ClusterType

class Strategy:
    NAME = "strategy"
    
    def __init__(self, args):
        self.args = args
        self.env = None
        self.task_duration = 0
        self.task_done_time = []
        self.deadline = 0
        self.restart_overhead = 0
        
    def solve(self, spec_path: str) -> "Strategy":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        raise NotImplementedError

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.last_decision = ClusterType.NONE
        self.work_done = 0.0
        self.overhead_remaining = 0.0
        self.consecutive_spot_failures = 0
        self.spot_availability_history = []
        self.spot_success_streak = 0
        self.critical_phase_started = False
        self.initial_work_rate = 0.0
        self.estimated_work_remaining = 0.0
        self.time_since_last_progress = 0.0
        self.conservative_mode = False
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _update_state(self, last_cluster_type: ClusterType, has_spot: bool):
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        if last_cluster_type == ClusterType.SPOT:
            if has_spot:
                self.consecutive_spot_failures = 0
                self.spot_success_streak += 1
            else:
                self.consecutive_spot_failures += 1
                self.spot_success_streak = 0
        else:
            self.consecutive_spot_failures = 0
            
        if last_cluster_type != ClusterType.NONE:
            self.time_since_last_progress = 0.0
        else:
            self.time_since_last_progress += self.env.gap_seconds
            
        total_work = sum(end - start for start, end in self.task_done_time)
        self.work_done = total_work
        self.estimated_work_remaining = self.task_duration - self.work_done
        
        time_left = self.deadline - self.env.elapsed_seconds
        required_rate = self.estimated_work_remaining / max(time_left, 0.001)
        
        if required_rate > 0.85:
            self.critical_phase_started = True
        elif required_rate < 0.6:
            self.critical_phase_started = False
            
        if required_rate > 0.95:
            self.conservative_mode = True
        elif required_rate < 0.7:
            self.conservative_mode = False
            
        if self.overhead_remaining > 0:
            self.overhead_remaining = max(0, self.overhead_remaining - self.env.gap_seconds)
    
    def _calculate_spot_availability_score(self) -> float:
        if not self.spot_availability_history:
            return 0.0
        recent_window = self.spot_availability_history[-20:] if len(self.spot_availability_history) >= 20 else self.spot_availability_history
        return sum(1 for avail in recent_window if avail) / len(recent_window)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state(last_cluster_type, has_spot)
        
        time_left = self.deadline - self.env.elapsed_seconds
        required_rate = self.estimated_work_remaining / max(time_left, 0.001)
        
        if self.estimated_work_remaining <= 0:
            return ClusterType.NONE
            
        if time_left <= 0:
            return ClusterType.NONE
        
        spot_availability = self._calculate_spot_availability_score()
        
        if self.conservative_mode:
            return self._conservative_strategy(has_spot, required_rate, time_left)
        
        if self.critical_phase_started:
            return self._critical_phase_strategy(has_spot, required_rate, time_left)
        
        return self._normal_phase_strategy(has_spot, spot_availability, required_rate, time_left)
    
    def _conservative_strategy(self, has_spot: bool, required_rate: float, time_left: float) -> ClusterType:
        if required_rate > 0.98:
            return ClusterType.ON_DEMAND
        
        if has_spot and self.overhead_remaining <= 0:
            if required_rate < 0.85:
                return ClusterType.SPOT
            elif self.consecutive_spot_failures < 2:
                return ClusterType.SPOT
        
        return ClusterType.ON_DEMAND
    
    def _critical_phase_strategy(self, has_spot: bool, required_rate: float, time_left: float) -> ClusterType:
        if required_rate > 0.9:
            return ClusterType.ON_DEMAND
        
        if has_spot and self.overhead_remaining <= 0:
            if self.spot_success_streak >= 2:
                if required_rate < 0.85:
                    return ClusterType.SPOT
                elif required_rate < 0.8 and self.consecutive_spot_failures == 0:
                    return ClusterType.SPOT
                elif time_left > self.estimated_work_remaining * 1.3:
                    return ClusterType.SPOT
        
        if self.time_since_last_progress > self.restart_overhead * 2:
            return ClusterType.ON_DEMAND
            
        return ClusterType.ON_DEMAND if required_rate > 0.75 else ClusterType.NONE
    
    def _normal_phase_strategy(self, has_spot: bool, spot_availability: float, required_rate: float, time_left: float) -> ClusterType:
        if has_spot and self.overhead_remaining <= 0:
            if spot_availability > 0.3:
                return ClusterType.SPOT
            elif spot_availability > 0.15 and self.spot_success_streak >= 1:
                return ClusterType.SPOT
            elif time_left > self.estimated_work_remaining * 1.5:
                return ClusterType.SPOT
        
        if required_rate > 0.7:
            return ClusterType.ON_DEMAND
        
        if self.time_since_last_progress > self.restart_overhead * 3:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Solution._from_args(parser)