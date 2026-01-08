import numpy as np
from typing import Dict, Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_threshold"
    
    def __init__(self, args):
        super().__init__(args)
        self.spec_data = None
        self.spot_availability = []
        self.time_steps = []
        self.remaining_work = 0
        self.safety_buffer = 0
        self.switch_to_od_threshold = 0
        self.last_decision = ClusterType.NONE
        self.in_restart_overhead = False
        self.restart_timer = 0
        self.spot_unavailable_streak = 0
        self.spot_available_streak = 0
        self.conservative_mode = False
        
    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, 'r') as f:
                import json
                self.spec_data = json.load(f)
        except:
            self.spec_data = {}
        return self
    
    def _update_state(self, last_cluster_type: ClusterType, has_spot: bool):
        # Update remaining work
        total_done = sum(self.task_done_time) if self.task_done_time else 0
        self.remaining_work = self.task_duration - total_done
        
        # Update restart overhead state
        if self.in_restart_overhead:
            self.restart_timer -= self.env.gap_seconds
            if self.restart_timer <= 0:
                self.in_restart_overhead = False
                self.restart_timer = 0
        
        # Track spot availability streaks
        if has_spot:
            self.spot_available_streak += 1
            self.spot_unavailable_streak = 0
        else:
            self.spot_unavailable_streak += 1
            self.spot_available_streak = 0
            
        # Check if we need to enter conservative mode
        time_left = self.deadline - self.env.elapsed_seconds
        min_time_needed = self.remaining_work + self.restart_overhead
        
        if time_left < min_time_needed * 1.2:
            self.conservative_mode = True
        elif time_left > min_time_needed * 2.0:
            self.conservative_mode = False
            
        # Calculate dynamic thresholds
        time_ratio = time_left / self.deadline if self.deadline > 0 else 1.0
        work_ratio = self.remaining_work / self.task_duration if self.task_duration > 0 else 1.0
        
        # Dynamic threshold based on remaining time and work
        base_threshold = 0.3 + 0.5 * work_ratio
        time_pressure_factor = max(0.1, 1.0 - time_ratio)
        self.switch_to_od_threshold = base_threshold + time_pressure_factor * 0.4
        
        # Safety buffer for restart overheads
        expected_restarts = max(1, work_ratio * 5)
        self.safety_buffer = expected_restarts * self.restart_overhead
        
    def _should_switch_to_ondemand(self, has_spot: bool) -> bool:
        if not has_spot:
            return True
            
        time_left = self.deadline - self.env.elapsed_seconds
        min_time_needed = self.remaining_work + self.safety_buffer
        
        # Emergency switch if we're running out of time
        if time_left < min_time_needed:
            return True
            
        # Switch if spot has been unstable recently
        if self.spot_unavailable_streak > 3:
            return True
            
        # Switch in conservative mode
        if self.conservative_mode and time_left < min_time_needed * 1.5:
            return True
            
        # Random early switch to avoid getting stuck in restart loops
        if np.random.random() < 0.01 * (1.0 - time_left/self.deadline):
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state(last_cluster_type, has_spot)
        
        # Handle restart overhead state
        if self.in_restart_overhead:
            return ClusterType.NONE
        
        # Check if we were preempted
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.in_restart_overhead = True
            self.restart_timer = self.restart_overhead
            return ClusterType.NONE
        
        # Decide on next action
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If no time left or work done, do nothing
        if time_left <= 0 or self.remaining_work <= 0:
            return ClusterType.NONE
            
        # Calculate urgency
        urgency = self.remaining_work / max(time_left, 0.001)
        
        if urgency > 1.0:  # We're behind schedule
            if has_spot and not self.conservative_mode:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        elif self._should_switch_to_ondemand(has_spot):
            return ClusterType.ON_DEMAND
        elif has_spot:
            # Use spot with probability based on urgency and availability
            spot_prob = 0.8 + 0.2 * (1.0 - urgency)
            if self.spot_available_streak > 10:  # Spot has been stable
                spot_prob = min(1.0, spot_prob + 0.1)
            if np.random.random() < spot_prob:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            if urgency > 0.5 or self.conservative_mode:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)