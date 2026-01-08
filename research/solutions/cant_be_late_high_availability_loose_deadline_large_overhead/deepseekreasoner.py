import sys
import os
from typing import List, Tuple, Optional
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args=None):
        super().__init__(args)
        self.remaining_work = 0.0
        self.remaining_time = 0.0
        self.last_decision = ClusterType.NONE
        self.restart_timer = 0.0
        self.spot_available_history = []
        self.consecutive_spot_failures = 0
        self.spot_utilization = 0.0
        self.aggressiveness = 1.0
        self.critical_mode = False
        
    def solve(self, spec_path: str) -> "Solution":
        # Read spec if needed
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    pass  # Read configuration if needed
            except:
                pass
        
        # Initialize state
        self.remaining_work = self.task_duration
        self.remaining_time = self.deadline
        self.last_decision = ClusterType.NONE
        self.restart_timer = 0.0
        self.spot_available_history = []
        self.consecutive_spot_failures = 0
        self.spot_utilization = 0.0
        self.aggressiveness = 1.0
        self.critical_mode = False
        
        return self
    
    def _update_state(self, has_spot: bool):
        # Update remaining work
        if self.task_done_time:
            self.remaining_work = self.task_duration - sum(self.task_done_time)
        else:
            self.remaining_work = self.task_duration
            
        # Update remaining time
        self.remaining_time = self.deadline - self.env.elapsed_seconds
        
        # Track spot availability history
        self.spot_available_history.append(has_spot)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)
            
        # Update spot utilization
        if self.spot_available_history:
            self.spot_utilization = sum(self.spot_available_history) / len(self.spot_available_history)
        
        # Update consecutive spot failures
        if not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = 0
            
        # Update restart timer
        if self.restart_timer > 0:
            self.restart_timer -= self.env.gap_seconds
            if self.restart_timer < 0:
                self.restart_timer = 0.0
        
        # Check if we need to enter critical mode
        self.critical_mode = self._should_enter_critical_mode()
        
        # Adjust aggressiveness based on progress
        progress_ratio = 1.0 - (self.remaining_work / self.task_duration)
        time_ratio = 1.0 - (self.remaining_time / self.deadline)
        
        if progress_ratio < 0.8 and time_ratio < 0.8:
            self.aggressiveness = max(0.7, 1.0 - (time_ratio - progress_ratio))
        else:
            self.aggressiveness = 1.0
    
    def _should_enter_critical_mode(self) -> bool:
        if self.remaining_work <= 0:
            return False
            
        # Calculate minimum time needed with on-demand
        min_time_needed = self.remaining_work
        if self.restart_timer > 0:
            min_time_needed += self.restart_timer
            
        # Calculate slack
        slack = self.remaining_time - min_time_needed
        
        # Enter critical mode if slack is less than 2 restart periods
        return slack < (2 * self.restart_overhead)
    
    def _spot_safety_score(self, has_spot: bool) -> float:
        if not has_spot:
            return 0.0
            
        # Consider recent history
        if len(self.spot_available_history) < 10:
            return 0.5
            
        recent_history = self.spot_available_history[-10:]
        recent_success_rate = sum(recent_history) / len(recent_history)
        
        # Penalize if we had recent failures
        if self.consecutive_spot_failures > 0:
            safety = recent_success_rate * 0.7
        else:
            safety = recent_success_rate
            
        # Adjust by global utilization
        safety = (safety + self.spot_utilization) / 2.0
        
        # Apply aggressiveness factor
        return min(1.0, safety * self.aggressiveness)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update internal state
        self._update_state(has_spot)
        self.last_decision = last_cluster_type
        
        # If no work left, do nothing
        if self.remaining_work <= 0:
            return ClusterType.NONE
            
        # If in restart period, wait
        if self.restart_timer > 0:
            return ClusterType.NONE
        
        # Critical mode: use on-demand to guarantee completion
        if self.critical_mode:
            return ClusterType.ON_DEMAND
        
        # Calculate progress rate needed
        needed_rate = self.remaining_work / max(self.remaining_time, 0.001)
        
        # Calculate safe spot rate
        spot_safety = self._spot_safety_score(has_spot)
        
        # Decision logic
        if has_spot:
            # Use spot if safe enough or we have good progress
            if spot_safety > 0.6 or needed_rate < 0.8:
                # Check if we're switching from on-demand (no restart needed)
                if last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.SPOT
                # Check if we were already on spot
                elif last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                # Starting fresh spot instance - will incur restart
                else:
                    # Only start spot if we have time for restart
                    if self.remaining_time - self.restart_overhead > self.remaining_work * 1.2:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND if needed_rate > 0.9 else ClusterType.NONE
            else:
                # Spot not safe enough
                if needed_rate > 0.9:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE
        else:
            # Spot not available
            if needed_rate > 0.8:
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)