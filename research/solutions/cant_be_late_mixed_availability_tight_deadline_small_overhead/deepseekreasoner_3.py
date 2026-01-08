import os
import json
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args=None):
        super().__init__(args)
        self.config = None
        self.spot_price = 0.97
        self.on_demand_price = 3.06
        self.price_ratio = self.spot_price / self.on_demand_price
        self.remaining_work = 0.0
        self.remaining_time = 0.0
        self.time_step = 0.0
        self.restart_timer = 0.0
        self.spot_available_ratio = 0.5
        self.spot_unavailable_streak = 0
        self.max_spot_unavailable_streak = 0
        self.consecutive_on_demand = 0
        self.switch_to_od_threshold = 0.15
        self.switch_to_spot_threshold = 0.3
        self.emergency_threshold = 0.05
        self.min_spot_burst = 5
        self.conservative_mode = False
        self.last_action = ClusterType.NONE
        self.work_done = 0.0
        
    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                self.config = json.load(f)
        return self
    
    def _update_state(self, last_cluster_type, has_spot):
        self.time_step = self.env.gap_seconds
        elapsed = self.env.elapsed_seconds
        
        # Calculate remaining work
        self.work_done = sum(end - start for start, end in self.task_done_time)
        self.remaining_work = max(0.0, self.task_duration - self.work_done)
        self.remaining_time = max(0.0, self.deadline - elapsed)
        
        # Update restart timer
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.restart_timer = self.restart_overhead
        elif self.restart_timer > 0:
            self.restart_timer = max(0.0, self.restart_timer - self.time_step)
        
        # Track spot availability pattern
        if not has_spot:
            self.spot_unavailable_streak += 1
            self.max_spot_unavailable_streak = max(
                self.max_spot_unavailable_streak, self.spot_unavailable_streak
            )
        else:
            self.spot_unavailable_streak = 0
            
        # Track consecutive on-demand usage
        if last_cluster_type == ClusterType.ON_DEMAND:
            self.consecutive_on_demand += 1
        else:
            self.consecutive_on_demand = 0
            
        # Update conservative mode based on remaining time
        time_ratio = self.remaining_work / max(0.1, self.remaining_time - self.restart_timer)
        self.conservative_mode = time_ratio > 1.2
        
    def _calculate_criticality(self):
        if self.remaining_time <= 0:
            return float('inf')
        
        # Adjusted remaining work accounting for potential restart
        adjusted_work = self.remaining_work
        if self.restart_timer > 0:
            adjusted_work += self.time_step  # Conservative estimate
            
        time_needed = adjusted_work
        safety_margin = self.restart_overhead * 2
        
        return (time_needed + safety_margin) / max(0.1, self.remaining_time)
    
    def _should_use_spot(self, has_spot, criticality):
        if not has_spot:
            return False
            
        if self.restart_timer > 0:
            return False
            
        if criticality > 1.5:
            return False
            
        if criticality > self.switch_to_od_threshold and self.spot_unavailable_streak > 3:
            return False
            
        if self.conservative_mode and criticality > 0.8:
            return False
            
        # Ensure we use spot for at least min_spot_burst steps to avoid frequent restarts
        if (self.last_action == ClusterType.SPOT and 
            self.consecutive_on_demand == 0 and
            self.spot_available_ratio > 0.3):
            return True
            
        return True
    
    def _should_use_on_demand(self, has_spot, criticality):
        if criticality > self.switch_to_od_threshold:
            return True
            
        if criticality > self.emergency_threshold and not has_spot:
            return True
            
        if self.restart_timer > 0 and criticality > 0.3:
            return True
            
        if self.conservative_mode and criticality > 0.6:
            return True
            
        if (self.spot_unavailable_streak > self.max_spot_unavailable_streak * 0.7 and 
            criticality > 0.2):
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state(last_cluster_type, has_spot)
        
        # If work is done, do nothing
        if self.remaining_work <= 0:
            self.last_action = ClusterType.NONE
            return ClusterType.NONE
            
        criticality = self._calculate_criticality()
        
        # Emergency: must use on-demand to meet deadline
        if criticality > 1.8:
            self.last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
            
        # If we're in restart, wait it out unless critical
        if self.restart_timer > 0:
            if criticality > 0.5:
                self.last_action = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND
            self.last_action = ClusterType.NONE
            return ClusterType.NONE
        
        # Decide on action
        if self._should_use_on_demand(has_spot, criticality):
            self.last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
            
        if self._should_use_spot(has_spot, criticality):
            if has_spot:
                self.last_action = ClusterType.SPOT
                return ClusterType.SPOT
            else:
                # Spot not available but we wanted it - wait or use on-demand
                if criticality > 0.3:
                    self.last_action = ClusterType.ON_DEMAND
                    return ClusterType.ON_DEMAND
                else:
                    self.last_action = ClusterType.NONE
                    return ClusterType.NONE
        else:
            # Wait for better conditions
            self.last_action = ClusterType.NONE
            return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)