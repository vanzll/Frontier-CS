import heapq
from collections import deque
from enum import Enum
import math

class Solution:
    NAME = "my_solution"
    
    def __init__(self, args):
        self.args = args
        self.env = None
        self.task_duration = None
        self.deadline = None
        self.restart_overhead = None
        self.task_done_time = None
        
        # Tuning parameters
        self.urgency_threshold = 0.15
        self.spot_usage_aggression = 0.7
        self.min_spot_uptime = 1800
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _setup(self):
        if self.env is None:
            return
            
        self.gap_seconds = self.env.gap_seconds
        self.on_demand_cost = 3.06 / 3600 * self.gap_seconds
        self.spot_cost = 0.97 / 3600 * self.gap_seconds
        
        # Time management
        self.remaining_work = self.task_duration
        self.time_elapsed = 0
        self.spot_history = deque(maxlen=100)
        
        # State tracking
        self.current_cluster = None
        self.restart_timer = 0
        self.consecutive_spot_failures = 0
        self.spot_uptime_counter = 0
        
    def _step(self, last_cluster_type, has_spot) -> "ClusterType":
        if self.env is None:
            self.env = type('Env', (), {
                'elapsed_seconds': 0,
                'gap_seconds': 1,
                'cluster_type': None
            })()
            
        self.env.elapsed_seconds += self.gap_seconds
        self.env.cluster_type = last_cluster_type
        
        if self.task_duration is None:
            return ClusterType.ON_DEMAND
            
        self._setup()
        
        # Update state
        self.time_elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        self.remaining_work = self.task_duration - work_done
        
        # Update restart timer
        if self.restart_timer > 0:
            self.restart_timer = max(0, self.restart_timer - self.gap_seconds)
        
        # Calculate urgency
        time_remaining = self.deadline - self.time_elapsed
        urgency = 1.0 - (time_remaining / (self.deadline))
        
        # Safety check - if we're going to miss deadline, go all-in on on-demand
        time_needed = self.remaining_work
        if time_needed > time_remaining - self.restart_overhead:
            return ClusterType.ON_DEMAND
        
        # If we have restart overhead pending, we need to wait
        if self.restart_timer > 0:
            if time_remaining - self.restart_timer < time_needed:
                # If restart would make us miss deadline, use on-demand
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        # Calculate risk-adjusted spot preference
        spot_preference = self._calculate_spot_preference(urgency, has_spot)
        
        # Make decision
        if has_spot and spot_preference > 0.5:
            # Use spot if we expect reasonable uptime
            expected_uptime = self._estimate_spot_uptime()
            if expected_uptime >= self.min_spot_uptime or urgency < self.urgency_threshold:
                if last_cluster_type != ClusterType.SPOT:
                    # Starting new spot instance - set restart timer
                    self.restart_timer = self.restart_overhead
                    return ClusterType.NONE
                return ClusterType.SPOT
        
        # Use on-demand if urgent or spot not available
        if urgency > self.urgency_threshold or not has_spot:
            return ClusterType.ON_DEMAND
        
        # Otherwise wait for better conditions
        return ClusterType.NONE
    
    def _calculate_spot_preference(self, urgency, has_spot):
        # Base preference based on spot availability
        base_preference = 1.0 if has_spot else 0.0
        
        # Adjust based on urgency (lower urgency = higher spot preference)
        urgency_factor = 1.0 - min(urgency / self.urgency_threshold, 1.0)
        
        # Adjust based on recent spot history
        history_factor = 1.0
        if self.spot_history:
            success_rate = sum(self.spot_history) / len(self.spot_history)
            history_factor = 0.3 + 0.7 * success_rate
        
        # Combine factors
        preference = base_preference * urgency_factor * history_factor * self.spot_usage_aggression
        
        # Reduce preference if we've had consecutive failures
        if self.consecutive_spot_failures > 2:
            preference *= 0.5
        
        return preference
    
    def _estimate_spot_uptime(self):
        # Simple estimation based on recent history
        if not self.spot_history:
            return 3600  # Default 1 hour
        
        success_rate = sum(self.spot_history) / len(self.spot_history)
        
        # Map success rate to expected uptime (in seconds)
        # Higher success rate = longer expected uptime
        if success_rate > 0.8:
            return 7200  # 2 hours
        elif success_rate > 0.6:
            return 3600  # 1 hour
        elif success_rate > 0.4:
            return 1800  # 30 minutes
        else:
            return 900   # 15 minutes
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

class ClusterType(Enum):
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"
    NONE = "NONE"