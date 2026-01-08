import math
import sys
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cbl_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.initialized = False
        self.spot_history = []
        self.consecutive_spot_unavailable = 0
        self.consecutive_spot_available = 0
        self.switching_counter = 0
        self.last_action = None
        self.total_progress = 0.0
        self.remaining_work = 0.0
        self.time_remaining = 0.0
        self.safety_factor = 1.1
        self.emergency_threshold = 0.2
        self.min_spot_streak = 3
        self.od_switch_threshold = 0.5
    
    def solve(self, spec_path: str) -> "Solution":
        # Read spec if needed, but no config file required
        return self
    
    def _initialize_state(self):
        if not self.initialized:
            self.remaining_work = self.task_duration
            self.initialized = True
    
    def _update_state(self, last_cluster_type, has_spot):
        # Update time remaining
        self.time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Update remaining work based on completed segments
        if hasattr(self, 'task_done_time') and self.task_done_time:
            self.total_progress = sum(self.task_done_time)
            self.remaining_work = max(0.0, self.task_duration - self.total_progress)
        
        # Track spot availability patterns
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 100:
            self.spot_history.pop(0)
        
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0
    
    def _calculate_conservative_progress_rate(self):
        # Calculate required progress rate to finish before deadline
        if self.time_remaining <= 0:
            return float('inf')
        
        required_rate = self.remaining_work / self.time_remaining
        return required_rate * self.safety_factor
    
    def _should_switch_to_od(self, required_rate, has_spot):
        # Emergency: if we're running out of time
        if self.time_remaining < self.remaining_work + self.restart_overhead:
            return True
        
        # Emergency: if progress rate too low
        if required_rate > 1.0:
            return True
        
        # If spot unavailable for too long
        if self.consecutive_spot_unavailable > 5:
            return True
        
        # If we need to make progress but spot is unreliable
        spot_availability = sum(self.spot_history[-20:]) / min(20, len(self.spot_history)) if self.spot_history else 0
        if required_rate > self.od_switch_threshold and spot_availability < 0.3:
            return True
        
        return False
    
    def _should_use_spot(self, required_rate, has_spot):
        if not has_spot:
            return False
        
        # Don't switch too frequently
        if self.switching_counter > 0:
            return False
        
        # Use spot if we have enough time buffer
        time_buffer = self.time_remaining - self.remaining_work
        if time_buffer > 2 * self.restart_overhead:
            return True
        
        # Use spot if we've seen consistent availability
        if self.consecutive_spot_available >= self.min_spot_streak:
            return True
        
        # Use spot if progress requirements are low
        if required_rate < 0.3:
            return True
        
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_state()
        self._update_state(last_cluster_type, has_spot)
        
        # Calculate conservative progress rate
        required_rate = self._calculate_conservative_progress_rate()
        
        # Update switching counter
        if self.last_action is not None and self.last_action != last_cluster_type:
            self.switching_counter = max(0, self.switching_counter - 1)
        elif self.switching_counter > 0:
            self.switching_counter -= 1
        
        # Emergency: must use on-demand
        if self._should_switch_to_od(required_rate, has_spot):
            self.last_action = ClusterType.ON_DEMAND
            if last_cluster_type != ClusterType.ON_DEMAND:
                self.switching_counter = int(self.restart_overhead / self.env.gap_seconds)
            return ClusterType.ON_DEMAND
        
        # Good conditions for spot
        if self._should_use_spot(required_rate, has_spot):
            self.last_action = ClusterType.SPOT
            if last_cluster_type != ClusterType.SPOT:
                self.switching_counter = int(self.restart_overhead / self.env.gap_seconds)
            return ClusterType.SPOT
        
        # Default to on-demand if spot not available
        if not has_spot:
            self.last_action = ClusterType.ON_DEMAND
            if last_cluster_type != ClusterType.ON_DEMAND:
                self.switching_counter = int(self.restart_overhead / self.env.gap_seconds)
            return ClusterType.ON_DEMAND
        
        # Pause if we're ahead of schedule
        time_buffer = self.time_remaining - self.remaining_work
        if time_buffer > 3 * self.restart_overhead:
            self.last_action = ClusterType.NONE
            return ClusterType.NONE
        
        # Conservative: use on-demand by default
        self.last_action = ClusterType.ON_DEMAND
        return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)