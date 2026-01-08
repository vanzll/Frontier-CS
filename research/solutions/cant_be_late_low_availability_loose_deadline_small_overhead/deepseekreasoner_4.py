import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import math

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.initialized = False
        self.remaining_work = 0
        self.time_left = 0
        self.last_restart_time = -float('inf')
        self.in_restart = False
        self.restart_end_time = 0
        self.spot_usage_history = []
        self.consecutive_spots = 0
        self.safety_factor = 1.2
        self.aggressive_mode = False

    def solve(self, spec_path: str) -> "Solution":
        self.initialized = True
        return self

    def _update_state(self, last_cluster_type: ClusterType):
        current_time = self.env.elapsed_seconds
        
        # Update remaining work
        if last_cluster_type == ClusterType.ON_DEMAND:
            if not self.in_restart:
                self.remaining_work -= self.env.gap_seconds
        elif last_cluster_type == ClusterType.SPOT:
            if not self.in_restart:
                self.remaining_work -= self.env.gap_seconds
                self.consecutive_spots += 1
            else:
                if current_time >= self.restart_end_time:
                    self.in_restart = False
                    # After restart ends, work for the remaining time in this gap
                    remaining_gap = current_time + self.env.gap_seconds - self.restart_end_time
                    if remaining_gap > 0:
                        self.remaining_work -= remaining_gap
                        self.consecutive_spots += 1
        else:
            self.consecutive_spots = 0
            
        # Check if we entered restart in last step
        if (last_cluster_type == ClusterType.SPOT and 
            self.env.elapsed_seconds - self.last_restart_time < self.restart_overhead):
            if not self.in_restart:
                self.in_restart = True
                self.restart_end_time = self.last_restart_time + self.restart_overhead
        
        self.time_left = self.deadline - current_time
        
        # Switch to aggressive mode if we're running out of time
        time_needed = self.remaining_work * self.safety_factor
        if self.in_restart:
            time_needed += self.restart_overhead
        
        self.aggressive_mode = self.time_left < time_needed * 1.5

    def _should_use_spot(self, has_spot: bool) -> bool:
        if not has_spot:
            return False
            
        current_time = self.env.elapsed_seconds
        
        # If we're in restart period, continue with spot if available
        if self.in_restart:
            return True
            
        # Calculate time needed with on-demand
        time_needed_od = self.remaining_work
        
        # Estimate time needed with spot (accounting for potential restarts)
        # Use historical spot availability to estimate success probability
        avg_spot_duration = 3600  # 1 hour default
        if len(self.spot_usage_history) > 10:
            successful_runs = [dur for dur in self.spot_usage_history if dur > 300]
            if successful_runs:
                avg_spot_duration = sum(successful_runs) / len(successful_runs)
        
        # Conservative estimate: assume we'll get interrupted once
        estimated_restarts = max(1, self.remaining_work / avg_spot_duration)
        time_needed_spot = self.remaining_work + estimated_restarts * self.restart_overhead
        
        # Add safety margin
        time_needed_spot *= self.safety_factor
        
        # Check if we have enough time for spot strategy
        if self.time_left > time_needed_spot and not self.aggressive_mode:
            return True
        elif self.consecutive_spots > 10 and self.time_left > time_needed_spot * 0.8:
            # If we've had good spot run, be more aggressive
            return True
        elif self.time_left > time_needed_od * 2 and not self.aggressive_mode:
            # Plenty of time, try spot
            return True
            
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized:
            self.remaining_work = self.task_duration
            self.time_left = self.deadline
        
        # Update internal state
        self._update_state(last_cluster_type)
        
        # If work is done, return NONE
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        current_time = self.env.elapsed_seconds
        
        # Emergency mode: if we're really behind, use on-demand
        if self.time_left < self.remaining_work * 1.1:
            return ClusterType.ON_DEMAND
        
        # Check if we should start a new spot instance
        if last_cluster_type != ClusterType.SPOT and self._should_use_spot(has_spot):
            self.last_restart_time = current_time
            self.spot_usage_history.append(0)
            return ClusterType.SPOT
        
        # If we're currently on spot
        if last_cluster_type == ClusterType.SPOT:
            if has_spot:
                # Continue with spot if available and not in aggressive mode
                if not self.aggressive_mode:
                    return ClusterType.SPOT
                # In aggressive mode, only continue if we've had good run
                elif self.consecutive_spots > 5:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # Spot unavailable, pause briefly to see if it comes back
                if self.time_left > self.remaining_work * 1.5 and not self.aggressive_mode:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND
        
        # If we're on on-demand, check if we should switch to spot
        if last_cluster_type == ClusterType.ON_DEMAND:
            if self._should_use_spot(has_spot):
                self.last_restart_time = current_time
                self.spot_usage_history.append(0)
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        # If we're paused (NONE), decide what to do next
        if last_cluster_type == ClusterType.NONE:
            if self._should_use_spot(has_spot):
                self.last_restart_time = current_time
                self.spot_usage_history.append(0)
                return ClusterType.SPOT
            elif self.time_left < self.remaining_work * 1.2:
                return ClusterType.ON_DEMAND
            else:
                # Wait a bit more for spot
                return ClusterType.NONE
        
        # Default fallback
        if has_spot and self.time_left > self.remaining_work * 1.5:
            self.last_restart_time = current_time
            self.spot_usage_history.append(0)
            return ClusterType.SPOT
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)