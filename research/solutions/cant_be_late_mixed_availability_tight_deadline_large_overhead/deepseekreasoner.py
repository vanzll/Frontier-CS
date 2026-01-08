import argparse
from typing import List, Tuple
import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.spot_availability = []
        self.spot_price = 0.97 / 3600  # $/second
        self.ondemand_price = 3.06 / 3600  # $/second
        self.time_step = 1.0  # seconds, will be updated
        self.restart_overhead_steps = 0
        self.consecutive_spot_failures = 0
        self.consecutive_spot_successes = 0
        self.pessimistic_factor = 1.2
        self.safety_margin = 0.1
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
    
    def _estimate_completion_time(self, remaining_work: float, use_ondemand: bool) -> float:
        """Estimate time to complete remaining work"""
        if use_ondemand:
            return remaining_work
        else:
            # Account for potential spot failures and restart overheads
            base_time = remaining_work
            # Add pessimistic estimate for restart overheads
            estimated_restarts = max(0, remaining_work / (self.time_step * 10) - 1)
            overhead_time = estimated_restarts * self.restart_overhead
            return base_time + overhead_time * self.pessimistic_factor
    
    def _should_use_ondemand(self, last_cluster_type: ClusterType, 
                           has_spot: bool, remaining_work: float) -> bool:
        """Determine if we should use on-demand based on current situation"""
        current_time = self.env.elapsed_seconds
        deadline = self.deadline
        time_left = deadline - current_time
        
        # If very little time left, use on-demand
        if time_left < remaining_work * (1 + self.safety_margin):
            return True
        
        # If we've had many consecutive spot failures, switch to on-demand
        if self.consecutive_spot_failures >= 3:
            return True
        
        # If spot is not available and we're behind schedule
        if not has_spot and remaining_work > time_left * 0.8:
            return True
        
        # If we're in restart overhead, consider on-demand
        if self.restart_overhead_steps > 0:
            # Only use on-demand if we're really behind
            if remaining_work > time_left * 0.9:
                return True
        
        # Conservative check: if spot completion estimate exceeds deadline
        spot_estimate = self._estimate_completion_time(remaining_work, use_ondemand=False)
        if spot_estimate > time_left:
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update time step from environment
        self.time_step = self.env.gap_seconds
        
        # Calculate remaining work
        completed_work = sum(segment for segment in self.task_done_time)
        remaining_work = self.task_duration - completed_work
        
        # Update restart overhead counter
        if self.restart_overhead_steps > 0:
            self.restart_overhead_steps -= 1
        
        # Update spot success/failure counters
        if last_cluster_type == ClusterType.SPOT:
            if has_spot:
                self.consecutive_spot_successes += 1
                self.consecutive_spot_failures = 0
            else:
                self.consecutive_spot_failures += 1
                self.consecutive_spot_successes = 0
                # If spot failed while we were using it, we need restart overhead
                if self.env.cluster_type == ClusterType.SPOT:
                    self.restart_overhead_steps = int(self.restart_overhead / self.time_step)
        else:
            self.consecutive_spot_failures = 0
            self.consecutive_spot_successes = 0
        
        # Check if we should use on-demand
        if self._should_use_ondemand(last_cluster_type, has_spot, remaining_work):
            if has_spot and remaining_work > 0:
                # If we're switching from non-spot to spot, account for restart overhead
                if (last_cluster_type != ClusterType.SPOT and 
                    self.env.cluster_type != ClusterType.SPOT):
                    self.restart_overhead_steps = int(self.restart_overhead / self.time_step)
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Default: use spot if available, otherwise on-demand
        if has_spot and remaining_work > 0:
            # If we're switching from non-spot to spot, account for restart overhead
            if (last_cluster_type != ClusterType.SPOT and 
                self.env.cluster_type != ClusterType.SPOT):
                self.restart_overhead_steps = int(self.restart_overhead / self.time_step)
            return ClusterType.SPOT
        
        # If no work left, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE
            
        return ClusterType.ON_DEMAND