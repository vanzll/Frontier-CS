import heapq
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class SpotState(Enum):
    AVAILABLE = 1
    UNAVAILABLE = 2
    PENDING = 3

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.initialized = False
        self.strategy_data = None
        
    def solve(self, spec_path: str) -> "Solution":
        """Optional initialization. Called once before evaluation."""
        try:
            with open(spec_path, 'r') as f:
                config = json.load(f)
                self.task_duration_sec = config.get("task_duration_seconds", 172800)  # 48h
                self.deadline_sec = config.get("deadline_seconds", 252000)  # 70h
                self.restart_overhead_sec = config.get("restart_overhead_seconds", 720)  # 0.2h
                self.spot_price = config.get("spot_price", 0.97)
                self.ondemand_price = config.get("ondemand_price", 3.06)
                
                # Calculate break-even probabilities
                self.initialize_strategy_params()
        except:
            # Use defaults if spec file not found
            self.task_duration_sec = 172800
            self.deadline_sec = 252000
            self.restart_overhead_sec = 720
            self.spot_price = 0.97
            self.ondemand_price = 3.06
            self.initialize_strategy_params()
            
        self.initialized = True
        return self
    
    def initialize_strategy_params(self):
        """Initialize strategy parameters based on known constraints."""
        # Pre-calculate deadlines and thresholds
        self.total_slack = self.deadline_sec - self.task_duration_sec  # 22h slack
        
        # Conservative safety margin (in seconds)
        self.safety_margin = min(3600, self.total_slack * 0.2)  # 1h or 20% of slack
        
        # When to switch to on-demand (in remaining seconds)
        # Calculate based on restart overhead and slack
        self.emergency_threshold = self.restart_overhead_sec * 3
        
        # Minimum spot run duration to justify restart overhead
        self.min_spot_duration = max(self.restart_overhead_sec * 2, 1800)  # 2x overhead or 30min
        
        # Track state
        self.consecutive_spots = 0
        self.last_switch_time = 0
        self.spot_available_history = []
        self.expected_completion = 0
        self.in_overhead = False
        self.overhead_remaining = 0
        
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        if not self.initialized:
            self.initialize_strategy_params()
        
        current_time = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        # Update spot history (window of last 100 steps)
        self.spot_available_history.append(1 if has_spot else 0)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)
        
        # Calculate work done and remaining
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # Calculate time remaining until deadline
        time_remaining = self.deadline - current_time
        
        # Update consecutive spot counter
        if last_cluster_type == ClusterType.SPOT:
            self.consecutive_spots += 1
        else:
            self.consecutive_spots = 0
        
        # Handle restart overhead state
        if self.in_overhead:
            self.overhead_remaining -= gap
            if self.overhead_remaining <= 0:
                self.in_overhead = False
                self.overhead_remaining = 0
        
        # EMERGENCY: If we're running out of time, use on-demand
        if work_remaining > 0 and time_remaining <= work_remaining + self.emergency_threshold:
            # Only switch if not already on on-demand to avoid unnecessary overhead
            if last_cluster_type != ClusterType.ON_DEMAND and not self.in_overhead:
                # Check if we should incur overhead for switching
                if work_remaining > self.restart_overhead_sec:
                    self.in_overhead = True
                    self.overhead_remaining = self.restart_overhead_sec
                return ClusterType.ON_DEMAND
            elif last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                # If in overhead and not on-demand, wait it out
                return ClusterType.NONE
        
        # If we're in overhead period, do nothing
        if self.in_overhead:
            return ClusterType.NONE
        
        # Calculate if we have enough time to use spot
        spot_time_needed = work_remaining + self.restart_overhead_sec
        can_use_spot = time_remaining >= spot_time_needed + self.safety_margin
        
        # Decision logic
        if has_spot and can_use_spot:
            # Use spot if available and we have time buffer
            
            # But check if we should pause to avoid short runs
            if last_cluster_type != ClusterType.SPOT:
                # Just switched to spot, will need to complete minimum duration
                estimated_spot_time = min(work_remaining, time_remaining - self.safety_margin)
                if estimated_spot_time >= self.min_spot_duration:
                    return ClusterType.SPOT
                else:
                    # Too short to justify restart overhead
                    return ClusterType.NONE
            else:
                # Already on spot, continue if we have reasonable work remaining
                return ClusterType.SPOT
                
        elif not has_spot and can_use_spot:
            # Spot not available but we have time
            
            # Check if we should wait for spot or use on-demand
            # Use recent history to estimate spot availability
            if len(self.spot_available_history) >= 10:
                spot_availability = sum(self.spot_available_history[-10:]) / 10.0
            else:
                spot_availability = 0.5
                
            # If spot is frequently available, wait
            if spot_availability > 0.6 and time_remaining >= work_remaining + 3600:  # 1h buffer
                return ClusterType.NONE
            else:
                # Use on-demand if spot is unreliable
                if last_cluster_type != ClusterType.ON_DEMAND and work_remaining > self.restart_overhead_sec:
                    self.in_overhead = True
                    self.overhead_remaining = self.restart_overhead_sec
                return ClusterType.ON_DEMAND
                
        else:
            # Not enough time buffer for spot, use on-demand
            if last_cluster_type != ClusterType.ON_DEMAND and work_remaining > self.restart_overhead_sec:
                self.in_overhead = True
                self.overhead_remaining = self.restart_overhead_sec
                return ClusterType.ON_DEMAND
            elif last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        """For evaluator instantiation."""
        args, _ = parser.parse_known_args()
        return cls(args)