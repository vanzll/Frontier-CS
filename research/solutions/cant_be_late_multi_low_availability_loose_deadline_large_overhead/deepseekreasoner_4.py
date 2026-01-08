import json
from argparse import Namespace
from enum import Enum
from typing import List, Tuple, Optional
import heapq
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class State(Enum):
    SPOT = 1
    ON_DEMAND = 2
    OVERHEAD = 3
    NONE = 4


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""
    
    NAME = "my_strategy"
    
    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            config = json.load(f)
        
        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Initialize strategy parameters
        self.spot_price = 0.9701  # $/hr
        self.on_demand_price = 3.06  # $/hr
        self.time_step = self.env.gap_seconds if hasattr(self.env, 'gap_seconds') else 3600.0
        
        # State tracking
        self.current_region = 0
        self.region_history = []
        self.spot_availability_history = []
        self.consecutive_no_spot = 0
        self.max_consecutive_no_spot_threshold = 3
        
        # Cost tracking
        self.total_cost = 0.0
        self.work_remaining = self.task_duration
        self.deadline_remaining = self.deadline
        
        return self
    
    def _get_work_remaining(self) -> float:
        """Get remaining work in seconds."""
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        return max(0.0, self.task_duration - work_done)
    
    def _get_time_remaining(self) -> float:
        """Get remaining time until deadline in seconds."""
        return max(0.0, self.deadline - self.env.elapsed_seconds)
    
    def _get_criticality(self) -> float:
        """Calculate how critical the situation is (0=not critical, 1=very critical)."""
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        
        if time_remaining <= 0:
            return 1.0
        
        # Base work rate needed
        needed_rate = work_remaining / time_remaining if time_remaining > 0 else float('inf')
        
        # Normalize to [0, 1] range
        # If needed_rate > 1, we're behind schedule
        return min(1.0, max(0.0, (needed_rate - 0.5) / 0.5))
    
    def _should_switch_region(self, has_spot: bool) -> bool:
        """Determine if we should switch regions."""
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        
        # If we're very close to deadline, don't switch
        if time_remaining < 2 * self.time_step:
            return False
        
        # If spot is available, don't switch
        if has_spot:
            self.consecutive_no_spot = 0
            return False
        
        # Track consecutive steps without spot
        self.consecutive_no_spot += 1
        
        # Switch if we've had too many consecutive steps without spot
        if self.consecutive_no_spot >= self.max_consecutive_no_spot_threshold:
            return True
        
        # Switch if we have plenty of time and work remaining is high
        if time_remaining > work_remaining * 1.5 and work_remaining > 4 * self.time_step:
            return True
            
        return False
    
    def _choose_best_region(self, current_region: int) -> int:
        """Choose the best region to switch to."""
        num_regions = self.env.get_num_regions()
        
        # Simple round-robin for now
        # In a more sophisticated implementation, we could track region performance
        return (current_region + 1) % num_regions
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Update current region
        self.current_region = self.env.get_current_region()
        
        # Get current state
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        criticality = self._get_criticality()
        
        # Check if we need to be aggressive (close to deadline)
        need_aggressive = criticality > 0.7 or time_remaining < work_remaining * 1.2
        
        # Check if we should switch region
        if self._should_switch_region(has_spot):
            new_region = self._choose_best_region(self.current_region)
            if new_region != self.current_region:
                self.env.switch_region(new_region)
                self.current_region = new_region
                self.consecutive_no_spot = 0
                # After switching, we might want to pause to assess the new region
                return ClusterType.NONE
        
        # Decision logic
        if need_aggressive:
            # When close to deadline, use on-demand to ensure completion
            return ClusterType.ON_DEMAND
        elif has_spot:
            # Use spot when available and we're not in critical state
            return ClusterType.SPOT
        elif work_remaining > time_remaining:
            # If we're falling behind, use on-demand
            return ClusterType.ON_DEMAND
        elif time_remaining > work_remaining * 1.5:
            # If we have plenty of time, wait for spot to save cost
            return ClusterType.NONE
        else:
            # Default to on-demand
            return ClusterType.ON_DEMAND