import json
from argparse import Namespace
from enum import Enum
from typing import List, Tuple
import heapq
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class RegionState:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.spot_availability = []  # List of (time, available) tuples
        self.spot_windows = []  # List of (start, end) for continuous spot availability
        self.future_spot_times = []  # List of future spot availability times
        self.spot_density = 0.0  # Percentage of time spot is available


class Solution(MultiRegionStrategy):
    """Efficient multi-region scheduling strategy with deadline awareness."""
    
    NAME = "deadline_aware_spot_optimizer"
    
    def solve(self, spec_path: str) -> "Solution":
        """Initialize solution from spec file."""
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
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        
        # State tracking
        self.region_data = []
        self.current_plan = []
        self.plan_index = 0
        self.last_decision_time = 0
        self.emergency_mode = False
        self.consecutive_spot_failures = 0
        self.best_region_cache = None
        self.region_attempts = {}
        
        return self
    
    def _load_region_data(self):
        """Load and preprocess region spot availability data."""
        if hasattr(self.env, 'trace_data') and self.env.trace_data:
            for region_idx in range(self.env.get_num_regions()):
                # Simulate loading trace data - in actual implementation
                # this would parse the trace files
                state = RegionState(region_idx)
                self.region_data.append(state)
    
    def _calculate_time_metrics(self) -> Tuple[float, float, float]:
        """Calculate critical time metrics for decision making."""
        elapsed = self.env.elapsed_seconds
        total_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = self.task_duration - total_done
        time_left = self.deadline - elapsed
        
        # Account for potential restart overhead
        effective_time_left = time_left
        if self.remaining_restart_overhead > 0:
            effective_time_left -= self.remaining_restart_overhead
            
        # Calculate safe thresholds
        time_per_work_unit = self.env.gap_seconds
        min_steps_needed = math.ceil(remaining_work / time_per_work_unit)
        min_time_needed = min_steps_needed * time_per_work_unit
        
        # Critical threshold: when we must use on-demand to guarantee completion
        critical_threshold = min_time_needed + self.restart_overhead * 2
        
        return remaining_work, time_left, effective_time_left, critical_threshold
    
    def _find_best_region(self, has_spot_current: bool) -> int:
        """Find the region with best spot availability prospects."""
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # If current region has spot, stay put
        if has_spot_current:
            return current_region
            
        # Simple round-robin exploration
        # Track attempts in each region
        if not hasattr(self, 'region_attempts'):
            self.region_attempts = {}
            
        if current_region not in self.region_attempts:
            self.region_attempts[current_region] = 0
            
        # If we've tried current region too many times without spot, switch
        if self.region_attempts[current_region] >= 3:
            next_region = (current_region + 1) % num_regions
            self.region_attempts[current_region] = 0
            return next_region
            
        return current_region
    
    def _should_use_emergency_mode(self, time_left: float, 
                                   remaining_work: float, 
                                   critical_threshold: float) -> bool:
        """Determine if we need to switch to emergency on-demand mode."""
        # Calculate if we can still finish with spot considering overheads
        max_spot_failures = 3
        spot_time_needed = remaining_work + (self.restart_overhead * max_spot_failures)
        
        # Emergency conditions
        condition1 = time_left < critical_threshold * 1.2
        condition2 = time_left < spot_time_needed
        condition3 = self.consecutive_spot_failures >= 2
        
        return condition1 or condition2 or condition3
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        Uses deadline-aware spot optimization with emergency fallback.
        """
        # Calculate critical metrics
        remaining_work, time_left, effective_time_left, critical_threshold = self._calculate_time_metrics()
        
        # Check if task is complete
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Update region attempts tracking
        current_region = self.env.get_current_region()
        if has_spot:
            if current_region in self.region_attempts:
                self.region_attempts[current_region] = 0
            self.consecutive_spot_failures = 0
        else:
            if current_region in self.region_attempts:
                self.region_attempts[current_region] += 1
            else:
                self.region_attempts[current_region] = 1
            self.consecutive_spot_failures += 1
        
        # Check emergency mode
        self.emergency_mode = self._should_use_emergency_mode(
            time_left, remaining_work, critical_threshold
        )
        
        # EMERGENCY MODE: Use on-demand to guarantee completion
        if self.emergency_mode:
            # Stay in current region to avoid overhead
            return ClusterType.ON_DEMAND
        
        # NORMAL MODE: Optimize for cost while maintaining deadline safety
        
        # Find best region
        best_region = self._find_best_region(has_spot)
        if best_region != current_region:
            self.env.switch_region(best_region)
            # After switching, we need to check spot availability for new region
            # Since we don't have that info, we'll be conservative
            return ClusterType.NONE
        
        # Decision logic based on spot availability and time pressure
        if has_spot:
            # Use spot if available and we have time buffer
            time_buffer_needed = self.restart_overhead * 2
            if time_left > remaining_work + time_buffer_needed:
                return ClusterType.SPOT
            else:
                # Not enough buffer, use on-demand for reliability
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            # If we have plenty of time, wait for spot
            generous_buffer = self.restart_overhead * 4
            if time_left > remaining_work + generous_buffer:
                return ClusterType.NONE
            else:
                # Time is getting tight, use on-demand
                return ClusterType.ON_DEMAND
    
    def _estimate_completion_with_spot(self) -> float:
        """Estimate completion time using spot instances with failures."""
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # Assume 80% spot availability (conservative estimate)
        spot_availability = 0.8
        avg_spot_run_before_failure = 4.0  # hours
        failures_expected = remaining_work / (avg_spot_run_before_failure * 3600)
        
        total_overhead = failures_expected * self.restart_overhead
        total_time = remaining_work + total_overhead
        
        return total_time