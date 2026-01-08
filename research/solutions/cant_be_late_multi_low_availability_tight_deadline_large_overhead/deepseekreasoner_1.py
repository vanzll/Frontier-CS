import json
from argparse import Namespace
from enum import Enum
import heapq
from typing import List, Tuple, Optional
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""
    
    NAME = "my_strategy"
    
    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)
        
        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Pre-calculate prices and constants
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.gap_seconds = self.env.gap_seconds
        
        # State for dynamic programming
        self.regions = self.env.get_num_regions()
        self.spot_availability = [0] * self.regions
        self.last_decision = ClusterType.NONE
        self.consecutive_spots = 0
        self.emergency_mode = False
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        self.spot_availability[current_region] = 1 if has_spot else 0
        
        elapsed = self.env.elapsed_seconds
        remaining_work = self.task_duration[0] - sum(self.task_done_time)
        time_left = self.deadline - elapsed
        
        # Calculate critical threshold
        if remaining_work <= 0:
            return ClusterType.NONE
            
        # If we're in emergency mode (close to deadline), use on-demand
        time_needed_for_ondemand = remaining_work
        if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead[0] <= 0:
            time_needed_for_ondemand += self.restart_overhead[0]
        
        safety_margin = max(3600, 2 * self.restart_overhead[0])  # 1 hour or 2x overhead
        if time_left - time_needed_for_ondemand < safety_margin:
            self.emergency_mode = True
        
        # Emergency mode: use on-demand if available
        if self.emergency_mode:
            return ClusterType.ON_DEMAND
        
        # Calculate efficiency metrics
        spot_efficiency = self._calculate_spot_efficiency(current_region)
        
        # Decision logic
        if has_spot:
            # Use spot if it's reasonably efficient
            if spot_efficiency > 0.7 and self.consecutive_spots < 10:
                self.consecutive_spots += 1
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
            elif spot_efficiency > 0.5 and time_left > remaining_work * 1.5:
                self.consecutive_spots += 1
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
        
        # If no spot available or inefficient, check other regions
        if not has_spot and self.remaining_restart_overhead[0] <= 0:
            best_region = self._find_best_region()
            if best_region != current_region:
                self.env.switch_region(best_region)
                # After switching, we'll have restart overhead, so return NONE
                self.consecutive_spots = 0
                self.last_decision = ClusterType.NONE
                return ClusterType.NONE
        
        # If we get here, use on-demand as fallback
        self.consecutive_spots = 0
        self.last_decision = ClusterType.ON_DEMAND
        return ClusterType.ON_DEMAND
    
    def _calculate_spot_efficiency(self, region: int) -> float:
        """Calculate the efficiency of using spot instances in this region."""
        if self.spot_availability[region] == 0:
            return 0.0
        
        # Simple efficiency metric based on time left vs work needed
        elapsed = self.env.elapsed_seconds
        remaining_work = self.task_duration[0] - sum(self.task_done_time)
        time_left = self.deadline - elapsed
        
        if time_left <= 0 or remaining_work <= 0:
            return 0.0
        
        # Efficiency is higher when we have more time buffer
        time_buffer = time_left / remaining_work
        efficiency = min(1.0, time_buffer / 2.0)  # Normalize
        
        return efficiency
    
    def _find_best_region(self) -> int:
        """Find the region with highest expected spot availability."""
        current_region = self.env.get_current_region()
        best_region = current_region
        best_score = self.spot_availability[current_region]
        
        # Try to find a region with spot available
        for region in range(self.regions):
            if region == current_region:
                continue
            if self.spot_availability[region] > best_score:
                best_score = self.spot_availability[region]
                best_region = region
        
        return best_region