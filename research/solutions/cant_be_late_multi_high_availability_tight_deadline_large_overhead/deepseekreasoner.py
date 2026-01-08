import json
from argparse import Namespace
import math
from typing import List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
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
        
        # Initialize tracking variables
        self.region_history = []
        self.spot_availability_history = []
        self.region_performance = {}
        self.last_region_switch = 0
        self.consecutive_failures = 0
        self.aggressiveness = 1.0
        
        return self
    
    def _get_remaining_work(self) -> float:
        return self.task_duration - sum(self.task_done_time)
    
    def _get_remaining_time(self) -> float:
        return self.deadline - self.env.elapsed_seconds
    
    def _get_criticality(self) -> float:
        """Calculate how critical the situation is (0-1)"""
        remaining_work = self._get_remaining_work()
        remaining_time = self._get_remaining_time()
        
        if remaining_work <= 0:
            return 0.0
        
        # Normal work rate (without overhead)
        normal_rate = self.env.gap_seconds
        
        # Critical if we need more than 80% of remaining time
        required_rate = remaining_work / remaining_time if remaining_time > 0 else float('inf')
        return min(1.0, max(0.0, required_rate / normal_rate - 0.2) * 2)
    
    def _should_use_ondemand(self, has_spot: bool) -> bool:
        """Determine if we should use on-demand based on criticality"""
        criticality = self._get_criticality()
        
        # If we're very critical, use on-demand
        if criticality > 0.8:
            return True
        
        # If we've had consecutive failures, use on-demand
        if self.consecutive_failures >= 3:
            return True
        
        # If we're moderately critical and no spot available
        if criticality > 0.4 and not has_spot:
            return True
        
        return False
    
    def _select_best_region(self, current_region: int) -> int:
        """Select the best region to switch to"""
        num_regions = self.env.get_num_regions()
        
        # Simple round-robin with preference for regions that recently had spot
        candidates = []
        for i in range(num_regions):
            if i == current_region:
                continue
            
            # Prefer regions that we haven't tried recently
            last_used = -float('inf')
            for j, (region, _) in enumerate(self.region_history):
                if region == i:
                    last_used = len(self.region_history) - j
                    break
            
            # Randomize slightly to avoid getting stuck
            candidates.append((last_used, i))
        
        # Sort by least recently used
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else current_region
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update failure tracking
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Get current state
        current_region = self.env.get_current_region()
        remaining_work = self._get_remaining_work()
        remaining_time = self._get_remaining_time()
        
        # Check if we're done
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Check if we'll miss deadline even with on-demand
        min_time_needed = remaining_work + (self.restart_overhead if last_cluster_type == ClusterType.NONE else 0)
        if min_time_needed > remaining_time:
            # Emergency mode - use on-demand regardless of cost
            return ClusterType.ON_DEMAND
        
        # Store history
        self.region_history.append((current_region, has_spot))
        if len(self.region_history) > 100:
            self.region_history.pop(0)
        
        # Determine if we should switch regions
        should_switch = False
        if not has_spot and last_cluster_type != ClusterType.ON_DEMAND:
            # No spot available and we're not on on-demand
            if self.consecutive_failures >= 2:
                should_switch = True
            elif len(self.region_history) >= 5:
                # Check if current region has been consistently bad
                recent_failures = sum(1 for _, spot in self.region_history[-5:] 
                                    if not spot)
                if recent_failures >= 4:
                    should_switch = True
        
        # Switch region if needed
        if should_switch and self.env.get_num_regions() > 1:
            new_region = self._select_best_region(current_region)
            if new_region != current_region:
                self.env.switch_region(new_region)
                self.last_region_switch = self.env.elapsed_seconds
                # After switching, we need to be conservative
                return ClusterType.NONE if remaining_time > remaining_work * 1.5 else ClusterType.ON_DEMAND
        
        # Decision logic
        if self._should_use_ondemand(has_spot):
            return ClusterType.ON_DEMAND
        
        # Use spot if available
        if has_spot:
            return ClusterType.SPOT
        
        # No spot available, pause if we have time
        safety_margin = 2.0 * self.restart_overhead
        if remaining_time > remaining_work + safety_margin:
            return ClusterType.NONE
        
        # Otherwise use on-demand to ensure we meet deadline
        return ClusterType.ON_DEMAND