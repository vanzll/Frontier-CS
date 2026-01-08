import json
from argparse import Namespace
import math
from typing import List
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""
    
    NAME = "my_strategy"
    
    def __init__(self, args):
        super().__init__(args)
        self.region_spot_history = None
        self.consecutive_failures = 0
        self.last_region = None
        self.switch_countdown = 0
        self.candidate_regions = None
        self.current_candidate_index = 0
    
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
        
        self.last_region = 0
        self.consecutive_failures = 0
        self.switch_countdown = 0
        self.region_spot_history = {}
        self.candidate_regions = list(range(self.env.get_num_regions()))
        self.current_candidate_index = 0
        
        return self
    
    def _should_switch_region(self, current_region: int, has_spot: bool) -> bool:
        if self.switch_countdown > 0:
            return False
        
        remaining_work = self.task_duration - sum(self.task_done_time)
        time_left = self.deadline - self.env.elapsed_seconds
        
        remaining_effective_time = time_left - self.remaining_restart_overhead
        required_time = remaining_work
        
        # If we're cutting it close, stay put
        if remaining_effective_time < required_time * 1.2:
            return False
        
        # If current region has no spot and we've been here a while, consider switching
        if not has_spot and self.consecutive_failures >= 3:
            return True
        
        # If we've been in this region for too long without consistent spot
        if self.consecutive_failures >= 5:
            return True
        
        return False
    
    def _select_best_region(self, current_region: int) -> int:
        if not self.candidate_regions:
            self.candidate_regions = list(range(self.env.get_num_regions()))
        
        current_idx = self.current_candidate_index
        
        for i in range(len(self.candidate_regions)):
            idx = (current_idx + i) % len(self.candidate_regions)
            candidate = self.candidate_regions[idx]
            if candidate != current_region:
                self.current_candidate_index = (idx + 1) % len(self.candidate_regions)
                return candidate
        
        return (current_region + 1) % self.env.get_num_regions()
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        
        # Update spot history
        if current_region not in self.region_spot_history:
            self.region_spot_history[current_region] = []
        self.region_spot_history[current_region].append(has_spot)
        if len(self.region_spot_history[current_region]) > 10:
            self.region_spot_history[current_region].pop(0)
        
        # Update consecutive failures counter
        if current_region == self.last_region:
            if not has_spot:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = max(0, self.consecutive_failures - 1)
        else:
            self.last_region = current_region
            self.consecutive_failures = 0 if has_spot else 1
        
        # Update switch cooldown
        if self.switch_countdown > 0:
            self.switch_countdown -= 1
        
        # Calculate remaining work and time
        remaining_work = self.task_duration - sum(self.task_done_time)
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Safety margin in seconds (1 hour)
        safety_margin = 3600.0
        
        # Check if we're in a critical situation
        critical_time = False
        if self.remaining_restart_overhead > 0:
            effective_time_left = time_left - self.remaining_restart_overhead
        else:
            effective_time_left = time_left
        
        # If we need to finish soon, use on-demand
        if remaining_work > 0 and effective_time_left - safety_margin <= remaining_work:
            critical_time = True
        
        # If we have no time left, use on-demand
        if time_left <= 0:
            return ClusterType.ON_DEMAND
        
        # If work is done, return NONE
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Check if we should switch region
        if self._should_switch_region(current_region, has_spot):
            new_region = self._select_best_region(current_region)
            if new_region != current_region:
                self.env.switch_region(new_region)
                self.switch_countdown = 3  # Don't switch again for 3 steps
                self.consecutive_failures = 0
                return ClusterType.NONE
        
        # Decision logic
        if critical_time:
            # In critical time, use on-demand to ensure completion
            return ClusterType.ON_DEMAND
        elif has_spot:
            # Use spot if available
            return ClusterType.SPOT
        else:
            # No spot available, check if we should wait or use on-demand
            if effective_time_left > remaining_work * 1.5:
                # We have plenty of time, wait for spot
                return ClusterType.NONE
            else:
                # Running out of time, use on-demand
                return ClusterType.ON_DEMAND