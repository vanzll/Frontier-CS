import json
from argparse import Namespace
import math
from typing import List, Tuple
from enum import Enum

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class State(Enum):
    INIT = 0
    SPOT_SEARCH = 1
    SPOT_RUNNING = 2
    ON_DEMAND = 3
    FINAL_PUSH = 4


class Solution(MultiRegionStrategy):
    NAME = "adaptive_spot_optimizer"

    def __init__(self, args):
        super().__init__(args)
        self.state = State.INIT
        self.region_spot_history = None
        self.current_spot_streak = 0
        self.best_spot_region = 0
        self.last_work_segment = 0
        self.initial_exploration_done = False
        self.regions_explored = 0
        self.spot_success_count = 0
        self.spot_total_count = 0
        self.switch_cooldown = 0
        self.consecutive_failures = 0
        self.time_since_last_switch = 0

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
        
        # Initialize region history
        num_regions = 9  # Maximum expected regions
        self.region_spot_history = [{"spot_count": 0, "total_count": 0, "reliability": 0.0} 
                                    for _ in range(num_regions)]
        
        return self

    def _update_region_history(self, region_idx: int, has_spot: bool):
        """Update spot availability history for a region."""
        if region_idx < len(self.region_spot_history):
            self.region_spot_history[region_idx]["total_count"] += 1
            if has_spot:
                self.region_spot_history[region_idx]["spot_count"] += 1
            
            total = self.region_spot_history[region_idx]["total_count"]
            if total > 0:
                self.region_spot_history[region_idx]["reliability"] = (
                    self.region_spot_history[region_idx]["spot_count"] / total
                )

    def _get_best_spot_region(self, current_region: int) -> int:
        """Find region with highest spot reliability, excluding current if recently switched."""
        if not self.initial_exploration_done or len(self.region_spot_history) == 0:
            return current_region
            
        best_region = current_region
        best_reliability = -1.0
        
        for idx, history in enumerate(self.region_spot_history):
            if history["total_count"] > 0:
                reliability = history["reliability"]
                # Penalize recent switches
                if idx == current_region and self.time_since_last_switch < 3:
                    reliability *= 0.7
                
                if reliability > best_reliability:
                    best_reliability = reliability
                    best_region = idx
        
        return best_region

    def _calculate_time_metrics(self) -> Tuple[float, float, float]:
        """Calculate key time metrics."""
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = self.task_duration - work_done
        time_remaining = self.deadline - elapsed
        
        return elapsed, remaining_work, time_remaining

    def _should_switch_to_ondemand(self, remaining_work: float, time_remaining: float) -> bool:
        """Determine if we should switch to on-demand based on time constraints."""
        # Conservative buffer: need at least 2 time steps margin
        time_needed_with_buffer = remaining_work + (2 * self.restart_overhead)
        
        # If we're running out of time, switch to on-demand
        if time_remaining < time_needed_with_buffer:
            return True
        
        # If spot has been unreliable recently
        if self.consecutive_failures >= 2:
            return True
            
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update region history
        current_region = self.env.get_current_region()
        self._update_region_history(current_region, has_spot)
        
        # Update time since last switch
        self.time_since_last_switch += 1
        
        # Calculate time metrics
        elapsed, remaining_work, time_remaining = self._calculate_time_metrics()
        
        # Check if we should do initial exploration
        if not self.initial_exploration_done and elapsed < 3600:  # First hour
            self.regions_explored += 1
            if self.regions_explored >= min(3, self.env.get_num_regions()):
                self.initial_exploration_done = True
                self.best_spot_region = self._get_best_spot_region(current_region)
        
        # Update best region periodically
        if elapsed % 7200 < 3600:  # Every 2 hours
            self.best_spot_region = self._get_best_spot_region(current_region)
        
        # Check if we're in final push mode
        if remaining_work > 0 and time_remaining < remaining_work + self.restart_overhead:
            # Must use on-demand to guarantee completion
            if current_region != self.best_spot_region:
                self.env.switch_region(self.best_spot_region)
                self.time_since_last_switch = 0
            return ClusterType.ON_DEMAND
        
        # Handle state transitions
        if self.state == State.INIT:
            if has_spot:
                self.state = State.SPOT_RUNNING
                return ClusterType.SPOT
            else:
                self.state = State.SPOT_SEARCH
                # Try to switch to best region
                if current_region != self.best_spot_region:
                    self.env.switch_region(self.best_spot_region)
                    self.time_since_last_switch = 0
                return ClusterType.NONE
                
        elif self.state == State.SPOT_SEARCH:
            if has_spot:
                self.state = State.SPOT_RUNNING
                self.current_spot_streak = 0
                self.consecutive_failures = 0
                return ClusterType.SPOT
            else:
                # Try another region if current one is bad
                if self.time_since_last_switch > 2 and current_region != self.best_spot_region:
                    self.env.switch_region(self.best_spot_region)
                    self.time_since_last_switch = 0
                return ClusterType.NONE
                
        elif self.state == State.SPOT_RUNNING:
            if not has_spot:
                self.state = State.SPOT_SEARCH
                self.consecutive_failures += 1
                # Try best region immediately
                if current_region != self.best_spot_region:
                    self.env.switch_region(self.best_spot_region)
                    self.time_since_last_switch = 0
                return ClusterType.NONE
            else:
                self.current_spot_streak += 1
                self.consecutive_failures = 0
                
                # Check if we should continue with spot
                if self._should_switch_to_ondemand(remaining_work, time_remaining):
                    self.state = State.ON_DEMAND
                    return ClusterType.ON_DEMAND
                    
                return ClusterType.SPOT
                
        elif self.state == State.ON_DEMAND:
            # Consider switching back to spot if conditions improve
            if (has_spot and 
                not self._should_switch_to_ondemand(remaining_work, time_remaining) and
                self.current_spot_streak > 0):  # Had some success with spot before
                self.state = State.SPOT_RUNNING
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Default fallback
        if has_spot and not self._should_switch_to_ondemand(remaining_work, time_remaining):
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND