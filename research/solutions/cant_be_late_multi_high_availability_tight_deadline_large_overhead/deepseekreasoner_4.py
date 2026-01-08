import json
from argparse import Namespace
import math
from enum import Enum

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Action(Enum):
    SPOT = 0
    ON_DEMAND = 1
    NONE = 2
    SWITCH_SPOT = 3
    SWITCH_ON_DEMAND = 4


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
        
        # Initialize strategy parameters
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.step_seconds = 0
        self.work_done = 0.0
        self.last_action = Action.NONE
        self.region_stability = {}
        self.region_history = {}
        self.current_region_idx = 0
        self.switch_countdown = 0
        
        return self

    def _get_work_remaining(self) -> float:
        return self.task_duration - sum(self.task_done_time)

    def _get_time_remaining(self) -> float:
        return self.deadline - self.env.elapsed_seconds

    def _get_critical_ratio(self) -> float:
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        
        if time_remaining <= 0:
            return float('inf')
        
        # Account for restart overhead
        effective_time = time_remaining - self.remaining_restart_overhead
        if effective_time <= 0:
            return float('inf')
            
        return work_remaining / effective_time

    def _update_region_stats(self, region_idx: int, had_spot: bool, successful: bool):
        if region_idx not in self.region_stability:
            self.region_stability[region_idx] = {"spot_count": 0, "total_count": 0, "success_count": 0}
            self.region_history[region_idx] = []
        
        stats = self.region_stability[region_idx]
        stats["total_count"] += 1
        if had_spot:
            stats["spot_count"] += 1
        if successful:
            stats["success_count"] += 1
            
        # Keep recent history (last 10 steps)
        self.region_history[region_idx].append(had_spot)
        if len(self.region_history[region_idx]) > 10:
            self.region_history[region_idx].pop(0)

    def _get_region_spot_probability(self, region_idx: int) -> float:
        if region_idx not in self.region_history or not self.region_history[region_idx]:
            return 0.5  # Default assumption
        history = self.region_history[region_idx]
        return sum(history) / len(history)

    def _find_best_alternative_region(self) -> int:
        current = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        best_region = current
        best_score = -float('inf')
        
        for i in range(num_regions):
            if i == current:
                continue
                
            prob = self._get_region_spot_probability(i)
            # Prefer regions with high spot probability
            score = prob * 2.0
            
            # Slightly prefer staying in same region to avoid switches
            if i == self.current_region_idx:
                score += 0.1
                
            if score > best_score:
                best_score = score
                best_region = i
                
        return best_region

    def _should_switch_region(self) -> bool:
        # Don't switch if we're in countdown
        if self.switch_countdown > 0:
            return False
            
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        
        # Don't switch if we're very close to deadline
        if time_remaining < 2 * self.restart_overhead:
            return False
            
        # Only consider switching if we haven't found spot for a while
        current = self.env.get_current_region()
        if current in self.region_history and len(self.region_history[current]) >= 5:
            recent_spot = sum(self.region_history[current][-5:]) / 5
            if recent_spot < 0.3:  # Less than 30% spot availability recently
                return True
                
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        
        # Update region statistics
        had_spot = has_spot
        successful = (last_cluster_type != ClusterType.NONE and 
                     self.remaining_restart_overhead == 0)
        self._update_region_stats(current_region, had_spot, successful)
        
        work_remaining = self._get_work_remaining()
        time_remaining = self._get_time_remaining()
        critical_ratio = self._get_critical_ratio()
        
        # If work is done, do nothing
        if work_remaining <= 0:
            return ClusterType.NONE
            
        # If no time remaining, try anyway
        if time_remaining <= 0:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Handle restart overhead
        if self.remaining_restart_overhead > 0:
            # Use on-demand to clear overhead quickly if time is critical
            if critical_ratio > 0.9 or time_remaining < 4 * self.restart_overhead:
                return ClusterType.ON_DEMAND
            # Otherwise use spot if available
            elif has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        # Check if we should switch region
        if self._should_switch_region():
            best_alt = self._find_best_alternative_region()
            if best_alt != current_region:
                self.env.switch_region(best_alt)
                self.switch_countdown = 3  # Don't switch again immediately
                # After switching, we'll have restart overhead, so use on-demand
                return ClusterType.ON_DEMAND
        
        # Decrease switch countdown
        if self.switch_countdown > 0:
            self.switch_countdown -= 1
        
        # Decision logic based on critical ratio
        if critical_ratio > 1.0:
            # Very critical - must use on-demand
            return ClusterType.ON_DEMAND
        elif critical_ratio > 0.7:
            # Moderately critical - prefer on-demand but use spot if available and time allows
            if has_spot and time_remaining > work_remaining * 1.5:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        elif critical_ratio > 0.4:
            # Somewhat critical - use spot when available
            if has_spot:
                return ClusterType.SPOT
            # If no spot and we have time, wait
            if time_remaining > work_remaining + 2 * self.restart_overhead:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND
        else:
            # Not critical - optimize for cost
            if has_spot:
                return ClusterType.SPOT
            # Wait for spot if we have plenty of time
            if time_remaining > work_remaining * 2.0:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND