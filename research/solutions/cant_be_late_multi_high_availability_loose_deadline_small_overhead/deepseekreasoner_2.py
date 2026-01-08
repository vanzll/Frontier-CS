import json
from argparse import Namespace
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
import math


class Solution(MultiRegionStrategy):
    NAME = "optimized_strategy"

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
        self.last_region = 0
        self.spot_unavailable_streak = 0
        self.consecutive_spot_runs = 0
        self.region_spot_history = {}
        self.region_switch_cooldown = 0
        self.on_demand_fallback_triggered = False
        self.remaining_work_cache = None
        
        return self

    def _should_use_on_demand(self, remaining_time, remaining_work):
        """Determine if we should use on-demand based on time pressure."""
        if remaining_time <= 0:
            return True
            
        # Calculate time needed with various strategies
        time_needed_ondemand = remaining_work + (self.restart_overhead[0] 
            if self.env.cluster_type != ClusterType.ON_DEMAND else 0)
        
        # If we're critically low on time, use on-demand
        if remaining_time < time_needed_ondemand * 1.2:
            return True
            
        # If spot hasn't been available for a while and we're running out of time
        if self.spot_unavailable_streak > 5 and remaining_time < remaining_work * 2:
            return True
            
        return False

    def _get_best_region(self, current_region):
        """Find the best region to switch to based on history."""
        num_regions = self.env.get_num_regions()
        
        # If we don't have enough history, use round-robin
        if len(self.region_spot_history) < 2:
            return (current_region + 1) % num_regions
            
        # Find region with best spot availability history
        best_region = current_region
        best_score = -1
        
        for region in range(num_regions):
            if region == current_region:
                continue
                
            score = self.region_spot_history.get(region, 0)
            if score > best_score:
                best_score = score
                best_region = region
                
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update region spot history
        current_region = self.env.get_current_region()
        self.region_spot_history[current_region] = self.region_spot_history.get(current_region, 0) + (1 if has_spot else 0)
        
        # Calculate remaining work and time
        work_done = sum(self.task_done_time)
        remaining_work = max(0, self.task_duration[0] - work_done)
        remaining_time = max(0, self.deadline[0] - self.env.elapsed_seconds)
        
        # Update streaks
        if has_spot and last_cluster_type == ClusterType.SPOT:
            self.consecutive_spot_runs += 1
            self.spot_unavailable_streak = 0
        elif not has_spot:
            self.spot_unavailable_streak += 1
            self.consecutive_spot_runs = 0
        else:
            self.consecutive_spot_runs = 0
            self.spot_unavailable_streak = 0
            
        # Update region switch cooldown
        self.region_switch_cooldown = max(0, self.region_switch_cooldown - 1)
        
        # Critical check: if we're about to miss deadline, use on-demand
        if self._should_use_on_demand(remaining_time, remaining_work):
            self.on_demand_fallback_triggered = True
            # If switching from spot to on-demand in same region
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.ON_DEMAND
            # If we're already on-demand, stay there
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            # Otherwise switch to on-demand
            return ClusterType.ON_DEMAND
        
        # If we're in on-demand fallback mode and have time, try to switch back to spot
        if self.on_demand_fallback_triggered and remaining_time > remaining_work * 1.5:
            self.on_demand_fallback_triggered = False
            
        # Prefer spot when available
        if has_spot:
            # If we've had many consecutive spot runs, stay in current region
            if self.consecutive_spot_runs < 3 or self.region_switch_cooldown > 0:
                return ClusterType.SPOT
                
            # Occasionally explore other regions
            if self.consecutive_spot_runs % 7 == 0 and self.region_switch_cooldown == 0:
                best_region = self._get_best_region(current_region)
                if best_region != current_region:
                    self.env.switch_region(best_region)
                    self.region_switch_cooldown = 3
                    self.last_region = best_region
                return ClusterType.SPOT
            return ClusterType.SPOT
        
        # Spot not available in current region
        if self.region_switch_cooldown > 0:
            # Wait out cooldown
            return ClusterType.NONE
            
        # Try switching regions to find spot
        if self.spot_unavailable_streak >= 2:
            best_region = self._get_best_region(current_region)
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.region_switch_cooldown = 2
                self.last_region = best_region
                self.spot_unavailable_streak = 0
                # After switching, wait one step to assess new region
                return ClusterType.NONE
        
        # Default to waiting
        return ClusterType.NONE