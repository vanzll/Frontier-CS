import json
import math
from argparse import Namespace
from typing import List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


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
        
        # Additional initialization
        self.remaining_work = self.task_duration
        self.spot_price = 0.9701 / 3600  # $ per second
        self.ondemand_price = 3.06 / 3600  # $ per second
        self.time_step = None  # Will be set in first _step call
        
        # Track region history for availability patterns
        self.region_spot_history = {}
        self.region_switches = 0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Initialize time step on first call
        if self.time_step is None:
            self.time_step = self.env.gap_seconds
        
        # Calculate critical values
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # Track spot availability history for current region
        if current_region not in self.region_spot_history:
            self.region_spot_history[current_region] = []
        self.region_spot_history[current_region].append(1 if has_spot else 0)
        
        # Calculate work progress
        completed_work = sum(self.task_done_time)
        self.remaining_work = self.task_duration - completed_work
        
        # Calculate time remaining
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # If work is done, stop everything
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        # Critical condition: if we can't finish even with continuous on-demand
        min_time_needed = self.remaining_work + (self.restart_overhead if last_cluster_type == ClusterType.NONE else 0)
        if time_remaining < min_time_needed:
            # Emergency mode: use on-demand to finish
            if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead <= 0:
                return ClusterType.ON_DEMAND
            elif last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
        
        # Calculate slack ratio (how much time we have vs work needed)
        slack_ratio = time_remaining / self.remaining_work if self.remaining_work > 0 else float('inf')
        
        # Strategic decision making
        if slack_ratio >= 1.5:
            # Plenty of time - aggressively pursue spot
            if has_spot and last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            elif has_spot and self.remaining_restart_overhead <= 0:
                return ClusterType.SPOT
            else:
                # Check other regions for spot availability
                best_region = self._find_best_spot_region(current_region, has_spot)
                if best_region != current_region and self.remaining_restart_overhead <= 0:
                    self.env.switch_region(best_region)
                    self.region_switches += 1
                    # After switching, use spot if available
                    if has_spot:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.NONE
                else:
                    # Wait for spot in current region
                    return ClusterType.NONE
        
        elif slack_ratio >= 1.1:
            # Moderate slack - balanced approach
            if has_spot:
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                elif self.remaining_restart_overhead <= 0:
                    return ClusterType.SPOT
                else:
                    return ClusterType.NONE
            else:
                # Check if we should switch region
                spot_prob_current = self._estimate_spot_probability(current_region)
                
                best_region = current_region
                best_prob = spot_prob_current
                
                for region in range(num_regions):
                    if region != current_region:
                        prob = self._estimate_spot_probability(region)
                        if prob > best_prob:
                            best_prob = prob
                            best_region = region
                
                if (best_region != current_region and 
                    best_prob > spot_prob_current + 0.2 and 
                    self.remaining_restart_overhead <= 0):
                    self.env.switch_region(best_region)
                    self.region_switches += 1
                
                # Use on-demand if spot not available and we're running out of time
                if self.remaining_work / time_remaining > 0.8:
                    if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead <= 0:
                        return ClusterType.ON_DEMAND
                    elif last_cluster_type == ClusterType.ON_DEMAND:
                        return ClusterType.ON_DEMAND
                
                return ClusterType.NONE
        
        else:
            # Low slack - conservative approach
            if has_spot and last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            elif has_spot and self.remaining_restart_overhead <= 0:
                return ClusterType.SPOT
            else:
                # Use on-demand to ensure progress
                if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead <= 0:
                    return ClusterType.ON_DEMAND
                elif last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE

    def _find_best_spot_region(self, current_region: int, has_spot: bool) -> int:
        """
        Find the best region to switch to for spot availability.
        """
        num_regions = self.env.get_num_regions()
        
        # If current region has spot, stay
        if has_spot:
            return current_region
        
        # Simple round-robin search for region with spot
        # In a real implementation, this would use trace data or learning
        for offset in range(1, num_regions):
            test_region = (current_region + offset) % num_regions
            if test_region in self.region_spot_history:
                # Check recent history
                recent_history = self.region_spot_history[test_region][-5:] if len(self.region_spot_history[test_region]) >= 5 else self.region_spot_history[test_region]
                if sum(recent_history) / len(recent_history) > 0.5:
                    return test_region
        
        # Default to current region
        return current_region

    def _estimate_spot_probability(self, region: int) -> float:
        """
        Estimate spot availability probability for a region.
        """
        if region not in self.region_spot_history or not self.region_spot_history[region]:
            return 0.5  # Default assumption
        
        history = self.region_spot_history[region]
        
        # Give more weight to recent observations
        if len(history) >= 10:
            recent = history[-10:]
            return sum(recent) / len(recent)
        else:
            return sum(history) / len(history)