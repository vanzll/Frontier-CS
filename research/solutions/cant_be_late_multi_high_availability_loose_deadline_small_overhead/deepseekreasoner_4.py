import json
import math
from argparse import Namespace
from typing import List

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "efficient_scheduler"

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
        
        # Precompute constants
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        
        # State tracking
        self.region_history = []
        self.spot_availability_history = []
        self.consecutive_spot_failures = 0
        self.last_action = ClusterType.NONE
        self.region_switch_count = 0
        
        return self

    def _compute_time_needed(self, remaining_work: float, use_ondemand: bool) -> float:
        """
        Compute time needed to finish remaining work.
        """
        if use_ondemand:
            # On-demand: no overheads after starting
            return remaining_work
        else:
            # Spot: assume some overheads
            # Estimate based on historical failure rate
            if len(self.spot_availability_history) > 0:
                failure_rate = 1.0 - (sum(self.spot_availability_history) / len(self.spot_availability_history))
            else:
                failure_rate = 0.1  # conservative default
            
            # Expected overheads = failures * overhead
            expected_overheads = failure_rate * (remaining_work / self.env.gap_seconds) * self.restart_overhead
            return remaining_work + expected_overheads

    def _get_best_region(self) -> int:
        """
        Choose the best region based on historical spot availability.
        """
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # If no history, use round-robin
        if len(self.region_history) < num_regions * 2:
            return (current_region + 1) % num_regions
        
        # Calculate spot availability score for each region
        region_scores = []
        for region in range(num_regions):
            region_indices = [i for i, r in enumerate(self.region_history) if r == region]
            if region_indices:
                # Get corresponding spot availability
                availabilities = [self.spot_availability_history[i] for i in region_indices]
                score = sum(availabilities) / len(availabilities)
                region_scores.append((score, region))
            else:
                region_scores.append((0.5, region))  # neutral score
        
        # Return region with highest spot availability
        region_scores.sort(reverse=True)
        return region_scores[0][1]

    def _should_switch_to_ondemand(self, remaining_work: float) -> bool:
        """
        Determine if we should switch to on-demand.
        """
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If very little time left, use on-demand
        if time_left < remaining_work + self.restart_overhead:
            return True
        
        # If we've had many consecutive spot failures
        if self.consecutive_spot_failures >= 3:
            return True
        
        # If remaining work can be done quickly with on-demand
        # and we're running out of slack time
        slack_ratio = time_left / (remaining_work + 1e-6)
        
        # Conservative threshold: switch if slack ratio < 1.5
        if slack_ratio < 1.5:
            return True
            
        # If we've switched regions too many times without progress
        if self.region_switch_count > 10 and remaining_work > self.task_duration * 0.7:
            return True
            
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update history
        self.region_history.append(self.env.get_current_region())
        self.spot_availability_history.append(has_spot)
        
        # Keep only recent history (last 100 steps)
        if len(self.region_history) > 100:
            self.region_history.pop(0)
            self.spot_availability_history.pop(0)
        
        # Update consecutive spot failures counter
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = 0
        
        # Check if we're in restart overhead
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
        
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        remaining_work = max(0, self.task_duration - work_done)
        
        # If no work left, pause
        if remaining_work <= 1e-6:
            return ClusterType.NONE
        
        # Check if we should use on-demand
        if self._should_switch_to_ondemand(remaining_work):
            # If currently in a region without spot, consider switching first
            if not has_spot and last_cluster_type != ClusterType.ON_DEMAND:
                best_region = self._get_best_region()
                current_region = self.env.get_current_region()
                if best_region != current_region:
                    self.env.switch_region(best_region)
                    self.region_switch_count += 1
                    return ClusterType.NONE
            return ClusterType.ON_DEMAND
        
        # Try to use spot
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot not available, try to switch region
            best_region = self._get_best_region()
            current_region = self.env.get_current_region()
            
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.region_switch_count += 1
                # After switching, pause for this timestep
                return ClusterType.NONE
            else:
                # Stay in current region but pause this timestep
                # (waiting for spot to become available)
                return ClusterType.NONE