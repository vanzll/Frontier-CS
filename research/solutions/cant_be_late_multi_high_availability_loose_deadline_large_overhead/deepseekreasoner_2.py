import json
import math
from argparse import Namespace
from collections import deque
from typing import List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with adaptive risk management."""

    NAME = "adaptive_risk_manager"

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
        
        # Initialize strategy state
        self.region_history = {}  # region -> availability history
        self.current_region = 0
        self.consecutive_spot_failures = 0
        self.spot_usage_rate = 1.0  # Start optimistic
        self.safety_margin = 0.1  # 10% safety margin for deadlines
        self.min_spot_probability = 0.7  # Minimum confidence to use spot
        self.region_switch_threshold = 2  # Failures before considering switch
        self.last_action = ClusterType.NONE
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Get current region
        current_region = self.env.get_current_region()
        
        # Update region history
        if current_region not in self.region_history:
            self.region_history[current_region] = deque(maxlen=10)
        self.region_history[current_region].append(has_spot)
        
        # Calculate progress metrics
        total_done = sum(self.task_done_time)
        remaining_work = self.task_duration - total_done
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate effective work rates
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Emergency check: if we're running out of time
        if self._is_critical(remaining_work, time_remaining, gap, overhead):
            return ClusterType.ON_DEMAND
        
        # Update spot failure tracking
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = max(0, self.consecutive_spot_failures - 1)
        
        # Calculate spot availability probability for current region
        spot_prob = self._calculate_spot_probability(current_region)
        
        # Consider switching regions if current region has poor spot availability
        if (spot_prob < self.min_spot_probability and 
            self.consecutive_spot_failures >= self.region_switch_threshold):
            best_region = self._find_best_region(current_region)
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.consecutive_spot_failures = 0
                # After switching, use on-demand to ensure progress
                return ClusterType.ON_DEMAND
        
        # Decision logic
        if has_spot:
            # Use spot if we have confidence in availability
            if spot_prob >= self.min_spot_probability:
                # Avoid frequent switching between spot and on-demand
                if last_cluster_type == ClusterType.ON_DEMAND:
                    # Only switch back to spot if we have sufficient time buffer
                    buffer_needed = overhead * 2  # Allow for potential restart
                    if time_remaining - remaining_work > buffer_needed:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
                return ClusterType.SPOT
            else:
                # Low confidence in spot, use on-demand
                return ClusterType.ON_DEMAND
        else:
            # No spot available, use on-demand if we need to make progress
            if remaining_work > 0 and time_remaining > remaining_work + overhead:
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available
                return ClusterType.NONE

    def _is_critical(self, remaining_work: float, time_remaining: float, 
                     gap: float, overhead: float) -> bool:
        """
        Check if situation is critical (must use on-demand to meet deadline).
        """
        # Minimum time needed if using on-demand continuously
        min_time_needed = remaining_work
        
        # Add safety margin
        min_time_needed *= (1 + self.safety_margin)
        
        # If we're cutting it too close, use on-demand
        return time_remaining <= min_time_needed + overhead

    def _calculate_spot_probability(self, region: int) -> float:
        """
        Calculate spot availability probability for a region based on history.
        """
        if region not in self.region_history or not self.region_history[region]:
            return 0.5  # Default assumption
        
        history = self.region_history[region]
        available_count = sum(1 for avail in history if avail)
        return available_count / len(history)

    def _find_best_region(self, current_region: int) -> int:
        """
        Find the region with the best spot availability history.
        """
        num_regions = self.env.get_num_regions()
        best_region = current_region
        best_prob = self._calculate_spot_probability(current_region)
        
        for region in range(num_regions):
            if region == current_region:
                continue
            prob = self._calculate_spot_probability(region)
            if prob > best_prob:
                best_prob = prob
                best_region = region
        
        return best_region