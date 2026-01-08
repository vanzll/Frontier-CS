import json
from argparse import Namespace
import numpy as np
from typing import List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

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
        
        # Pre-calculate important values
        self.spot_price = 0.9701  # $/hr
        self.on_demand_price = 3.06  # $/hr
        self.gap_hours = self.env.gap_seconds / 3600.0
        
        # State tracking
        self.region_history = {}  # Track spot availability history per region
        self.current_strategy = "conservative"
        self.switch_counter = 0
        self.consecutive_failures = 0
        
        return self

    def _calculate_remaining_work_hours(self) -> float:
        """Calculate remaining work in hours."""
        work_done_seconds = sum(self.task_done_time)
        remaining_work_seconds = max(0, self.task_duration - work_done_seconds)
        return remaining_work_seconds / 3600.0

    def _calculate_time_remaining_hours(self) -> float:
        """Calculate time remaining until deadline in hours."""
        return (self.deadline - self.env.elapsed_seconds) / 3600.0

    def _update_region_history(self, region_idx: int, has_spot: bool):
        """Update spot availability history for a region."""
        if region_idx not in self.region_history:
            self.region_history[region_idx] = []
        self.region_history[region_idx].append(1 if has_spot else 0)
        
        # Keep only recent history (last 50 steps)
        if len(self.region_history[region_idx]) > 50:
            self.region_history[region_idx] = self.region_history[region_idx][-50:]

    def _calculate_region_reliability(self, region_idx: int) -> float:
        """Calculate spot reliability for a region based on history."""
        if region_idx not in self.region_history or not self.region_history[region_idx]:
            return 0.5  # Default reliability if no history
        
        history = self.region_history[region_idx]
        recent = history[-min(20, len(history)):]  # Last 20 or fewer
        return sum(recent) / len(recent)

    def _find_best_alternative_region(self, current_region: int, has_spot_current: bool) -> Tuple[int, float]:
        """Find the best alternative region based on reliability."""
        best_region = current_region
        best_reliability = self._calculate_region_reliability(current_region) if has_spot_current else 0
        
        num_regions = self.env.get_num_regions()
        for region in range(num_regions):
            if region == current_region:
                continue
            reliability = self._calculate_region_reliability(region)
            if reliability > best_reliability + 0.1:  # Threshold for switching
                best_region = region
                best_reliability = reliability
        
        return best_region, best_reliability

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        current_region = self.env.get_current_region()
        
        # Update history for current region
        self._update_region_history(current_region, has_spot)
        
        # Calculate critical metrics
        remaining_work_hours = self._calculate_remaining_work_hours()
        time_remaining_hours = self._calculate_time_remaining_hours()
        overhead_hours = self.restart_overhead / 3600.0
        
        # If work is done, return NONE
        if remaining_work_hours <= 0:
            return ClusterType.NONE
        
        # Check if we're in critical time (must use on-demand to finish)
        min_time_needed = remaining_work_hours + overhead_hours
        if time_remaining_hours < min_time_needed * 1.2:  # 20% safety margin
            return ClusterType.ON_DEMAND
        
        # Check if we're getting close to deadline
        if time_remaining_hours < min_time_needed * 1.5:  # Getting tight
            # Use on-demand if spot not available or region unreliable
            if not has_spot:
                return ClusterType.ON_DEMAND
                
            reliability = self._calculate_region_reliability(current_region)
            if reliability < 0.7:  # Region not reliable enough
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.SPOT
        
        # We have reasonable time remaining
        # Check if we should switch regions
        if not has_spot:
            best_alt_region, alt_reliability = self._find_best_alternative_region(
                current_region, has_spot
            )
            
            # Switch only if alternative is significantly better
            if (best_alt_region != current_region and 
                alt_reliability > 0.4 and  # At least somewhat reliable
                self.switch_counter < 5):  # Limit switches
                
                self.env.switch_region(best_alt_region)
                self.switch_counter += 1
                self.consecutive_failures = 0
                return ClusterType.NONE  # Switching causes overhead
        
        # Main decision logic
        if has_spot:
            reliability = self._calculate_region_reliability(current_region)
            
            # If we've had recent failures, be more conservative
            if self.consecutive_failures > 2:
                if reliability > 0.8:
                    return ClusterType.SPOT
                else:
                    # Try to find better region
                    best_alt_region, alt_reliability = self._find_best_alternative_region(
                        current_region, has_spot
                    )
                    if best_alt_region != current_region and alt_reliability > reliability + 0.2:
                        self.env.switch_region(best_alt_region)
                        self.switch_counter += 1
                        return ClusterType.NONE
                    else:
                        return ClusterType.ON_DEMAND
            
            # Normal operation - use spot if reasonably reliable
            if reliability > 0.6:
                self.consecutive_failures = 0
                return ClusterType.SPOT
            elif reliability > 0.4 and time_remaining_hours > min_time_needed * 2:
                # Can afford to try spot in less reliable region
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            # No spot available in current region
            self.consecutive_failures += 1
            
            # Check if we can wait (enough time remaining)
            wait_threshold = remaining_work_hours + overhead_hours * 2
            if time_remaining_hours > wait_threshold * 1.5:
                # Wait for spot to potentially come back
                return ClusterType.NONE
            else:
                # Need to make progress - use on-demand
                return ClusterType.ON_DEMAND