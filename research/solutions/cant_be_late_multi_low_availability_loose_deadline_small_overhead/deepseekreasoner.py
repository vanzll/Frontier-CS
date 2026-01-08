import json
from argparse import Namespace
import math
from typing import List, Tuple

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "adaptive_multi_region"

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
        
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.price_ratio = self.spot_price / self.ondemand_price
        
        # State tracking
        self.current_region = 0
        self.region_history = []
        self.spot_availability_history = []
        self.consecutive_failures = 0
        self.last_action = ClusterType.NONE
        self.critical_threshold = 0.2  # 20% of remaining time as safety margin
        
        return self

    def _compute_remaining_work(self) -> float:
        """Return remaining work in seconds."""
        return self.task_duration - sum(self.task_done_time)

    def _compute_time_remaining(self) -> float:
        """Return time remaining until deadline in seconds."""
        return self.deadline - self.env.elapsed_seconds

    def _compute_safety_factor(self) -> float:
        """Compute safety factor based on remaining time."""
        time_remaining = self._compute_time_remaining()
        work_remaining = self._compute_remaining_work()
        
        if work_remaining <= 0:
            return float('inf')
        
        # How many standard steps we need if using on-demand
        steps_needed = work_remaining / self.env.gap_seconds
        steps_available = time_remaining / self.env.gap_seconds
        
        if steps_available <= 0:
            return 0
        
        safety = (steps_available - steps_needed) / steps_needed
        return safety

    def _evaluate_region_switch(self) -> int:
        """Evaluate if switching region is beneficial."""
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # Simple round-robin if we've had recent failures
        if self.consecutive_failures > 2:
            return (current_region + 1) % num_regions
        
        return current_region  # Stay put

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update tracking
        self.last_action = last_cluster_type
        self.current_region = self.env.get_current_region()
        
        # Record history
        self.region_history.append(self.current_region)
        self.spot_availability_history.append(has_spot)
        if len(self.region_history) > 100:
            self.region_history.pop(0)
            self.spot_availability_history.pop(0)
        
        # Check if done
        if self._compute_remaining_work() <= 0:
            return ClusterType.NONE
        
        # Check if already failed (past deadline)
        if self._compute_time_remaining() <= 0:
            return ClusterType.NONE
        
        # Get critical metrics
        work_remaining = self._compute_remaining_work()
        time_remaining = self._compute_time_remaining()
        safety = self._compute_safety_factor()
        
        # Critical condition: must use on-demand to guarantee completion
        if time_remaining < work_remaining + self.restart_overhead:
            return ClusterType.ON_DEMAND
        
        # High risk condition: low safety margin
        if safety < self.critical_threshold:
            # Use on-demand to ensure progress
            if last_cluster_type != ClusterType.ON_DEMAND:
                self.consecutive_failures = 0
            return ClusterType.ON_DEMAND
        
        # Check if we should switch regions
        target_region = self._evaluate_region_switch()
        if target_region != self.current_region:
            self.env.switch_region(target_region)
            self.current_region = target_region
            self.consecutive_failures = 0
            # After switching, be conservative for one step
            if has_spot and safety > 0.3:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        # Normal operation: try to use spot when available
        if has_spot:
            # If we just failed on spot recently, be cautious
            if self.consecutive_failures > 0:
                # Use on-demand for a while after failures
                if self.consecutive_failures > 1:
                    self.consecutive_failures -= 1
                    return ClusterType.ON_DEMAND
            
            # Good conditions for spot
            if safety > 0.25:
                self.consecutive_failures = 0
                return ClusterType.SPOT
        
        # No spot available or too risky
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_failures += 1
        
        # Default to on-demand
        return ClusterType.ON_DEMAND