import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Cost-aware multi-region scheduling strategy."""

    NAME = "CostOptimizedStrategy"

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        Priority 1: Meet Deadline (Panic to On-Demand if tight).
        Priority 2: Minimize Cost (Use Spot if safe, Search if Spot unavailable).
        """
        # Calculate remaining work needed
        needed_work = self.task_duration - sum(self.task_done_time)
        if needed_work <= 1e-6:
            return ClusterType.NONE

        time_now = self.env.elapsed_seconds
        time_left = self.deadline - time_now
        
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Safety Logic:
        # We must ensure that if we try Spot and fail/preempt (losing 'gap' time),
        # we still have enough time to finish using On-Demand.
        # Time required for fallback = Overhead (switch to OD) + Remaining Work.
        # We add 'gap' to account for the potential wasted step.
        # Multiplier 1.05 adds a small buffer for float precision/safety.
        safety_threshold = needed_work + overhead + (gap * 1.05)
        
        # If remaining time is critically low, force On-Demand to guarantee deadline
        if time_left < safety_threshold:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Logic:
        if has_spot:
            # Spot is available and we have slack -> Use Spot (cheapest)
            return ClusterType.SPOT
        else:
            # Spot unavailable in current region, but we have slack.
            # Strategy: Switch to the next region to search for Spot availability.
            # We return NONE to pause execution (0 cost) while switching/checking.
            num_regions = self.env.get_num_regions()
            if num_regions > 1:
                current_region = self.env.get_current_region()
                next_region = (current_region + 1) % num_regions
                self.env.switch_region(next_region)
            
            return ClusterType.NONE