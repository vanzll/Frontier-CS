import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "AggressiveHunterStrategy"

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
        Strategy:
        1. Calculate slack (time remaining - work remaining).
        2. If slack is low (Panic Mode), force On-Demand to guarantee deadline.
        3. If slack is sufficient:
           - If Spot available in current region: Use Spot.
           - If Spot unavailable: Switch to next region and use On-Demand. 
             This "hunts" for Spot availability in other regions while maintaining 
             progress (minus restart overhead).
        """
        # Retrieve state
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        needed = self.task_duration - done
        
        remaining_time = self.deadline - elapsed
        slack = remaining_time - needed
        
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Panic Threshold:
        # If we have less slack than ~2 steps + margin, stop optimizing and ensure finish.
        # This prevents missing the deadline due to switching overheads or preemption.
        panic_threshold = 2.0 * gap + 3.0 * overhead

        if slack < panic_threshold:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            # Hunt Mode:
            # Current region has no Spot. Switch to next region cyclically.
            # Return ON_DEMAND to make progress during the switch (incurs overhead).
            # In the next step, we will be in the new region and check has_spot.
            num_regions = self.env.get_num_regions()
            current_region = self.env.get_current_region()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            return ClusterType.ON_DEMAND