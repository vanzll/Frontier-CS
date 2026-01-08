import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cant_be_late_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
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

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Gather current environment state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        duration = self.task_duration
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Calculate remaining work and time
        done = sum(self.task_done_time)
        needed = max(0.0, duration - done)
        time_left = deadline - elapsed
        
        # Define safety buffer
        # We need a buffer to ensure we switch to On-Demand before it's too late.
        # A buffer of 2 * gap_seconds allows for the current step to potentially fail/wait
        # and provides a safety margin against granular time steps.
        buffer = 2.0 * gap
        
        # Calculate time required if we run on On-Demand
        # If currently On-Demand, no overhead is incurred to continue.
        # Otherwise, we must account for restart overhead.
        current_overhead_cost = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead
        od_time_required = needed + current_overhead_cost
        
        # Panic Check: Force On-Demand if remaining time is critical
        if time_left < (od_time_required + buffer):
            return ClusterType.ON_DEMAND
            
        # Standard Strategy: Prefer Spot if available and safe
        if has_spot:
            return ClusterType.SPOT
            
        # Spot Unavailable Handling:
        # If Spot is not available in the current region, switch to the next region.
        # We return NONE for this step to allow the region switch to take effect.
        # The cost of this wait/switch is acceptable because we have sufficient slack (checked above).
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        next_region = (current_region + 1) % num_regions
        self.env.switch_region(next_region)
        
        return ClusterType.NONE