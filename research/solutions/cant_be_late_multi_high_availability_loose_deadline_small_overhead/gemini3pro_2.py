import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "CostAwareGreedyStrategy"  # REQUIRED: unique identifier

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
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        work_needed = self.task_duration - work_done
        
        if work_needed <= 0:
            return ClusterType.NONE

        # Calculate time constraints
        elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - elapsed
        
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate safety buffer
        # We must ensure we have enough time to finish using On-Demand instances (work_needed + overhead).
        # We add a buffer to account for:
        # 1. The granularity of the simulation steps (gap_seconds)
        # 2. Potential overheads and a margin of safety to avoid the massive failure penalty.
        # A buffer of max(3 steps, 2 hours) is conservative enough given the 24h slack.
        safety_buffer = max(3.0 * gap, 7200.0)
        
        min_required_time = work_needed + overhead + safety_buffer
        
        # Panic Mode: If time is tight, switch to On-Demand to guarantee completion
        if time_remaining < min_required_time:
            return ClusterType.ON_DEMAND

        # Economy Mode: Prefer Spot instances to minimize cost
        if has_spot:
            # Spot available in current region, use it
            return ClusterType.SPOT
        else:
            # Spot not available in current region.
            # Switch to the next region (Round Robin) and wait (NONE) for one step
            # to check availability in the new region.
            curr_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (curr_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            return ClusterType.NONE