import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

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
        """
        # Calculate work remaining
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time remaining
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate buffer parameters
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate the time required to complete the task using On-Demand (OD).
        # We add 'overhead' to be conservative: if we switch to OD or restart,
        # we incur this overhead. Even if we are already on OD, treating this as
        # required buffer ensures we don't cut it too close.
        time_needed_od = work_remaining + overhead
        
        # Safety Buffer:
        # We must decide actions for the *next* 'gap' seconds.
        # If we choose not to run OD now, we lose 'gap' seconds of potential OD time (or slack).
        # We switch to OD if time_remaining is getting close to the minimum time needed.
        # A buffer of 2.0 * gap provides a safety margin against granular updates.
        safety_buffer = 2.0 * gap
        
        # Panic Condition: If we are close to the deadline, force On-Demand.
        if time_remaining <= time_needed_od + safety_buffer:
            return ClusterType.ON_DEMAND

        # Cost Optimization Strategy:
        # If we have slack, we prefer Spot instances to save money.
        if has_spot:
            # Spot is available in the current region. Use it.
            return ClusterType.SPOT
        else:
            # Spot is not available in the current region.
            # Since we have sufficient slack (checked above), we can afford to "search"
            # for a better region rather than paying for On-Demand immediately.
            
            # Switch to the next region (Round Robin strategy)
            num_regions = self.env.get_num_regions()
            current_region = self.env.get_current_region()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            
            # We return NONE for this step because:
            # 1. We cannot use Spot in the *current* region (has_spot is False).
            # 2. We just switched, so we must wait for the next step to check availability in the new region.
            return ClusterType.NONE