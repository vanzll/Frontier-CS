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
        # Calculate current progress
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        work_left = self.task_duration - done

        # If work is completed, pause
        if work_left <= 0:
            return ClusterType.NONE

        time_left = self.deadline - elapsed
        gap = self.env.gap_seconds

        # Calculate time needed if we switch to (or continue) On-Demand
        # If we are already running On-Demand, we only need to wait for any remaining overhead
        # If we are NOT running On-Demand, switching to it incurs the full restart overhead
        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_cost = self.remaining_restart_overhead
        else:
            overhead_cost = self.restart_overhead

        needed_for_od = work_left + overhead_cost

        # Safety Buffer:
        # We need to ensure that if we waste this step searching (returning NONE),
        # we still have enough time to finish using On-Demand in the subsequent steps.
        # We add a small 5-minute (300s) margin for safety.
        safety_threshold = needed_for_od + gap + 300.0

        # Panic Mode: If we are close to the deadline, force On-Demand
        if time_left < safety_threshold:
            return ClusterType.ON_DEMAND

        # Economy Mode: Use Spot if available
        if has_spot:
            return ClusterType.SPOT

        # Search Mode: Spot is unavailable in current region, but we have slack.
        # Switch to the next region and return NONE to wait for the switch.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        self.env.switch_region(next_region)

        return ClusterType.NONE