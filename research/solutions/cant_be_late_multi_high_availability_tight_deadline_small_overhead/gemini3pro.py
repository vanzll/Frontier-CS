import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A strategy that prioritizes Spot instances and switches regions to find them
    as long as there is sufficient slack before the deadline.
    """

    NAME = "CostOptimizedStrategy"

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - self.env.elapsed_seconds

        if work_remaining <= 0:
            return ClusterType.NONE

        # 1. Prefer Spot if available in the current region
        if has_spot:
            return ClusterType.SPOT

        # 2. Spot unavailable. Decide whether to Search (Switch & Wait) or use On-Demand.
        # Calculate time needed to finish if we were to switch to On-Demand immediately.
        # We assume worst-case overhead (full restart overhead).
        time_needed_for_od = work_remaining + self.restart_overhead
        
        # Define a safety buffer. We need to ensure that if we waste this step searching,
        # we still have enough time (plus a margin) to finish using OD.
        # 1.5 gaps ensures we handle discretization and potential boundary issues safely.
        safety_buffer = 1.5 * self.env.gap_seconds
        
        future_time_remaining = time_remaining - self.env.gap_seconds
        
        if future_time_remaining > (time_needed_for_od + safety_buffer):
            # We have slack. Switch to the next region and return NONE (pause) to probe.
            # This incurs a time cost (gap_seconds) but saves money if we find Spot later.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            return ClusterType.NONE
        
        # 3. Not enough slack to search safely. Fallback to On-Demand.
        return ClusterType.ON_DEMAND