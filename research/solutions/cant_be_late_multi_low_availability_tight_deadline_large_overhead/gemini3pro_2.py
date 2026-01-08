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
        Prioritize deadline (using On-Demand if slack is low), then minimize cost (using Spot).
        """
        # 1. State observation
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done

        # If task is effectively done, stop (though environment usually catches this)
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        time_left = self.deadline - elapsed

        # 2. Safety / Deadline Logic
        # We calculate the time required to finish if we switch to On-Demand NOW.
        # If we are already On-Demand, we don't pay restart overhead to continue.
        # If we are Spot or None, we pay overhead to start On-Demand.
        is_od = (last_cluster_type == ClusterType.ON_DEMAND)
        overhead_cost = 0.0 if is_od else self.restart_overhead
        
        # Buffer to protect against step granularity and minor delays (10 minutes)
        buffer_seconds = 600.0
        
        required_time_for_od = remaining_work + overhead_cost + buffer_seconds

        # If we don't have enough time to gamble with Spot, or if we are already
        # using On-Demand (latching to avoid thrashing), use On-Demand.
        if is_od or time_left < required_time_for_od:
            return ClusterType.ON_DEMAND

        # 3. Cost Minimization Logic (Spot)
        if has_spot:
            # If current region has Spot available, use it.
            return ClusterType.SPOT
        else:
            # If current region lost Spot, switch to the next region and wait.
            # We assume partial observability: we must switch to check availability.
            curr_region_idx = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region_idx = (curr_region_idx + 1) % num_regions
            
            self.env.switch_region(next_region_idx)
            
            # Return NONE for this step. In the next step, has_spot will reflect
            # the availability of the new region. We avoid returning SPOT blindly
            # to prevent errors if the new region also lacks capacity.
            return ClusterType.NONE