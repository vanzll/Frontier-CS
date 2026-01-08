import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "CantBeLateStrategy"

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
        1. Calculate slack time. If deadline is approaching, force ON_DEMAND to guarantee completion.
        2. If slack is sufficient and Spot is available, use SPOT (cheapest).
        3. If slack is sufficient but Spot is unavailable, switch to next region and wait (NONE).
           This cycles through regions to find one with Spot availability while preserving money.
        """
        # Gather environment state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        done_time = sum(self.task_done_time)
        work_remaining = self.task_duration - done_time
        
        # Calculate time budget
        time_left = self.deadline - elapsed
        
        # Calculate time needed to finish if we commit to ON_DEMAND now.
        # If we are already running OD, we only need to pay the remaining overhead (if any).
        # If we are not OD (Spot or None), switching to OD incurs full restart overhead.
        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_cost = self.remaining_restart_overhead
        else:
            overhead_cost = self.restart_overhead
            
        # Safety buffer: 3 timesteps.
        # This handles discrete timestep quantization and provides a margin of safety.
        safety_buffer = 3.0 * gap
        
        required_time_on_od = work_remaining + overhead_cost + safety_buffer
        
        # 1. Panic Mode: If slack is critically low, force On-Demand
        if time_left < required_time_on_od:
            return ClusterType.ON_DEMAND
            
        # 2. Cost Optimization: Use Spot if available
        if has_spot:
            return ClusterType.SPOT
        
        # 3. Search Mode: Spot unavailable in current region
        # Switch to next region and Wait (NONE) to check its status next step.
        # We use NONE because we don't know the new region's availability yet,
        # and returning SPOT blindly would raise an error if unavailable.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        self.env.switch_region(next_region)
        
        return ClusterType.NONE