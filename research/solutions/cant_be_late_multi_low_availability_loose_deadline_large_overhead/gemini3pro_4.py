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
        # Calculate current progress and time constraints
        completed_work = sum(self.task_done_time)
        remaining_work = self.task_duration - completed_work
        elapsed_time = self.env.elapsed_seconds
        time_left = self.deadline - elapsed_time
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds

        # --- Safety Logic ---
        # Calculate minimum time required to finish if we use On-Demand (guaranteed capacity).
        # We must account for startup overhead if we are not already running On-Demand.
        
        startup_cost = 0.0
        if last_cluster_type == ClusterType.ON_DEMAND:
            # If already On-Demand, we only wait for any currently pending overhead
            startup_cost = self.remaining_restart_overhead
        else:
            # If switching to On-Demand from Spot/None, we pay the full restart overhead
            startup_cost = overhead
            
        required_time_od = remaining_work + startup_cost
        
        # We need a safety buffer. If we attempt a Spot step and it fails (unavailable or preempted),
        # we lose 'gap' seconds of time. We must ensure that AFTER losing that time, 
        # we still have enough time to finish on On-Demand.
        # Buffer = gap * 1.5 provides a robust margin.
        safety_buffer = gap * 1.5
        
        if time_left <= (required_time_od + safety_buffer):
            # Critical zone: insufficient slack to risk Spot failure. 
            # Must run On-Demand to guarantee deadline.
            return ClusterType.ON_DEMAND

        # --- Cost Optimization Logic ---
        # If we have enough slack, prioritize Spot instances (cheaper).
        
        if has_spot:
            # Spot is available in the current region. Use it.
            return ClusterType.SPOT
        else:
            # Spot is unavailable in current region.
            # Strategy: Switch to the next region (Round-Robin) to search for availability.
            current_idx = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_idx = (current_idx + 1) % num_regions
            
            self.env.switch_region(next_idx)
            
            # Return NONE for this step to account for "switching" or "travel" time.
            # In the next step, has_spot will reflect the new region's status.
            return ClusterType.NONE