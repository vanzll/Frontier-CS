import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"

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
        # Current state
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        needed = self.task_duration - done
        time_left = self.deadline - elapsed
        
        overhead_full = self.restart_overhead
        overhead_remaining = self.remaining_restart_overhead
        
        # Panic Logic: Check if we must use On-Demand to meet deadline
        # Calculate time required to finish if we commit to On-Demand NOW
        if last_cluster_type == ClusterType.ON_DEMAND:
            # If already OD, we only pay the remaining overhead (if booting)
            switch_overhead = overhead_remaining if overhead_remaining > 0 else 0
        else:
            # If not OD, we must pay full overhead to switch/start OD
            switch_overhead = overhead_full
            
        time_req_od = needed + switch_overhead
        
        # Buffer to account for time step granularity (gap_seconds) and safety
        # We need at least one gap to make a decision, plus margin
        buffer = 2.0 * self.env.gap_seconds
        
        # If time is tight, strictly use On-Demand
        if time_left < time_req_od + buffer:
            return ClusterType.ON_DEMAND
            
        # Optimization: If we are already on OD and the work remaining is less than 
        # the overhead to switch back to Spot, stay on OD to save time/cost trade-off.
        # (Spot is cheaper, but paying overhead for small work might not be worth it 
        # or risky close to deadline).
        if last_cluster_type == ClusterType.ON_DEMAND and needed < overhead_full:
            return ClusterType.ON_DEMAND

        # Spot Strategy
        if has_spot:
            # Spot is available, use it (cheapest option)
            return ClusterType.SPOT
        else:
            # Spot unavailable in current region.
            # Since we are not in panic mode, we hunt for Spot in other regions.
            
            # Switch to next region in a round-robin fashion
            num_regions = self.env.get_num_regions()
            current_idx = self.env.get_current_region()
            next_idx = (current_idx + 1) % num_regions
            
            self.env.switch_region(next_idx)
            
            # Return NONE to bridge the time step.
            # The next _step call will reflect availability in the new region.
            return ClusterType.NONE