import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cost_aware_scheduler"

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
        
        Strategy:
        1. Calculate remaining work and time.
        2. If time is tight (Panic Mode), force On-Demand to meet deadline.
        3. If time permits, prefer Spot instances.
        4. If Spot is unavailable in current region, switch region and wait (NONE) 
           to search for Spot elsewhere, exploiting the price difference.
        """
        # Current state
        current_elapsed = self.env.elapsed_seconds
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        gap = self.env.gap_seconds
        
        # Progress calculation
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - current_elapsed
        
        # Panic Threshold Calculation
        # We need to ensure we have enough time to finish the work plus overhead.
        # We include a safety buffer proportional to the time step (gap) to account for
        # the latency of decision making (e.g., if we return NONE this step).
        # Buffer = Restart Overhead + 2 * Gap (allows for 1 miss + safety margin)
        panic_buffer = self.restart_overhead + (gap * 2.0)
        
        # Check if we must use On-Demand to meet the deadline
        if time_remaining < (work_remaining + panic_buffer):
            return ClusterType.ON_DEMAND

        # High Slack Logic (Cost Optimization)
        if has_spot:
            # Cheapest option available
            return ClusterType.SPOT
        else:
            # Spot unavailable here, but we have time to search.
            # Switch to next region. We return NONE to avoid paying for On-Demand 
            # or failing a Spot request in the current blind step.
            # The next step will check availability in the new region.
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            return ClusterType.NONE