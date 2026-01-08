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

        # Initialize strategy state
        # Track how many regions we have checked consecutively without finding Spot
        self.regions_checked_count = 0
        # Track how long we have stayed in a fallback region
        self.forcing_od_stay_count = 0
        # Threshold to restart searching for Spot after sticking to OD for a while
        self.search_reset_threshold = 2

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Current environment state
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # Calculate progress and slack
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If work is practically done, handle gracefully (though env usually terminates)
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - time_elapsed
        
        # Pending overhead that must be paid before effective work starts
        pending_overhead = self.remaining_restart_overhead
        
        # Time needed to finish if we run perfectly on On-Demand from now on
        time_needed_safe = work_remaining + pending_overhead
        
        slack = time_remaining - time_needed_safe
        
        # Safety Buffer Calculation:
        # We need a buffer to absorb potential restart overheads if we switch strategies/regions
        # and to account for the granularity of time steps (gap_seconds).
        # We use a conservative buffer to avoid the catastrophic -100,000 penalty.
        # Buffer = (3 * Restart Overhead) + (2 * Time Step Gap)
        gap = self.env.gap_seconds
        safety_buffer = (3 * self.restart_overhead) + (2.0 * gap)
        
        is_critical = slack < safety_buffer

        # 1. Critical Condition: Threat of missing deadline
        # Action: Prioritize guaranteed progress via On-Demand in current region.
        if is_critical:
            # Reset search state as we are locking in
            self.regions_checked_count = 0
            self.forcing_od_stay_count = 0
            return ClusterType.ON_DEMAND

        # 2. Optimal Condition: Spot is available in current region
        # Action: Use Spot to minimize cost.
        if has_spot:
            self.regions_checked_count = 0
            self.forcing_od_stay_count = 0
            return ClusterType.SPOT

        # 3. Search Condition: No Spot, but we have Slack
        # Action: Try to find a region with Spot availability.
        
        # If we have cycled through all regions (or a significant number) without success,
        # we stop switching to avoid accumulating restart overheads which reduce progress.
        # We stick to On-Demand in the current region for a while.
        if self.regions_checked_count >= num_regions:
            self.forcing_od_stay_count += 1
            
            # Periodically reset search to check if Spot has returned elsewhere
            if self.forcing_od_stay_count > self.search_reset_threshold:
                self.regions_checked_count = 0
                self.forcing_od_stay_count = 0
            
            return ClusterType.ON_DEMAND

        # Switch to the next region to probe for Spot availability
        next_region = (current_region + 1) % num_regions
        self.env.switch_region(next_region)
        self.regions_checked_count += 1
        
        # After switching, we do not know if the new region has Spot until the next step.
        # We must not return ClusterType.SPOT here to avoid errors.
        # We return ClusterType.ON_DEMAND to ensure we make progress (minus overhead) 
        # rather than wasting time with ClusterType.NONE.
        return ClusterType.ON_DEMAND