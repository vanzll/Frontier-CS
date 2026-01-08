import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "robust_search_strategy"  # REQUIRED: unique identifier

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
        1. Panic Check: If remaining time is close to the minimum time required to finish 
           using On-Demand instances (including restart overheads), force switch to On-Demand.
        2. Opportunity Check: If Spot is available in the current region, use it.
           - If currently on On-Demand, only switch to Spot if slack is very high to avoid thrashing.
        3. Search Strategy: If no Spot in current region and plenty of slack, switch region 
           and wait (NONE) to check availability in the new region at the next step.
        """
        # Calculate state variables
        current_work = sum(self.task_done_time)
        remaining_work = self.task_duration - current_work
        
        # If task is effectively done, do nothing
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Safety buffer: Must be larger than gap to prevent overshooting the panic threshold
        # during a search/wait step (which consumes 'gap' time).
        buffer = 2.0 * gap

        # --- 1. Panic Logic: Guaranteed Completion ---
        # Calculate time required if we commit to On-Demand right now.
        # If we are not currently running On-Demand, we pay overhead to switch/start.
        time_needed_od = remaining_work
        if last_cluster_type != ClusterType.ON_DEMAND:
            time_needed_od += overhead
            
        # If remaining time is tight, force On-Demand to ensure deadline is met.
        if remaining_time < time_needed_od + buffer:
            return ClusterType.ON_DEMAND

        # --- 2. Opportunity Logic: Prefer Spot ---
        if has_spot:
            # If we are currently on On-Demand, only switch to Spot if we have significant slack.
            # Switching OD -> Spot costs 'overhead' time immediately.
            # We want to ensure we have enough buffer to pay this overhead and potentially 
            # another overhead if we have to switch back later.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if remaining_time > remaining_work + 2 * overhead + buffer:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            # If currently Spot or None, use Spot immediately.
            return ClusterType.SPOT

        # --- 3. Search Logic: Hunt for Spot ---
        # If no Spot in current region, but we have plenty of slack (not in panic mode).
        # We switch to the next region and return NONE.
        # Returning NONE incurs no cost but advances time by 'gap'.
        # In the next step, we will see 'has_spot' for the new region.
        num_regions = self.env.get_num_regions()
        current_region = self.env.get_current_region()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        return ClusterType.NONE