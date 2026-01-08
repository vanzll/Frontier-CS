import json
from argparse import Namespace
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class Solution(MultiRegionStrategy):
    """Robust multi-region scheduling strategy."""

    NAME = "robust_search_strategy"

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
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = max(0.0, self.task_duration - work_done)
        time_remaining = self.deadline - elapsed
        gap = self.env.gap_seconds
        
        # Calculate minimum time to finish if we switch to On-Demand immediately.
        # This includes clearing any pending overhead and completing the remaining work.
        # self.remaining_restart_overhead represents the time penalty we must pay 
        # before productive work resumes in the current configuration.
        time_needed_od = self.remaining_restart_overhead + work_remaining
        
        # Define a safety buffer. If the remaining slack drops below this buffer,
        # we stop optimizing for cost and switch to On-Demand to guarantee completion.
        # We use 2.5 * gap to ensure we have a margin of roughly 2 time steps 
        # to handle discrete decision boundaries and potential overhead variations.
        buffer = 2.5 * gap
        
        # Panic Mode: If deadline is tight (slack < buffer), use On-Demand immediately.
        # This minimizes the risk of missing the deadline (-100,000 penalty).
        if time_remaining < time_needed_od + buffer:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Mode: We have sufficient slack.
        if has_spot:
            # Spot is available and cheapest ($0.97 vs $3.06). Use it.
            return ClusterType.SPOT
        else:
            # Spot unavailable in current region, but we have slack.
            # Strategy: Switch to the next region and return NONE.
            # We assume switching regions is better than running On-Demand in a spot-less region
            # as long as we have slack to burn on the search.
            # Returning NONE saves money ($0 cost) while we wait for the next step to check availability.
            
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            
            return ClusterType.NONE