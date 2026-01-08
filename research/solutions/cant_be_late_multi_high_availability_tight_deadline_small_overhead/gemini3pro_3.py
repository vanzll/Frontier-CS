import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Adaptive multi-region scheduling strategy."""

    NAME = "AdaptiveCostOptimizer"

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
        Prioritizes deadline guarantee, then minimizes cost by hunting for Spot instances.
        """
        # Retrieve current state
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        current_overhead = self.remaining_restart_overhead
        
        # Calculate time required if we switch to reliable On-Demand now.
        # OD execution is continuous, so we need exactly (overhead + work) seconds.
        time_needed_od = current_overhead + work_remaining
        time_available = self.deadline - elapsed
        
        # Calculate Slack
        # We need a safety buffer because decisions are made in discrete 'gap_seconds' intervals.
        # If we skip the current step (return NONE), we lose 'gap_seconds' of time.
        # We must ensure we switch to OD before (deadline - time_needed_od) is reached.
        # A buffer of 2.0 * gap allows us to waste this step and still safely catch the deadline next step.
        gap = self.env.gap_seconds
        safety_buffer = 2.0 * gap
        
        slack = time_available - time_needed_od
        
        # 1. DEADLINE PROTECTION (Panic Mode)
        # If slack is low, we cannot afford to search or wait. Force On-Demand.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. COST MINIMIZATION (Opportunistic Mode)
        # If we have plenty of time, try to run on Spot.
        if has_spot:
            # Spot is available in the current region. Best option.
            return ClusterType.SPOT
        else:
            # Spot is NOT available in the current region.
            # Since we have slack, we should look for Spot elsewhere rather than paying for OD.
            
            # Switch to the next region (Round-Robin search)
            # This incurs a restart overhead (penalty to time), but we have slack to absorb it.
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            
            # After switching, we don't know if the new region has Spot available (checked next step).
            # We return NONE to advance time by one step without incurring cost.
            # This consumes 'gap_seconds' of our slack.
            return ClusterType.NONE