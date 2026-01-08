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
        """
        # Calculate current state
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - done)
        
        # Check if task is done
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        deadline = self.deadline
        time_left = deadline - elapsed
        
        overhead = self.restart_overhead
        current_overhead = self.remaining_restart_overhead
        gap = self.env.gap_seconds

        # --- Panic Logic: Ensure Deadline Completion ---
        # Calculate minimum time needed if we commit to On-Demand (OD) right now.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Already on OD: Need to finish current overhead (if any) + remaining work
            time_needed_od = current_overhead + remaining_work
        else:
            # Switching to OD: Incurs full restart overhead + remaining work
            time_needed_od = overhead + remaining_work

        # Define safety buffer.
        # We need enough slack to handle:
        # 1. The quantization of time steps (gap_seconds)
        # 2. Potential small delays or last-minute switches
        # Given 36h deadline and 24h task (12h slack), a 2-hour buffer is safe/conservative.
        # Also ensure buffer is larger than one time step to prevent overshooting.
        buffer = max(7200.0, 1.5 * gap)
        
        # If remaining time is tight, force OD usage to guarantee completion.
        if time_left - time_needed_od < buffer:
            return ClusterType.ON_DEMAND

        # --- Economy Logic: Optimize Cost ---
        # If Spot is available in current region, use it.
        # Even if we were on OD, switching to Spot is cheaper if remaining work > ~10 mins.
        if has_spot:
            return ClusterType.SPOT
        
        # --- Search Logic: Find Spot in other regions ---
        # If no Spot in current region and we have slack, search elsewhere.
        # Switching regions costs no money immediately, but forces overhead if we run.
        # We switch region and return NONE to "peek" at availability in the next step.
        num_regions = self.env.get_num_regions()
        curr_region = self.env.get_current_region()
        next_region = (curr_region + 1) % num_regions
        self.env.switch_region(next_region)
        
        # Pause for this step to avoid Spot unavailability error and wait for switch
        return ClusterType.NONE