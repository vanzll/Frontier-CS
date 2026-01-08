import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A strategy that minimizes cost by prioritizing Spot instances and exploring 
    multiple regions, while guaranteeing deadline satisfaction via a safety fallback 
    to On-Demand instances.
    """

    NAME = "adaptive_spot_explorer"

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

        # Initialize cache for efficient sum of work done
        self._cached_done_sum = 0.0
        self._cached_list_len = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action:
        1. If slack is critical, force On-Demand to guarantee deadline.
        2. If Spot is available, use it to minimize cost.
        3. If Spot unavailable but slack permits, switch region and wait (NONE) to explore.
        """
        # --- 1. Efficiently calculate total work done ---
        current_len = len(self.task_done_time)
        if current_len > self._cached_list_len:
            new_work = sum(self.task_done_time[self._cached_list_len:])
            self._cached_done_sum += new_work
            self._cached_list_len = current_len
        
        work_done = self._cached_done_sum

        # --- 2. Gather environment parameters ---
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        elapsed = self.env.elapsed_seconds
        total_needed = self.task_duration
        deadline = self.deadline

        # --- 3. Calculate Slack ---
        remaining_work = total_needed - work_done
        # Effective remaining includes pending overhead we must clear before real work
        effective_remaining = remaining_work + self.remaining_restart_overhead
        time_left = deadline - elapsed
        slack = time_left - effective_remaining

        # --- 4. Define Safety Threshold ---
        # We need enough buffer to cover:
        # - The overhead if we are forced to switch to On-Demand
        # - Several timesteps (gaps) to account for simulator granularity/delays
        # 4.0 * gap is a conservative safe margin.
        safety_threshold = overhead + (4.0 * gap)

        # --- 5. Decision Logic ---

        # Case A: Safety Violation Risk
        # If we are running out of time, we must use the reliable On-Demand resource.
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND

        # Case B: Greedy Spot Usage
        # If Spot is available in the current region, use it to save costs.
        if has_spot:
            return ClusterType.SPOT

        # Case C: Exploration Mode
        # Spot is unavailable in current region, but we have plenty of slack.
        # Strategy: Switch to the next region and return NONE.
        # Returning NONE allows us to "wait" one step to check the new region's availability
        # without incurring the cost of On-Demand.
        curr_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (curr_region + 1) % num_regions

        self.env.switch_region(next_region)
        return ClusterType.NONE