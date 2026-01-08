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
        
        # Cache for efficient sum calculation of done segments
        self._last_done_len = 0
        self._last_done_sum = 0.0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Efficiently calculate total work done
        current_len = len(self.task_done_time)
        if current_len > self._last_done_len:
            new_segments = self.task_done_time[self._last_done_len:]
            self._last_done_sum += sum(new_segments)
            self._last_done_len = current_len
            
        done = self._last_done_sum
        needed = max(0.0, self.task_duration - done)
        elapsed = self.env.elapsed_seconds
        time_left = max(0.0, self.deadline - elapsed)
        
        # Calculate overhead cost if we were to ensure execution via On-Demand (OD)
        # If currently on OD, we pay remaining overhead (if any)
        # If not on OD, we must pay full restart overhead to switch to OD
        if last_cluster_type == ClusterType.ON_DEMAND:
            overhead_cost = self.remaining_restart_overhead
        else:
            overhead_cost = self.restart_overhead
            
        # Panic Buffer:
        # Use 5% of needed work, 2 time steps, or 30 mins (1800s), whichever is larger.
        # This provides a safety margin to switch to OD before the deadline becomes impossible.
        buffer_time = max(needed * 0.05, 2.0 * self.env.gap_seconds, 1800.0)
        
        # Panic Mode: If remaining time is tight, force On-Demand to guarantee finish
        if time_left < (needed + overhead_cost + buffer_time):
            return ClusterType.ON_DEMAND
            
        # Economy Mode: Use Spot if available in current region
        if has_spot:
            return ClusterType.SPOT
            
        # Spot unavailable in current region: Switch strategy
        # Attempt to find spot in other regions by switching in round-robin fashion
        num_regions = self.env.get_num_regions()
        if num_regions > 1:
            curr_region = self.env.get_current_region()
            next_region = (curr_region + 1) % num_regions
            self.env.switch_region(next_region)
            
        # We must return NONE after switching (or if no spot exists) because we cannot
        # verify spot availability in the new region until the next step.
        # Returning SPOT blindly raises an error if unavailable.
        return ClusterType.NONE