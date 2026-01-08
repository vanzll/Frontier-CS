import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Cant-Be-Late Multi-Region Scheduling Strategy."""

    NAME = "CantBeLate_Solution"

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
        
        # Optimization: Cache accumulated work to avoid O(N^2) sum in _step
        self._cached_work = 0.0
        self._last_list_len = 0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Efficiently calculate remaining work
        current_len = len(self.task_done_time)
        if current_len > self._last_list_len:
            # Only sum new segments
            new_work = sum(self.task_done_time[self._last_list_len:current_len])
            self._cached_work += new_work
            self._last_list_len = current_len
            
        remaining_work = self.task_duration - self._cached_work

        # If task is effectively done, do nothing
        if remaining_work <= 1e-4:
            return ClusterType.NONE

        # 2. Gather Environment Parameters
        gap = self.env.gap_seconds
        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time
        overhead = self.restart_overhead

        # 3. Safety Check (Panic Logic)
        # Calculate strict deadline threshold to switch to On-Demand
        # We need: remaining_work + overhead (for potential switch to OD)
        # Buffer:
        # - 1.5 * gap: To ensure we don't start a 'hunt/wait' step that consumes the last safe slot
        # - 10% of remaining_work: General safety margin against variations
        
        safety_buffer = (gap * 1.5) + (remaining_work * 0.1)
        required_time_od = remaining_work + overhead
        
        # If we are close to the deadline, force On-Demand to guarantee completion
        if time_left < (required_time_od + safety_buffer):
            return ClusterType.ON_DEMAND

        # 4. Cost Optimization Logic (Spot Hunting)
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot not available in current region.
            # Strategy: Round-Robin switch to next region.
            # Return NONE to pause execution (save cost) while switching context.
            # Next step will evaluate the new region.
            curr_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (curr_region + 1) % num_regions
            
            self.env.switch_region(next_region)
            return ClusterType.NONE