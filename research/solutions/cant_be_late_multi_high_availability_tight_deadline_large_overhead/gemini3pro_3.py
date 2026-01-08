import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "cost_optimized_switcher"  # REQUIRED: unique identifier

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

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        # If task is effectively done, stop
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # Current time status
        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        
        # Parameters
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate Panic Threshold
        # If we choose to probe (switch region & wait), we lose 'gap' seconds.
        # After probing, we need enough time to complete work + pay restart overhead.
        # Threshold = Work + Overhead + Gap + SafetyBuffer
        # If time_left is below this, we cannot afford to search/wait.
        panic_threshold = remaining_work + (1.1 * overhead) + gap

        if time_left < panic_threshold:
            # Panic Mode: Must run now
            if has_spot:
                return ClusterType.SPOT
            else:
                # No Spot available and no time to search -> Force On-Demand
                return ClusterType.ON_DEMAND
        else:
            # Normal Mode: Optimize for cost
            if has_spot:
                return ClusterType.SPOT
            else:
                # No Spot in current region, but have time -> Switch Region
                current_region = self.env.get_current_region()
                num_regions = self.env.get_num_regions()
                
                # Cycle to next region
                next_region = (current_region + 1) % num_regions
                self.env.switch_region(next_region)
                
                # Return NONE to absorb the switch safely and check availability next step
                return ClusterType.NONE