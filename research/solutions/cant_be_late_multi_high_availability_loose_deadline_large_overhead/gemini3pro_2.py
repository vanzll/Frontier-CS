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
        """
        # Retrieve environment parameters
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        total_duration = self.task_duration
        done = sum(self.task_done_time)
        
        # Calculate remaining work and time
        remaining_work = total_duration - done
        time_left = deadline - elapsed
        
        # Calculate Slack
        # Slack is the time buffer we have before we MUST run continuously on OD to finish.
        # We subtract overhead to account for the potential restart cost to switch to OD.
        slack = time_left - (remaining_work + overhead)
        
        # Thresholds
        # Critical: If we are close to the wire, prioritize reliability (OD).
        # We use a small buffer (0.5 * gap) to ensure we don't accidentally miss the deadline 
        # due to quantization effects or minimal interruptions.
        CRITICAL_SLACK = 0.5 * gap
        
        # Search: If we have plenty of time, we can trade time for cost savings by searching.
        # We need enough slack to absorb the wait time (1 gap) plus maintain safety buffer.
        SEARCH_SLACK = 3.0 * gap
        
        if has_spot:
            # Current region has Spot capacity.
            # Use Spot unless we are dangerously close to the deadline.
            if slack < CRITICAL_SLACK:
                 return ClusterType.ON_DEMAND
            return ClusterType.SPOT
        else:
            # No Spot in current region.
            if slack > SEARCH_SLACK:
                # We have enough slack to search for a better region.
                # Switch to the next region in a round-robin fashion.
                current_idx = self.env.get_current_region()
                num_regions = self.env.get_num_regions()
                next_idx = (current_idx + 1) % num_regions
                self.env.switch_region(next_idx)
                
                # Return NONE to pause execution for this step.
                # This incurs no monetary cost. We consume 1 gap of time.
                # In the next step, we will check has_spot for the new region.
                return ClusterType.NONE
            else:
                # Slack is tight. We cannot afford to waste time searching.
                # We must make progress now, so pay for On-Demand.
                return ClusterType.ON_DEMAND