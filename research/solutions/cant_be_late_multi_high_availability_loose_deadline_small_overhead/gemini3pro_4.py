import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "efficient_solution"

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

        Available attributes:
        - self.env.get_current_region(): Get current region index
        - self.env.get_num_regions(): Get total number of regions
        - self.env.switch_region(idx): Switch to region by index
        - self.env.elapsed_seconds: Current time elapsed
        - self.task_duration: Total task duration needed (seconds)
        - self.deadline: Deadline time (seconds)
        - self.restart_overhead: Restart overhead (seconds)
        - self.task_done_time: List of completed work segments
        - self.remaining_restart_overhead: Current pending overhead

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Calculate current progress and time
        current_time = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_remaining = self.deadline - current_time
        
        # Calculate slack: Time available minus time strictly needed.
        # We subtract one restart_overhead conservatively to account for at least one setup/switch.
        slack = time_remaining - (remaining_work + self.restart_overhead)
        
        # Safety buffer:
        # If slack drops below this threshold, we switch to On-Demand to guarantee completion.
        # We set it to 2 hours or 2 time steps, whichever is larger, to safely absorb overheads and step granularity.
        safety_buffer = max(7200.0, 2.0 * self.env.gap_seconds)
        
        # 1. Safety Check: If near deadline, force reliable On-Demand
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND

        # 2. Prefer Spot if available in the current region
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Spot unavailable in current region
        else:
            # Switch to next region (round-robin) to search for Spot availability
            current_region = self.env.get_current_region()
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            
            # Determine action for this step while switching:
            # Since has_spot was False for the current region, we cannot return SPOT this step.
            # We must choose between NONE (wait/switch) or ON_DEMAND (work/switch).
            
            # If we have plenty of slack, return NONE to save money.
            # We add gap_seconds to the buffer because returning NONE consumes one step of time.
            if slack > safety_buffer + self.env.gap_seconds:
                return ClusterType.NONE
            else:
                # If slack is getting tighter, use On-Demand to avoid wasting time
                return ClusterType.ON_DEMAND