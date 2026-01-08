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

        # Custom initialization of strategy state and parameters
        num_regions = len(config.get('trace_files', []))
        self.num_regions = num_regions
        self.consecutive_no_spot = {i: 0 for i in range(num_regions)}

        # If slack time drops below this threshold (as a fraction of total task
        # duration), switch to On-Demand to be safe.
        self.ON_DEMAND_SLACK_THRESHOLD = self.task_duration * 0.15

        # If Spot is unavailable, wait for this many timesteps before
        # attempting to switch to a different region.
        self.SWITCH_PATIENCE = 2

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
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done

        # 1. Termination Check: If the task is done, do nothing to save cost.
        if work_rem <= 0:
            return ClusterType.NONE

        current_region = self.env.get_current_region()
        time_rem = self.deadline - self.env.elapsed_seconds

        # If we are already past the deadline, we must try to make progress.
        if time_rem < 0:
            return ClusterType.ON_DEMAND

        slack = time_rem - work_rem

        # 2. Panic Mode: If slack is less than one restart overhead, we can't
        # risk a preemption. We must use On-Demand to guarantee completion.
        if slack <= self.restart_overhead:
            return ClusterType.ON_DEMAND

        # 3. Main Logic: Not in panic mode, decide based on Spot availability.
        if has_spot:
            # Spot is available and we have enough slack. Use it.
            self.consecutive_no_spot[current_region] = 0
            return ClusterType.SPOT
        else:
            # Spot is not available. Decide whether to use On-Demand, wait, or switch.
            self.consecutive_no_spot[current_region] += 1

            # If slack is getting low, be safe and use On-Demand.
            if slack <= self.ON_DEMAND_SLACK_THRESHOLD:
                return ClusterType.ON_DEMAND

            # We have sufficient slack, so we can afford to wait or switch.
            if self.consecutive_no_spot[current_region] < self.SWITCH_PATIENCE:
                # We haven't waited long. Let's wait for Spot to recover.
                return ClusterType.NONE
            else:
                # We've been patient. Time to try another region.
                if self.num_regions > 1:
                    next_region = (current_region + 1) % self.num_regions
                    self.env.switch_region(next_region)
                    # Reset the counter for the region we are leaving.
                    self.consecutive_no_spot[current_region] = 0

                # After deciding to switch (or if there's only one region and
                # we can't switch), we do nothing in this step.
                # The switch itself incurs an overhead.
                return ClusterType.NONE