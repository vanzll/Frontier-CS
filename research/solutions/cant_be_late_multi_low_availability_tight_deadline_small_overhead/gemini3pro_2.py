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
        # Calculate progress and time metrics
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds

        # If task is already completed, do not schedule further resources
        if work_left <= 0:
            return ClusterType.NONE

        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        # Determine Safety Threshold
        # We must guarantee task completion before deadline.
        # If we use On-Demand, we need roughly (work_left + overhead) time.
        # We add a buffer of 2.5 * gap_seconds to account for:
        # 1. Discrete time steps (we decide per gap)
        # 2. Potential overhead variations or pre-step delays
        # If remaining time falls below this threshold, we MUST use On-Demand to avoid failure penalty.
        safety_threshold = work_left + overhead + 2.5 * gap

        # 1. Panic Mode: Force On-Demand if time is tight
        if time_left < safety_threshold:
            return ClusterType.ON_DEMAND

        # 2. Cost Optimization: Use Spot if available
        if has_spot:
            return ClusterType.SPOT

        # 3. Hunt Mode: Spot unavailable in current region, but we have slack.
        # Switch to the next region and return NONE (pause) to check availability in the next step.
        # This incurs 'gap' time loss but searches for cheaper Spot resources.
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        next_region = (current_region + 1) % num_regions
        
        self.env.switch_region(next_region)
        return ClusterType.NONE