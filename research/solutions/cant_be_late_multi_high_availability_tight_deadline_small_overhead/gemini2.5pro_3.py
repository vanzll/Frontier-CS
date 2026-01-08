import json
import math
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

        # Custom initialization for the strategy
        self.num_regions = len(config["trace_files"])
        self.consecutive_spot_failures_by_region = [0] * self.num_regions
        
        # All time values are in seconds after super().__init__
        slack_time = self.deadline - self.task_duration
        
        # A dynamic safety buffer to decide when to switch to ON_DEMAND.
        # It's the larger of a multiple of the overhead or a fraction of the total slack time.
        self.safety_buffer = max(10 * self.restart_overhead, slack_time * 0.15)

        # Cache gap_seconds to avoid repeated access to self.env
        self._gap_seconds = None

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
        if self._gap_seconds is None:
            self._gap_seconds = self.env.gap_seconds

        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_left_until_deadline = self.deadline - self.env.elapsed_seconds

        # --- Urgency Check ("Panic Mode") ---
        # Calculate the minimum time required to finish if we use ON_DEMAND.
        if work_remaining > 0:
            steps_needed = math.ceil(work_remaining / self._gap_seconds)
            time_for_work = steps_needed * self._gap_seconds
        else:
            time_for_work = 0
        
        time_needed_on_demand = time_for_work + self.remaining_restart_overhead

        # If time required (plus buffer) exceeds time left, use ON_DEMAND.
        if time_needed_on_demand + self.safety_buffer >= time_left_until_deadline:
            return ClusterType.ON_DEMAND

        # --- Normal Operation (Cost-Saving Mode) ---
        current_region = self.env.get_current_region()

        if has_spot:
            # Spot is available and cheap; use it.
            self.consecutive_spot_failures_by_region[current_region] = 0
            return ClusterType.SPOT
        else:
            # Spot unavailable; explore other regions.
            self.consecutive_spot_failures_by_region[current_region] += 1
            
            if self.num_regions > 1:
                # Find the best region to switch to (least consecutive failures).
                candidates = []
                for i in range(self.num_regions):
                    if i != current_region:
                        candidates.append((self.consecutive_spot_failures_by_region[i], i))
                
                # Sort by failure count, then by index to break ties.
                candidates.sort()
                
                best_next_region = candidates[0][1]
                self.env.switch_region(best_next_region)

            # After switching or if only one region, wait a step.
            return ClusterType.NONE