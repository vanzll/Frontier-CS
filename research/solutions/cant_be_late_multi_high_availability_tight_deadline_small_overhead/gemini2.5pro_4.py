import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    Your multi-region scheduling strategy.
    
    This strategy uses an offline, lookahead approach based on full trace information.
    1.  In `solve`, it pre-processes all spot availability traces to build fast
        lookup tables for future states. This includes the next available spot time
        across all regions and the consecutive availability streak for each region.
    2.  In `_step`, it follows a priority-based decision logic:
        a.  **Safety Net**: It first calculates if finishing on On-Demand is still
            possible before the deadline. If not, it switches to On-Demand
            immediately to guarantee completion (Panic Mode).
        b.  **Spot Optimization**: If there's a safe time buffer, it seeks to use
            the cheaper Spot instances. It checks all regions for current spot
            availability. If found, it switches to the region with the longest
            predicted future uptime to minimize context switching.
        c.  **Strategic Waiting**: If no spot is currently available, it calculates
            the time until the next one appears. If the safety buffer allows for
            this wait, it pauses (NONE) to save costs. Otherwise, it uses
            On-Demand to make progress.
    """

    NAME = "my_strategy"

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

        # Load all trace data into memory for offline planning
        self.num_regions = len(config["trace_files"])
        self.availability = []
        for trace_file in config["trace_files"]:
            with open(trace_file, 'r') as f:
                self.availability.append(json.load(f))

        if not self.availability or not self.availability[0]:
            self.num_timesteps = 0
            return self

        self.num_timesteps = len(self.availability[0])

        # Pre-compute a lookup table for the next timestep with any spot availability.
        self.next_any_spot_timestep = [math.inf] * self.num_timesteps
        next_t_any = math.inf
        for t in range(self.num_timesteps - 1, -1, -1):
            is_any_spot_available = False
            for r in range(self.num_regions):
                if self.availability[r][t]:
                    is_any_spot_available = True
                    break
            if is_any_spot_available:
                next_t_any = t
            self.next_any_spot_timestep[t] = next_t_any

        # Pre-compute a lookup table for streaks of consecutive spot availability.
        self.consecutive_spot = [[0] * self.num_timesteps for _ in range(self.num_regions)]
        for r in range(self.num_regions):
            streak = 0
            for t in range(self.num_timesteps - 1, -1, -1):
                if self.availability[r][t]:
                    streak += 1
                else:
                    streak = 0
                self.consecutive_spot[r][t] = streak

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
        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time

        if time_left <= 0:
            # Past the deadline, but must still choose a valid action.
            return ClusterType.ON_DEMAND

        # Calculate time needed to finish on On-Demand from this point,
        # assuming one restart for the switch.
        od_time_needed = self.restart_overhead + remaining_work

        # 1. Panic Mode: If time left is insufficient for guaranteed completion on On-Demand.
        if od_time_needed >= time_left:
            return ClusterType.ON_DEMAND

        # Use integer division for safe indexing into pre-computed lists.
        current_timestep = int(current_time / self.env.gap_seconds)
        if current_timestep >= self.num_timesteps:
            # If simulation runs past our trace data, fall back to safe On-Demand.
            return ClusterType.ON_DEMAND

        # 2. Opportunistic Spot Usage: Check for currently available spot instances.
        spot_regions = [r for r in range(self.num_regions) if self.availability[r][current_timestep]]

        if spot_regions:
            current_region = self.env.get_current_region()
            if current_region in spot_regions:
                return ClusterType.SPOT
            else:
                # Switch to the best available spot region (one with the longest future streak).
                best_region = max(spot_regions, key=lambda r: self.consecutive_spot[r][current_timestep])
                self.env.switch_region(best_region)
                return ClusterType.SPOT

        # 3. Calculated Waiting: No spot available now. Decide to wait or use On-Demand.
        else:
            t_next_spot_idx = self.next_any_spot_timestep[current_timestep]

            if t_next_spot_idx == math.inf:
                # No more spot available in the future. Must use On-Demand.
                return ClusterType.ON_DEMAND

            wait_steps = t_next_spot_idx - current_timestep
            wait_time = wait_steps * self.env.gap_seconds

            # Check if we can afford to wait and still finish on On-Demand afterward.
            if (od_time_needed + wait_time) < time_left:
                return ClusterType.NONE
            else:
                # Waiting is too risky. Make progress with On-Demand.
                return ClusterType.ON_DEMAND