import json
from argparse import Namespace
import math
import numpy as np

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A heuristic-based strategy for the Cant-Be-Late Multi-Region Scheduling Problem.

    The strategy operates in a few modes:
    1.  **Panic Mode**: If the remaining time to deadline is critically short, it
        switches to On-Demand instances to guarantee completion, as this is
        preferable to the large penalty for failure.
    2.  **Opportunistic Spot Mode**: It pre-processes spot availability traces for all
        regions to identify the best spot opportunities at any given time. The "best"
        opportunity is defined as the longest continuous block of future spot
        availability.
    3.  **Region Switching**: If a significantly better spot opportunity exists in another
        region, it will switch to that region, accepting the restart overhead,
        provided the future spot block is long enough to justify the switch cost.
    4.  **Wait vs. Work**: If no good spot opportunities are available, it calculates
        an "urgency" metric (ratio of work remaining to time remaining). If urgency
        is high, it uses On-Demand to make progress. If urgency is low (plenty of
        slack), it chooses to wait (NONE) to save costs and hope for a spot
        instance to become available later.

    All pre-computation is done in a lazy-initialized method to ensure the environment
    is available.
    """

    NAME = "heuristic_solver"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            self.config = json.load(f)

        args = Namespace(
            deadline_hours=float(self.config["deadline"]),
            task_duration_hours=[float(self.config["duration"])],
            restart_overhead_hours=[float(self.config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Defer initialization of env-dependent attributes until the first step
        self.initialized = False
        return self

    def _lazy_init(self):
        """
        Perform one-time initialization on the first call to _step,
        once self.env is available. This is for pre-computing lookup tables.
        """
        self.num_regions = self.env.get_num_regions()
        
        # Tunable parameters for the heuristic logic
        self.urgency_threshold = 0.80
        self.panic_buffer = 1.05
        self.min_block_to_switch = 2

        # Pre-computation of spot availability traces
        # Add a small buffer to handle potential off-by-one in step counting
        self.max_timesteps = int(math.ceil(self.deadline / self.env.gap_seconds)) + 5
        self.spot_availability = np.zeros((self.num_regions, self.max_timesteps), dtype=bool)

        trace_files = self.config["trace_files"]
        for r, trace_file in enumerate(trace_files):
            try:
                with open(trace_file) as f:
                    trace_data = [bool(int(line.strip())) for line in f]
                trace_len = len(trace_data)
                
                # Cap the trace at max_timesteps or pad if it's shorter
                if trace_len >= self.max_timesteps:
                    self.spot_availability[r, :] = trace_data[:self.max_timesteps]
                else:
                    self.spot_availability[r, :trace_len] = trace_data
                    # Assume spot is unavailable after the trace ends
                    self.spot_availability[r, trace_len:] = False
            except (IOError, ValueError):
                # Fallback in case of file issues: assume no spot availability
                self.spot_availability[r, :] = False

        # Pre-computation of future consecutive spot availability blocks
        # This allows O(1) lookup of how good a spot opportunity is.
        self.future_spot_block = np.zeros((self.num_regions, self.max_timesteps), dtype=int)
        for r in range(self.num_regions):
            if self.max_timesteps > 0 and self.spot_availability[r, self.max_timesteps - 1]:
                self.future_spot_block[r, self.max_timesteps - 1] = 1
            for t in range(self.max_timesteps - 2, -1, -1):
                if self.spot_availability[r, t]:
                    self.future_spot_block[r, t] = 1 + self.future_spot_block[r, t + 1]

        self.initialized = True

    def _get_time_step(self) -> int:
        """Calculate the current discrete time step from elapsed seconds."""
        # Using round to handle potential floating point inaccuracies
        return int(round(self.env.elapsed_seconds / self.env.gap_seconds))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.initialized:
            self._lazy_init()

        # 0. Check if the task is already completed
        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        if time_to_deadline <= 0:
            return ClusterType.NONE # No time left

        time_step = self._get_time_step()
        if time_step >= self.max_timesteps:
            # We are past our pre-computed data; act safely.
            return ClusterType.ON_DEMAND

        # 1. Panic Mode: If finishing on On-Demand is tight, use it immediately.
        od_restart_time = self.restart_overhead if last_cluster_type == ClusterType.SPOT else 0.0
        time_needed_od = remaining_work + od_restart_time

        if time_to_deadline <= time_needed_od * self.panic_buffer:
            return ClusterType.ON_DEMAND

        # 2. Normal Mode: Look for the best spot opportunity across all regions.
        best_spot_region = -1
        max_future_block = 0
        for r in range(self.num_regions):
            if self.spot_availability[r, time_step]:
                future_block = self.future_spot_block[r, time_step]
                if future_block > max_future_block:
                    max_future_block = future_block
                    best_spot_region = r
        
        # 3. Decision: Act on the best spot opportunity if it's good enough.
        if best_spot_region != -1:
            current_region = self.env.get_current_region()
            if best_spot_region == current_region:
                return ClusterType.SPOT
            else:
                # Heuristic: switch if the spot block is long enough to justify the restart cost
                if max_future_block >= self.min_block_to_switch:
                    self.env.switch_region(best_spot_region)
                    return ClusterType.SPOT
        
        # 4. No spot, or not worth switching: Decide between On-Demand and None.
        # This decision is based on the urgency, i.e., how much slack time is left.
        
        remaining_work_steps = math.ceil(remaining_work / self.env.gap_seconds)
        time_to_deadline_steps = math.floor(time_to_deadline / self.env.gap_seconds)

        if time_to_deadline_steps <= 0:
             return ClusterType.ON_DEMAND # Last chance to make progress

        urgency = remaining_work_steps / time_to_deadline_steps
        
        if urgency > self.urgency_threshold:
            # Slack is low, so we must make progress.
            return ClusterType.ON_DEMAND
        else:
            # We have enough slack, so wait for a better (cheaper) opportunity.
            return ClusterType.NONE