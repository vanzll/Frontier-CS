import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that uses future spot trace information
    to make decisions.
    """

    NAME = "lookahead_planner"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        - Loads configuration.
        - Loads and pre-processes spot availability traces.
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

        # Load trace files.
        self.traces = []
        for trace_file in config["trace_files"]:
            with open(trace_file) as f:
                self.traces.append([int(line.strip()) for line in f])

        # Precompute prefix sums for fast range sum queries on traces.
        self.trace_prefix_sums = []
        for trace in self.traces:
            prefix_sum = [0] * (len(trace) + 1)
            for i in range(len(trace)):
                prefix_sum[i + 1] = prefix_sum[i] + trace[i]
            self.trace_prefix_sums.append(prefix_sum)
            
        return self

    def get_spot_steps_in_window(self, region_idx: int, start_step: int,
                                 window_size: int) -> int:
        """
        Calculates the number of available spot steps in a given window
        for a specific region using pre-computed prefix sums.
        """
        if start_step >= len(self.traces[region_idx]):
            return 0
        
        end_step = min(start_step + window_size, len(self.traces[region_idx]))
        prefix_sums = self.trace_prefix_sums[region_idx]
        return prefix_sums[end_step] - prefix_sums[start_step]

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Check if the task is already completed.
        work_remaining = self.task_duration - sum(self.task_done_time)
        if work_remaining <= 0:
            return ClusterType.NONE

        # 2. Calculate current state variables.
        time_to_deadline = self.deadline - self.env.elapsed_seconds
        current_step = int(self.env.elapsed_seconds / self.env.gap_seconds)
        current_region = self.env.get_current_region()

        # 3. Determine if we are in "panic mode".
        # This is the minimum time required to finish if we use on-demand
        # exclusively from now on, assuming one final restart might occur.
        time_needed_od_critical = work_remaining + self.restart_overhead
        
        # If the time left is less than this critical time plus a safety
        # buffer (e.g., one time step), we must use on-demand.
        if time_to_deadline <= time_needed_od_critical + self.env.gap_seconds:
            return ClusterType.ON_DEMAND

        # 4. If not in panic mode, apply the opportunistic strategy.
        # This is the available slack time we have over the critical OD path.
        slack_time = time_to_deadline - time_needed_od_critical

        # Find the best region by scoring them based on future spot availability.
        num_regions = self.env.get_num_regions()
        
        # Look ahead for a fixed period (e.g., 24 hours worth of steps).
        lookahead_horizon = int(24 * 3600 / self.env.gap_seconds)
        
        scores = [
            self.get_spot_steps_in_window(r, current_step + 1, lookahead_horizon)
            for r in range(num_regions)
        ]
        best_region = scores.index(max(scores))

        # Decide whether to switch to the best region.
        should_switch = (best_region != current_region and
                         scores[best_region] > scores[current_region] and
                         slack_time > self.restart_overhead)

        if should_switch:
            self.env.switch_region(best_region)
            # After switching, use on-demand for one step as a safe choice,
            # since spot availability in the new region is unknown for this step.
            return ClusterType.ON_DEMAND
        else:
            # Stay in the current region.
            if has_spot:
                # If spot is available, use it.
                return ClusterType.SPOT
            else:
                # No spot. Decide whether to wait or use on-demand.
                # We can afford to wait (use NONE) if our slack time is
                # greater than the duration of one time step.
                if slack_time > self.env.gap_seconds:
                    return ClusterType.NONE
                else:
                    # Not enough slack to wait, must use on-demand to make progress.
                    return ClusterType.ON_DEMAND