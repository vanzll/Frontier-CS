import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline guarantees."""
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

        # Internal state for efficient tracking of work done
        self._total_done_time = 0.0
        self._last_task_done_len = 0

        # Precompute a conservative maximum time for a "bad" step with no progress
        # Worst case: one gap interval plus one restart overhead
        # (Environment should not charge more than one restart per step.)
        self._dt_max = self.env.gap_seconds + self.restart_overhead

        return self

    def _update_total_done_time(self) -> None:
        """Incrementally maintain total done work to avoid O(n) sum each step."""
        cur_len = len(self.task_done_time)
        if cur_len != self._last_task_done_len:
            if cur_len > self._last_task_done_len:
                # Only sum new segments appended since last step
                new_segments = self.task_done_time[self._last_task_done_len:cur_len]
                if new_segments:
                    self._total_done_time += sum(new_segments)
            else:
                # In unexpected cases (e.g., env reset), recompute from scratch
                self._total_done_time = sum(self.task_done_time)
            self._last_task_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Update cached total done work
        self._update_total_done_time()

        # Compute remaining required work (seconds)
        remaining_work = self.task_duration - self._total_done_time
        if remaining_work <= 0.0:
            # Task finished: avoid any further cost
            return ClusterType.NONE

        now = self.env.elapsed_seconds

        # Check if we can afford one more potentially unproductive step
        # (e.g., using Spot or idling) and still be guaranteed to finish
        # by switching to On-Demand afterwards.
        #
        # Worst-case for this step: takes self._dt_max seconds and yields 0 work.
        # After that, we might need at most `restart_overhead + remaining_work`
        # additional seconds on On-Demand to finish.
        #
        # Guarantee condition:
        #   now + dt_max + restart_overhead + remaining_work <= deadline
        safe_to_gamble = (
            now + self._dt_max + self.restart_overhead + remaining_work <= self.deadline
        )

        if safe_to_gamble:
            # We have enough slack to risk Spot or to wait if Spot is unavailable.
            if has_spot:
                return ClusterType.SPOT
            # Spot unavailable but still safe: wait to avoid expensive On-Demand.
            return ClusterType.NONE

        # Slack is tight: must use On-Demand to ensure completion before deadline.
        return ClusterType.ON_DEMAND