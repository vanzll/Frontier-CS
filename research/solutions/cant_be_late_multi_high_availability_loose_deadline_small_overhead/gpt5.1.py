import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy focusing on Spot with safe On-Demand fallback."""

    NAME = "cant_be_late_multi_region_v1"

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

        # Internal state
        self.commit_to_on_demand = False
        self._commit_margin_computed = False
        self.commit_margin = 0.0

        # Efficient tracking of accumulated work without re-summing the whole list each step
        self._accumulated_work = 0.0
        self._last_tdt_len = 0

        return self

    def _compute_commit_margin_if_needed(self):
        """Compute safety margin (in seconds) for switching to On-Demand."""
        if not self._commit_margin_computed:
            # gap_seconds and restart_overhead are in seconds
            gap = float(getattr(self.env, "gap_seconds", 1.0))
            overhead = float(self.restart_overhead)
            # Safety margin: large enough to cover worst-case drop per step and some extra.
            # This is conservative but still small relative to the total slack.
            self.commit_margin = 10.0 * max(gap, overhead)
            self._commit_margin_computed = True

    def _update_accumulated_work(self):
        """Update cached sum of task_done_time to avoid O(n) per step."""
        tdt = self.task_done_time
        if not tdt:
            return
        current_len = len(tdt)
        if current_len > self._last_tdt_len:
            new_sum = 0.0
            for i in range(self._last_tdt_len, current_len):
                new_sum += tdt[i]
            self._accumulated_work += new_sum
            self._last_tdt_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Strategy:
        - Prefer Spot instances whenever available.
        - If Spot is unavailable and we're not close to the deadline, pause (NONE).
        - As the deadline approaches, switch to On-Demand and stick with it,
          ensuring we always finish before the deadline.
        """
        # Ensure commit margin computed when env is ready
        self._compute_commit_margin_if_needed()

        # Update progress tracking
        self._update_accumulated_work()

        # Remaining work (seconds)
        remaining_work = float(self.task_duration) - self._accumulated_work
        if remaining_work <= 0.0:
            # Task is complete; do not incur further cost
            self.commit_to_on_demand = False
            return ClusterType.NONE

        # Time left until deadline (seconds)
        time_left = float(self.deadline) - float(self.env.elapsed_seconds)

        # If somehow at/after deadline, run On-Demand to minimize further loss
        if time_left <= 0.0:
            self.commit_to_on_demand = True
            return ClusterType.ON_DEMAND

        # If not yet committed to On-Demand, decide whether to commit now
        if not self.commit_to_on_demand:
            # Minimal time needed to finish if we switch to On-Demand now:
            # one restart overhead plus remaining work.
            od_time_needed = remaining_work + float(self.restart_overhead)

            # Commit when we're within a safety margin of the latest safe switch time.
            # Also commit if it's already impossible to finish on Spot alone.
            if time_left <= od_time_needed + self.commit_margin or time_left <= od_time_needed:
                self.commit_to_on_demand = True

        # Once committed, always use On-Demand
        if self.commit_to_on_demand:
            return ClusterType.ON_DEMAND

        # Not committed yet: use Spot when available, otherwise pause (NONE) to save cost.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE