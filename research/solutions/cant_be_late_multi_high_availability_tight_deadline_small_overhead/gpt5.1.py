import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy aimed at minimizing cost while meeting deadline."""

    NAME = "cant_be_late_multi_region_robust"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Internal bookkeeping for efficient progress tracking.
        self._work_done = 0.0
        self._last_segments_len = 0
        self._committed_to_on_demand = False

        return self

    def _update_work_done(self) -> None:
        """Incrementally track total work done without summing the full list each step."""
        segments = self.task_done_time
        cur_len = len(segments)
        if cur_len > self._last_segments_len:
            # Environment appends at most one new segment per step.
            self._work_done += segments[-1]
            self._last_segments_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cumulative work done based on new segments from the last step.
        self._update_work_done()

        remaining_work = self.task_duration - self._work_done
        if remaining_work <= 0:
            # Task completed; no further computation needed.
            return ClusterType.NONE

        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_time <= 0:
            # Past deadline; further work only adds cost with no benefit.
            return ClusterType.NONE

        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        # Once we decide to rely on on-demand, never go back to spot to avoid extra risk.
        if self._committed_to_on_demand:
            return ClusterType.ON_DEMAND

        # Compute the minimum wall-clock time required to finish the remaining work
        # if we switch to on-demand after one more step that might yield zero progress
        # (e.g., due to spot failure or choosing to idle).
        min_on_demand_time_after_one_step = remaining_work + overhead

        # If taking one more non-progress step would make it impossible to finish
        # before the deadline using on-demand only, commit to on-demand now.
        if remaining_time <= min_on_demand_time_after_one_step + gap:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        # We have enough slack to take a "risky" step:
        # Prefer Spot when available to minimize cost.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available but still plenty of slack: idle to avoid expensive on-demand.
        return ClusterType.NONE