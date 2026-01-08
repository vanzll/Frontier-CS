import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "slack_guard_v1"

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
        # Internal state for efficiency and decision making
        self._od_committed = False
        self._last_done_len = 0
        self._total_done_time = 0.0
        return self

    def _update_progress_cache(self):
        # Incrementally update total done time to avoid O(n) per step
        current_len = len(self.task_done_time)
        if current_len > self._last_done_len:
            # Sum only the new entries
            delta = 0.0
            for i in range(self._last_done_len, current_len):
                delta += self.task_done_time[i]
            self._total_done_time += delta
            self._last_done_len = current_len

    def _remaining_work(self) -> float:
        self._update_progress_cache()
        remaining = self.task_duration - self._total_done_time
        return remaining if remaining > 0.0 else 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, never switch back.
        if self._od_committed or last_cluster_type == ClusterType.ON_DEMAND:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Compute remaining work and time
        remaining_work = self._remaining_work()
        # If finished, do nothing
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        # If no time left, attempt OD to minimize penalty (though too late)
        if time_left <= 0.0:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Conservative slack: how much time we can afford to "waste" with zero progress
        # before we must switch to OD and still finish by deadline.
        # We assume a one-time restart overhead when committing to OD.
        commit_overhead = self.restart_overhead
        slack_time = time_left - (remaining_work + commit_overhead)

        # We operate with step granularity: avoid taking a non-OD step if we can't
        # afford to lose a full step with zero progress.
        step = self.env.gap_seconds

        # If slack is less than one step, commit to OD to guarantee finishing.
        if slack_time < step:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Otherwise, use Spot if available; pause if not.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable: if we have ample slack (>= 1 step), pause to save cost.
        # We already checked slack_time >= step above.
        return ClusterType.NONE