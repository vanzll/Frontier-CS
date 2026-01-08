from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args: Any):
        super().__init__(args)
        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.force_on_demand = False
        self._task_progress = 0.0
        self._last_task_done_len = 0
        self._progress_is_duration = True

    def solve(self, spec_path: str) -> "Solution":
        # Reset internal state before a new evaluation run.
        self._reset_internal_state()
        return self

    def _update_progress(self) -> float:
        """Update and return conservative estimate of task progress in seconds."""
        # If we already determined task_done_time is not duration-based, progress stays 0.
        if not self._progress_is_duration:
            return 0.0

        cur_len = len(self.task_done_time)
        if cur_len > self._last_task_done_len:
            new_segments = self.task_done_time[self._last_task_done_len:cur_len]
            # Sum new segments as additional progress.
            self._task_progress += float(sum(new_segments))
            self._last_task_done_len = cur_len

            # If reported progress exceeds elapsed wall time, task_done_time entries
            # are likely not durations (e.g., timestamps). Fall back to zero progress.
            elapsed = getattr(self.env, "elapsed_seconds", 0.0)
            if self._task_progress > elapsed + 1e-6:
                self._progress_is_duration = False
                self._task_progress = 0.0

        if not self._progress_is_duration:
            return 0.0

        # Clamp to [0, task_duration]
        if self._task_progress < 0.0:
            self._task_progress = 0.0
        if self._task_progress > self.task_duration:
            self._task_progress = self.task_duration

        return self._task_progress

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, stay on-demand.
        if self.force_on_demand:
            return ClusterType.ON_DEMAND

        # Update estimate of work done.
        done = self._update_progress()
        remaining_work = self.task_duration - done

        # If no remaining work is estimated, pause (no cost).
        if remaining_work <= 0.0:
            return ClusterType.NONE

        t = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        overhead = self.restart_overhead

        time_left = deadline - t

        # If we're at or past the deadline, just use on-demand (nothing else helps).
        if time_left <= 0.0:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Ensure there's at least enough time left to finish with on-demand if we commit now.
        # time_left >= remaining_work + overhead is required for a guaranteed finish.
        if time_left < remaining_work + overhead:
            # We are already in a risky region; switch to on-demand immediately.
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Check if it is still safe to wait one more step without committing to on-demand.
        # After one more step with *no progress*, time_left_future = time_left - gap.
        # We need time_left_future >= remaining_work + overhead to still be safe.
        if time_left - gap < remaining_work + overhead:
            # Cannot safely delay any longer; commit to on-demand from now on.
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Safe region: use spot when available; otherwise, wait (NONE) to save cost.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)