from typing import Any, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args: Any):
        super().__init__(args)
        self._episode_initialized: bool = False
        self._last_elapsed_seconds: float = -1.0
        self._task_done_cache: float = 0.0
        self._last_task_done_idx: int = 0
        self._committed_to_od: bool = False
        self._safety_margin: float = 0.0

    def solve(self, spec_path: str) -> "Solution":
        # Optional global initialization based on spec_path if needed.
        return self

    def _reset_episode_state(self) -> None:
        """Reset per-episode state when a new environment run starts."""
        self._episode_initialized = True
        self._task_done_cache = 0.0
        self._last_task_done_idx = 0
        self._committed_to_od = False

        # Compute a safety margin (seconds) based on problem parameters.
        # Margin is chosen to be:
        #   - at least several restart_overheads / gaps
        #   - at most a fraction of total slack
        #   - non-negative
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        try:
            restart_overhead = float(self.restart_overhead)
        except Exception:
            restart_overhead = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        total_slack = max(deadline - task_duration, 0.0)

        base_min_margin = max(3.0 * restart_overhead, 5.0 * gap)
        # Cap margin to at most 25% of total slack to avoid over-conservatism.
        if total_slack > 0.0:
            cap_margin = 0.25 * total_slack
            self._safety_margin = min(base_min_margin, cap_margin)
        else:
            # If no slack, just use a small margin equal to max(overhead, gap).
            self._safety_margin = max(restart_overhead, gap)

        if self._safety_margin < 0.0:
            self._safety_margin = 0.0

    def _update_progress_cache(self) -> None:
        """Incrementally maintain cached total work done in seconds."""
        # task_done_time is a list of completed work segments (in seconds).
        # We maintain a running sum to avoid O(n^2) summing.
        segments = self.task_done_time
        if not isinstance(segments, list):
            # Fallback: try to coerce to list if it's another iterable.
            try:
                segments = list(segments)
            except Exception:
                segments = []
        current_len = len(segments)
        if current_len > self._last_task_done_idx:
            new_segments = segments[self._last_task_done_idx : current_len]
            try:
                increment = float(sum(new_segments))
            except Exception:
                increment = 0.0
            self._task_done_cache += increment
            self._last_task_done_idx = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Detect new episode (env reset) by elapsed_seconds decreasing.
        current_time = float(self.env.elapsed_seconds)
        if (not self._episode_initialized) or (current_time < self._last_elapsed_seconds):
            self._reset_episode_state()
        self._last_elapsed_seconds = current_time

        # Update cached work progress.
        self._update_progress_cache()

        # Compute remaining work and time to deadline.
        total_task = float(self.task_duration)
        work_done = self._task_done_cache
        remaining_work = max(total_task - work_done, 0.0)

        if remaining_work <= 0.0:
            # Task completed; no need to run more.
            self._committed_to_od = True
            return ClusterType.NONE

        deadline = float(self.deadline)
        time_to_deadline = deadline - current_time

        # If deadline already passed, still run on-demand to minimize lateness.
        if time_to_deadline <= 0.0:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        gap = float(self.env.gap_seconds)
        restart_overhead = float(self.restart_overhead)
        safety_margin = float(self._safety_margin)

        # Once committed to on-demand, always stay on on-demand.
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Compute slack if we switched to on-demand *after* one more risky step
        # (either SPOT or NONE), assuming worst-case zero progress this step.
        # slack_after_risk = remaining time after this step minus time needed
        # for remaining work plus one restart overhead.
        time_to_deadline_after_risk = time_to_deadline - gap
        slack_after_risk = time_to_deadline_after_risk - (remaining_work + restart_overhead)

        # If even committing now cannot meet the deadline with on-demand +
        # one overhead, just go on-demand (best-effort).
        if time_to_deadline < remaining_work + restart_overhead:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Decide action:
        # - If we have enough slack_after_risk >= safety_margin, we can afford
        #   to "gamble" one more step: use SPOT if available, otherwise idle.
        # - Otherwise, we must commit to ON_DEMAND now.
        if slack_after_risk >= safety_margin:
            if has_spot:
                # Use spot while we still have sufficient slack buffer.
                return ClusterType.SPOT
            else:
                # Spot unavailable; if we can still safely wait one step, idle to
                # save cost and hope spot returns.
                return ClusterType.NONE
        else:
            # Not enough slack to risk another non-on-demand step.
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser) -> "Solution":
        args, _ = parser.parse_known_args()
        return cls(args)