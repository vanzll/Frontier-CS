from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_strategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_policy(self):
        self._policy_initialized = True
        self._committed_to_od = False
        self._progress_done = 0.0
        self._last_task_done_index = 0

        # Fallback defaults in case attributes are missing
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = task_duration
        try:
            restart_overhead = float(self.restart_overhead)
        except Exception:
            restart_overhead = 0.0

        initial_slack = max(deadline - task_duration, 0.0)
        self.initial_slack = initial_slack
        self.H = restart_overhead

        if initial_slack <= 0.0:
            # No slack: always prefer on-demand to minimize risk.
            self.commit_threshold = restart_overhead
            self.safe_idle_slack = 0.0
            return

        H = restart_overhead

        # Commit threshold: when slack gets below this, permanently switch to on-demand.
        commit_from_H = 3.0 * H
        commit_from_frac = 0.05 * initial_slack  # 5% of slack
        self.commit_threshold = min(
            max(commit_from_H, commit_from_frac),
            0.5 * initial_slack,  # never more than 50% of initial slack
        )

        # Idle threshold: while slack is above this, we can afford to pause when spot is unavailable.
        idle_from_frac = 0.15 * initial_slack  # 15% of slack
        idle_from_commit = self.commit_threshold + 2.0 * H
        self.safe_idle_slack = min(
            max(idle_from_frac, idle_from_commit),
            0.8 * initial_slack,  # never more than 80% of initial slack
        )

    def _update_progress(self):
        # Incrementally track completed work to avoid re-summing the entire list.
        current_len = len(self.task_done_time)
        if current_len > self._last_task_done_index:
            new_segments = self.task_done_time[self._last_task_done_index:current_len]
            self._progress_done += sum(new_segments)
            self._last_task_done_index = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not getattr(self, "_policy_initialized", False):
            self._init_policy()

        # Update progress
        self._update_progress()
        remaining = max(self.task_duration - self._progress_done, 0.0)

        # If task is already finished, do nothing.
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        slack = time_left - remaining

        # If somehow behind schedule, avoid any further risk.
        if slack <= 0.0:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Permanent on-demand region.
        if getattr(self, "_committed_to_od", False):
            return ClusterType.ON_DEMAND

        # Trigger permanent on-demand if slack is too small.
        if slack <= self.commit_threshold:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Not yet committed to on-demand.
        if has_spot:
            # Always prefer spot when available in the non-committed region.
            return ClusterType.SPOT

        # No spot available.
        if slack > self.safe_idle_slack:
            # Plenty of slack left: we can afford to pause and wait for cheaper spot.
            return ClusterType.NONE
        else:
            # Slack is limited: use on-demand to keep up.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)