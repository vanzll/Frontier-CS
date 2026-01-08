from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_solution_v1"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_policy(self):
        self._policy_initialized = True

        self._gap = float(getattr(self.env, "gap_seconds", 1.0))
        self._restart = float(getattr(self, "restart_overhead", 0.0))

        if self._gap <= 0.0:
            self._gap = 1.0
        if self._restart < 0.0:
            self._restart = 0.0

        self._initial_task_duration = float(self.task_duration)
        self._initial_deadline = float(self.deadline)
        self._initial_slack = max(self._initial_deadline - self._initial_task_duration, 0.0)

        # Track cumulative completed work incrementally.
        self._done_sum = 0.0
        self._last_done_index = 0

        if self._initial_slack <= 0.0:
            # No slack: be maximally conservative.
            self.commit_slack = 0.0
            self.idle_slack = 0.0
            return

        safety_margin = 4.0 * (self._gap + self._restart)

        # Threshold at which we permanently switch to on-demand.
        commit_fraction = 0.2
        commit = commit_fraction * self._initial_slack
        if commit < safety_margin:
            commit = safety_margin
        max_commit = 0.8 * self._initial_slack
        if commit > max_commit:
            commit = max_commit
        if commit < 0.0:
            commit = 0.0

        # Threshold above which we are willing to idle waiting for spot.
        idle_fraction = 0.6
        idle = idle_fraction * self._initial_slack
        min_idle = commit + safety_margin
        if idle < min_idle:
            idle = min_idle
        if idle > self._initial_slack:
            idle = self._initial_slack
        if idle < commit:
            idle = commit

        self.commit_slack = commit
        self.idle_slack = idle

    def _update_done_sum(self):
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return
        n = len(segments)
        if n > self._last_done_index:
            new_sum = 0.0
            for i in range(self._last_done_index, n):
                new_sum += float(segments[i])
            self._done_sum += new_sum
            self._last_done_index = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not getattr(self, "_policy_initialized", False):
            self._initialize_policy()

        self._update_done_sum()
        remaining = max(float(self.task_duration) - self._done_sum, 0.0)

        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        time_to_deadline = max(deadline - elapsed, 0.0)
        slack = time_to_deadline - remaining

        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        if self._initial_slack <= 0.0 or self.commit_slack <= 0.0:
            return ClusterType.ON_DEMAND

        if slack <= self.commit_slack:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        if slack > self.idle_slack:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)