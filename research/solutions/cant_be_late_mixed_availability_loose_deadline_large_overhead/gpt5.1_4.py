from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state; spec_path ignored for now.
        self._policy_initialized = False
        self._committed_to_od = False
        return self

    def _init_policy(self):
        # Initialize policy parameters once we have access to the environment.
        initial_slack = max(self.deadline - self.task_duration, 0.0)
        gap = float(self.env.gap_seconds)
        overhead = float(self.restart_overhead)

        # Base critical slack: leave multiple overhead+gap margins.
        base_crit_slack = 4.0 * (overhead + gap)

        # Ensure the threshold is:
        # - Not smaller than a safe minimum.
        # - Not larger than a large fraction of the initial slack to avoid
        #   committing to on-demand from the very beginning.
        min_crit_slack = overhead + gap
        max_crit_slack = 0.9 * initial_slack if initial_slack > 0 else base_crit_slack

        crit_slack = base_crit_slack
        if crit_slack < min_crit_slack:
            crit_slack = min_crit_slack
        if crit_slack > max_crit_slack:
            crit_slack = max_crit_slack

        self._critical_slack = crit_slack
        if not hasattr(self, "_committed_to_od"):
            self._committed_to_od = False
        self._policy_initialized = True

    def _remaining_compute_and_slack(self):
        # Compute remaining compute time and slack (all in seconds).
        work_done = 0.0
        if self.task_done_time:
            work_done = float(sum(self.task_done_time))

        remaining_compute = max(self.task_duration - work_done, 0.0)
        time_left = max(self.deadline - self.env.elapsed_seconds, 0.0)
        slack = time_left - remaining_compute
        return remaining_compute, time_left, slack

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize policy on first step, when env is available.
        if not getattr(self, "_policy_initialized", False):
            self._init_policy()

        remaining_compute, time_left, slack = self._remaining_compute_and_slack()

        # If task already complete or no time left, do nothing.
        if remaining_compute <= 0.0 or time_left <= 0.0:
            return ClusterType.NONE

        # If we haven't committed yet, check if it's time to fall back to OD.
        if not self._committed_to_od:
            if slack <= self._critical_slack:
                self._committed_to_od = True

        # Once committed, always use on-demand to guarantee completion.
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Risk-taking phase: use spot when available, otherwise pause.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)