from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cb_late_threshold_safe_v1"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_policy(self):
        if getattr(self, "_policy_initialized", False):
            return
        self._policy_initialized = True

        total_slack = 0.0
        try:
            if hasattr(self, "deadline") and hasattr(self, "task_duration"):
                total_slack = max(self.deadline - self.task_duration, 0.0)
        except Exception:
            total_slack = 0.0

        H = getattr(self, "restart_overhead", 0.0) or 0.0

        if total_slack <= 0.0:
            margin = max(3.0 * H, 0.0)
        else:
            margin_fraction = 0.15
            margin_min = 6.0 * H
            margin_max = 0.5 * total_slack
            margin = max(margin_min, margin_fraction * total_slack)
            margin = min(margin, margin_max)
        self._margin_time = margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_policy()

        dt = getattr(self.env, "gap_seconds", 0.0)
        H = getattr(self, "restart_overhead", 0.0) or 0.0
        margin = getattr(self, "_margin_time", 0.0)

        work_done = 0.0
        if getattr(self, "task_done_time", None):
            work_done = float(sum(self.task_done_time))

        remaining_work = max(self.task_duration - work_done, 0.0)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds

        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        slack = time_left - remaining_work

        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        threshold = dt + H + margin
        safe_to_gamble = slack > threshold

        if safe_to_gamble:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)